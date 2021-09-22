//===- BlackboxMemory.cpp - Create modules for memory -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Transform memory operations in to instances of external modules for memory
// generators.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OperationSupport.h"
#include "set"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

using namespace circt;
using namespace firrtl;

/// Compute a hash code for a MemOp. This specialized hash ignores the naming of
/// the memory and ports.
llvm::hash_code computeHash(MemOp op) {
  // MemOp attributes
  llvm::hash_code hash =
      llvm::hash_combine(op.readLatencyAttr(), op.writeLatencyAttr());
  hash = llvm::hash_combine(hash, op.depthAttr());
  hash = llvm::hash_combine(hash, op.ruwAttr());

  // Result Types
  return llvm::hash_combine(hash, op->getResultTypes());
}

/// Compare memory operations for equivalence.  Only compares the types of the
/// memory and not the name or the memory, or the ports.
static bool isEquivalentTo(MemOp lhs, MemOp rhs) {
  if (lhs == rhs)
    return true;
  // Compare attributes.
  if (lhs.readLatencyAttr() != rhs.readLatencyAttr())
    return false;
  if (lhs.writeLatencyAttr() != rhs.writeLatencyAttr())
    return false;
  if (lhs.depthAttr() != rhs.depthAttr())
    return false;
  if (lhs.ruwAttr() != rhs.ruwAttr())
    return false;
  // Compare result types.  Taken from operation equivalence.
  if (lhs->getResultTypes() != rhs->getResultTypes())
    return false;
  return true;
}

namespace {
struct MemOpInfo : public llvm::DenseMapInfo<MemOp> {
  static MemOp getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return MemOp::getFromOpaquePointer(pointer);
  }
  static MemOp getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return MemOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(MemOp op) { return computeHash(op); }
  static bool isEqual(MemOp lhs, MemOp rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return isEquivalentTo(lhs, rhs);
  }
};
} // end anonymous namespace

/// Create an instance of a Module using the module name and the port list.
static InstanceOp createInstance(OpBuilder builder, Location loc,
                                 StringRef moduleName, StringAttr instanceName,
                                 ArrayRef<PortInfo> modulePorts) {
  // Make a bundle of the inputs and outputs of the specified module.
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(modulePorts.size());
  for (auto port : modulePorts)
    resultTypes.push_back(port.type);

  return builder.create<InstanceOp>(loc, resultTypes, moduleName,
                                    instanceName.getValue());
}

/// Get the pohwist for an external module representing a blackbox memory. This
/// external module must be compatible with the modules which are generated for
/// `vlsi_mem_gen`, which can be found in the rocketchip project.
static void getBlackBoxPortsForMemOp(MemOp op,
                                     ArrayRef<MemOp::NamedPort> memPorts,
                                     SmallVectorImpl<PortInfo> &extPorts,
                                     bool flattenVector) {
  OpBuilder builder(op);
  unsigned readPorts = 0;
  unsigned writePorts = 0;
  unsigned readWritePorts = 0;
  for (size_t i = 0, e = memPorts.size(); i != e; ++i) {
    // Calculate the naming prefix to use based on the kind of the port.
    std::string prefix;
    switch (memPorts[i].second) {
    case MemOp::PortKind::Read:
      prefix = ("R" + Twine(readPorts++) + "_").str();
      break;
    case MemOp::PortKind::Write:
      prefix = ("W" + Twine(writePorts++) + "_").str();
      break;
    case MemOp::PortKind::ReadWrite:
      prefix = ("RW" + Twine(readWritePorts++) + "_").str();
      break;
    }
    // Flatten the bundle representing a memory port, name-mangling and adding
    // every field in the bundle to the exter module's port list.  All memory
    // ports have an outer flip, so we just strip this.
    auto type = op.getResult(i).getType();
    for (auto bundleElement : type.cast<BundleType>().getElements()) {
      auto name = (prefix + bundleElement.name.getValue()).str();
      auto type = bundleElement.type;
      auto direction = Direction::In;
      if (bundleElement.isFlip)
        direction = Direction::Out;
      // If data or mask field is of vector type, then merge all elements.
      // This is to ensure LowerTypes does not create multiple ports after
      // lowering the vector.
      if (flattenVector)
        if (auto vecType = type.dyn_cast_or_null<FVectorType>()) {
          auto elemType = vecType.getElementType();
          auto width = elemType.getBitWidthOrSentinel();
          if (width >= 1)
            type =
                IntType::get(builder.getContext(), elemType.isSignedInteger(),
                             vecType.getNumElements() * width);
        }
      extPorts.push_back(
          {builder.getStringAttr(name), type, direction, op.getLoc()});
    }
  }
}

/// Create an external module black box representing the memory operation.
/// Returns the port list of the external module.
static FExtModuleOp createBlackboxModuleForMem(MemOp op,
                                               ArrayRef<PortInfo> extPorts) {

  OpBuilder builder(op->getContext());

  // The module's name is the name of the memory postfixed with "_ext".
  auto memName = op.name().str();
  if (memName.empty())
    memName = "mem";
  std::string extName = memName + "_ext";

  // Create the black box external module.
  auto extModuleOp = builder.create<FExtModuleOp>(
      op.getLoc(), builder.getStringAttr(extName), extPorts);

  // Insert the external module into the circuit.  This will rename the
  // external module if there is a conflict with another module name.
  auto circuitOp = op->getParentOfType<CircuitOp>();
  SymbolTable symbolTable(circuitOp);
  symbolTable.insert(extModuleOp);

  // Move the external module to the beginning of the circuitOp.  Inserting into
  // the symbol table may have moved the operation to the end of the circuit.
  extModuleOp->moveBefore(&circuitOp.front());

  // Add the memory parameters as attributes on the external module.  These will
  // be used by the generator tools to create memory configuration files, or
  // create the memory itself.  We use a named attr list to avoid creating lots
  // of intermediate dictionary attributes.
  NamedAttrList attrs(extModuleOp->getAttrDictionary());
  attrs.set("generator", builder.getStringAttr("FIRRTLMemory"));
  attrs.set("readLatency", op.readLatencyAttr());
  attrs.set("writeLatency", op.writeLatencyAttr());
  attrs.set("depth", op.depthAttr());
  attrs.set("ruw", op.ruwAttr());
  extModuleOp->setAttrs(attrs.getDictionary(builder.getContext()));

  return extModuleOp;
}

/// Create a regular module to wrap the external module.  The wrapper module
/// instantiates the external module, and connects all the inputs and outputs
/// together. This module will match the bundle return type of the memory op,
/// and connects to the flattened parameters of the external module. This is
/// done for compatibility with the Scala FIRRTL compiler and it is unclear if
/// this will be needed in the long run.
static FModuleOp createWrapperModule(MemOp op,
                                     ArrayRef<MemOp::NamedPort> memPorts,
                                     ModuleOp memModuleOp,
                                     ArrayRef<PortInfo> extPorts,
                                     SmallVectorImpl<PortInfo> &modPorts, bool flattenVector) {
  OpBuilder builder(op->getContext());

  // The wrapper module's name is the name of the memory.
  auto memName = op.name();
  if (memName.empty())
    memName = "mem";

  // Create a wrapper module with the same type as the memory
  modPorts.reserve(op.getResults().size());
  for (size_t i = 0, e = memPorts.size(); i != e; ++i) {
    auto name = op.getPortName(i);
    auto type = op.getPortType(i);
    modPorts.push_back({name, type, Direction::In, op.getLoc()});
  }
  auto moduleOp = builder.create<FModuleOp>(
      op.getLoc(), builder.getStringAttr(memName), modPorts);

  // Insert the externaml module into the circuit.  This will rename the module
  // if there is a conflict with another module name.
  auto circuitOp = op->getParentOfType<CircuitOp>();
  SymbolTable symbolTable(circuitOp);
  symbolTable.insert(moduleOp);

  // Create the module
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());
  FIRRTLType memDataType = op.getDataType();
  unsigned maskBits = 1;
    if (flattenVector) {
      if (auto vecType = op.getDataType().dyn_cast<FVectorType>()){
          auto elemType = vecType.getElementType();
          auto width = elemType.getBitWidthOrSentinel();
          if (width >= 1) {
            memDataType =
                IntType::get(builder.getContext(), elemType.isSignedInteger(),
                             vecType.getNumElements() * width);
            maskBits = vecType.getNumElements();
          }
      }
    }
  auto oldPorts = op.getPorts();
  SmallVector<Type, 8> ports;
  for (size_t portIdx = 0, e = oldPorts.size(); portIdx < e; ++portIdx) {
    auto port = oldPorts[portIdx];
    
    ports.push_back(
        MemOp::getTypeForPort(op.depth(), memDataType, port.second, maskBits));
  }
  auto newMemOp = builder.create<MemOp>(
      op.getLoc(), ports, op.readLatency(), op.writeLatency(), op.depth(),
      op.ruw(), op.portNames().getValue(), op.name().str(),
      op.annotations().getValue(), op.portAnnotations().getValue());

  // Connect the ports between the memory module and the instance of the black
  // box memory module. The outer module has a single bundle representing each
  // Memory port, while the inner module has a separate field for each memory
  // port bundle in the memory bundle.
  for (auto memPort : llvm::enumerate(moduleOp.getArguments())) {
    //builder.create<ConnectOp>(op.getLoc(), newMemOp.getResult(memPort.index()),memPort.value());
    auto memPortType = memPort.value().getType().cast<FIRRTLType>();
    auto newMemRes = newMemOp.getResult(memPort.index());
    for (auto field :
         llvm::enumerate(memPortType.cast<BundleType>().getElements())) {
      auto fieldValue =
          builder.create<SubfieldOp>(op.getLoc(), memPort.value(), field.index());
      auto memResField = builder.create<SubfieldOp>(op.getLoc(), newMemRes, field.index());
      auto fieldType = field.value().type.cast<FIRRTLType>();
      bool isReadPort = field.value().isFlip;
      // Flatten the vector, only if the elements of the vector are of simple
      // ground type with known bitwidth.
      if (fieldType.isa<FVectorType>() && flattenVector &&
          fieldType.dyn_cast<FVectorType>()
                  .getElementType()
                  .getBitWidthOrSentinel() > 0) {
        // For read ports extract bits from the memory read port and distribute
        // them over the vector elements. Example for an 8 bit vector element,
        // wrapper_readPort[0] = Mem_readPort[7:0];
        // wrapper_readPort[1] = Mem_readPort[15:8];
        // ...
        // For write ports, concat all the vector elements and assign to the
        // write port. Example for a vector with 4 elements. Mem_writePort =
        // {wrapper_writePort[3],wrapper_writePort[2],wrapper_writePort[1],wrapper_writePort[0]}
        // Concat all mask bits to create a single multibit mask.
        auto fVecType = fieldType.dyn_cast<FVectorType>();
        auto elemWidth = fVecType.getElementType().getBitWidthOrSentinel();
        WireOp lastIndexWire;
        // Map vector elements to memory fields.
        // Read data is mapped to vector elements using bit extraction.
        // Write data is mapped from vector elements using bit concat.
        // Mask data is mapped from vector elements using bit concat.
        for (size_t vecI = 0, numElems = fVecType.getNumElements();
             vecI < numElems; ++vecI) {
          auto vI = builder.create<SubindexOp>(op.getLoc(), fieldValue, vecI);
          if (isReadPort) {
            auto extractBits = builder.create<BitsPrimOp>(
                op.getLoc(), memResField, (vecI * elemWidth + elemWidth - 1),
                (vecI * elemWidth));
            builder.create<ConnectOp>(op.getLoc(), vI, extractBits);
          } else {
            FIRRTLType wireType;
            CatPrimOp catOp;
            if (vecI == 0)
              wireType = fVecType.getElementType();
            else {
              catOp = builder.create<CatPrimOp>(op.getLoc(), vI, lastIndexWire);
              wireType = catOp.getType();
            }
            auto writeCatWire = builder.create<WireOp>(op.getLoc(), wireType);
            if (vecI == 0)
              builder.create<ConnectOp>(op.getLoc(), writeCatWire, vI);
            else
              builder.create<ConnectOp>(op.getLoc(), writeCatWire, catOp);
            lastIndexWire = writeCatWire;
          }
        }
        if (!isReadPort)
          builder.create<ConnectOp>(op.getLoc(), memResField, lastIndexWire);
      } else {
        if (!isReadPort)
          builder.create<ConnectOp>(op.getLoc(), memResField, fieldValue);
        else
          builder.create<ConnectOp>(op.getLoc(), fieldValue, memResField);
      }
    }
  }

  return moduleOp;
}

/// Create a bundle wire for each memory port.  This takes all the individual
/// fields returned from instantiating the external module, and wraps them in to
/// bundles to match the memory return type.
static void createWiresForMemoryPorts(OpBuilder builder, Location loc, MemOp op,
                                      InstanceOp instanceOp,
                                      ArrayRef<PortInfo> extPorts,
                                      SmallVectorImpl<Value> &results) {

  auto extResultIt = instanceOp.result_begin();

  for (auto memPort : op.getResults()) {
    // Create  a wire bundle for each memory port
    auto wireOp = builder.create<WireOp>(loc, memPort.getType());
    results.push_back(wireOp.getResult());

    // Connect each wire to the corresponding ports in the external module
    auto wireBundle = memPort.getType().cast<FIRRTLType>();
    for (auto field :
         llvm::enumerate(wireBundle.cast<BundleType>().getElements())) {
      auto fieldValue =
          builder.create<SubfieldOp>(op.getLoc(), wireOp, field.index());
      // Create the connection between module arguments and the external module,
      // making sure that sinks are on the LHS
      if (field.value().isFlip)
        builder.create<ConnectOp>(op.getLoc(), fieldValue, *extResultIt);
      else
        builder.create<ConnectOp>(op.getLoc(), *extResultIt, fieldValue);
      // advance the external module field iterator
      ++extResultIt;
    }
  }
}

static void
replaceMemWithWrapperModule(DenseMap<MemOp, FModuleOp, MemOpInfo> &knownMems,
                            MemOp memOp, std::set<FModuleOp> &newMods, bool flattenVector) {

  // The module we will be replacing the MemOp with.  If we don't have a
  // suitable memory module already created, a new one representing the memory
  // will be created.
  FModuleOp moduleOp;
  SmallVector<PortInfo, 2> modPorts;

  // Check if we have already created a suitable wrapper module. If we have not
  // seen a similar memory, create a new wrapper module.
  auto it = knownMems.find(memOp);
  auto found = it != knownMems.end();
  if (found) {
    // Create an instance of the wrapping module.  We have to retrieve the
    // module port information back from the module.
    moduleOp = it->second;
    modPorts = moduleOp.getPorts();
  } else {
    // Get the memory port descriptors. This gives us the name and kind of each
    // memory port created by the MemOp.
    auto memPorts = memOp.getPorts();

    // Get the pohwist for a module which represents the black box memory.
    // Typically has 1R + 1W memory port, which has 4+5=9 fields.
    SmallVector<PortInfo, 9> extPortList;
    getBlackBoxPortsForMemOp(memOp, memPorts, extPortList, flattenVector);
    auto memMod = memOp->getParentOfType<ModuleOp>(); // createBlackboxModuleForMem(memOp,
                                                      // extPortList);
    moduleOp =
        createWrapperModule(memOp, memPorts, memMod, extPortList, modPorts, flattenVector);
    knownMems[memOp] = moduleOp;
    newMods.insert(moduleOp);
  }

  // Create an instance of the wrapping module
  auto instanceOp =
      createInstance(OpBuilder(memOp), memOp.getLoc(), moduleOp.getName(),
                     memOp.nameAttr(), modPorts);

  // Replace the memory operation with the module instance
  memOp.replaceAllUsesWith(instanceOp.getResults());

  // If we are using the memory operation as a key in the map, we cannot delete
  // it yet.
  if (found)
    memOp->erase();
}

/// Replaces all memories which match the predicate function with external
/// modules and simple wrapper module which instantiates it.  Returns true if
/// any memories were replaced.
static bool
replaceMemsWithWrapperModules(CircuitOp circuit,
                              function_ref<bool(MemOp)> shouldReplace,
                              bool flattenVector) {
  /// A set of replaced memory operations.  When two memory operations
  /// share the same types, they can share the same modules.
  DenseMap<MemOp, FModuleOp, MemOpInfo> knownMems;
  std::set<FModuleOp> newMods;
  auto mods = circuit.getOps<FModuleOp>();
  for (auto fmodule : mods) {
    if (newMods.count(fmodule))
      continue;
    for (auto memOp : llvm::make_early_inc_range(fmodule.getOps<MemOp>())) {
      if (shouldReplace(memOp)) {
        replaceMemWithWrapperModule(knownMems, memOp, newMods, flattenVector);
      }
    }
  }
  // Erase any of the remaining memory operations.  These were kept around
  // to check if other memory operations were equivalent.
  for (auto it : knownMems)
    it.first.erase();

  // Return true if something changed.
  return !knownMems.empty();
}

static void
replaceMemWithExtModule(DenseMap<MemOp, FExtModuleOp, MemOpInfo> &knownMems,
                        MemOp memOp) {

  FExtModuleOp extModuleOp;
  SmallVector<PortInfo, 9> extPortList;

  auto it = knownMems.find(memOp);
  auto found = it != knownMems.end();
  if (found) {
    // Create an instance of the wrapping module.  We have to retrieve the
    // module port information back from the module.
    extModuleOp = it->second;
    extPortList = extModuleOp.getPorts();
  } else {
    // Get the memory port descriptors.  This gives us the name and kind of each
    // memory port created by the MemOp.
    auto memPorts = memOp.getPorts();

    // Get the pohwist for a module which represents the black box memory.
    // Typically has 1R + 1W memory port, which has 4+5=9 fields.
    getBlackBoxPortsForMemOp(memOp, memPorts, extPortList, false);
    extModuleOp = createBlackboxModuleForMem(memOp, extPortList);
    knownMems[memOp] = extModuleOp;
  }

  OpBuilder builder(memOp);

  // Create an instance of the black box module
  auto instanceOp =
      createInstance(builder, memOp.getLoc(), extModuleOp.getName(),
                     memOp.nameAttr(), extPortList);

  // Create a wire for every memory port
  SmallVector<Value, 2> results;
  results.reserve(memOp.getNumResults());
  createWiresForMemoryPorts(builder, memOp.getLoc(), memOp, instanceOp,
                            extPortList, results);

  // Replace each memory port with a wire
  memOp.replaceAllUsesWith(results);

  // If we are using the memory operation as a key in the map, we cannot delete
  // it yet.
  if (found)
    memOp->erase();
}

/// Replaces all memories which match the predicate function with external
/// modules.  Returns true if any modules were replaced.
static bool replaceMemsWithExtModules(CircuitOp circuit,
                                      function_ref<bool(MemOp)> shouldReplace) {
  /// A set of replaced memory operations.  When two memory operations
  /// share the same types, they can share the same modules.
  DenseMap<MemOp, FExtModuleOp, MemOpInfo> knownMems;
  for (auto fmodule : circuit.getOps<FModuleOp>()) {
    for (auto memOp : llvm::make_early_inc_range(fmodule.getOps<MemOp>())) {
      if (shouldReplace(memOp)) {
        replaceMemWithExtModule(knownMems, memOp);
      }
    }
  }
  // Erase any of the remaining memory operations.  These were kept around
  // to check if other memory operations were equivalent.
  for (auto op : knownMems)
    op.first.erase();

  // Return true if something changed.
  return !knownMems.empty();
}

namespace {
struct BlackBoxMemoryPass : public BlackBoxMemoryBase<BlackBoxMemoryPass> {
  void runOnOperation() override {
    // A memory must have read and write latencies of 1 in order to be
    // blackboxed. In the future this will probably be configurable.
    auto shouldReplace = [](MemOp memOp) -> bool {
      return memOp.readLatency() == 1 && memOp.writeLatency() == 1;
    };
    auto anythingChanged = false;
    if (emitWrapper)
      anythingChanged =
          replaceMemsWithWrapperModules(getOperation(), shouldReplace, flattenVector);
    else
      anythingChanged =
          replaceMemsWithExtModules(getOperation(), shouldReplace);

    if (!anythingChanged)
      markAllAnalysesPreserved();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createBlackBoxMemoryPass() {
  return std::make_unique<BlackBoxMemoryPass>();
}
