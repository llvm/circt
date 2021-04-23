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

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Twine.h"

using namespace circt;
using namespace firrtl;

using MemoryPortList = SmallVectorImpl<std::pair<Identifier, MemOp::PortKind>>;
using ModulePortList = SmallVectorImpl<ModulePortInfo>;

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
                                 const ModulePortList &modulePorts) {
  // Make a bundle of the inputs and outputs of the specified module.
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute, 4> resultNames;
  resultTypes.reserve(modulePorts.size());
  resultNames.reserve(modulePorts.size());
  for (auto port : modulePorts) {
    resultTypes.push_back(FlipType::get(port.type));
    resultNames.push_back(port.name);
  }

  return builder.create<InstanceOp>(loc, resultTypes, moduleName,
                                    builder.getArrayAttr(resultNames),
                                    instanceName.getValue());
}

/// Get the portlist for an external module representing a blackbox memory. This
/// external module must be compatible with the modules which are generated for
/// `vlsi_mem_gen`, which can be found in the rocketchip project.
static void
getBlackboxPortsForMemOp(MemOp op, const MemoryPortList &memPorts,
                         SmallVectorImpl<ModulePortInfo> &extPorts) {
  OpBuilder builder(op);
  unsigned readPorts = 0;
  unsigned writePorts = 0;
  unsigned readWritePorts = 0;
  for (unsigned i = 0; i < memPorts.size(); ++i) {
    // Calculate the naming prefix to use based on the kind of the port.
    std::string prefix;
    switch (memPorts[i].second) {
    case MemOp::PortKind::Read:
      prefix = (Twine("R") + Twine(readPorts++) + "_").str();
      break;
    case MemOp::PortKind::Write:
      prefix = (Twine("W") + Twine(writePorts++) + "_").str();
      break;
    case MemOp::PortKind::ReadWrite:
      prefix = (Twine("RW") + Twine(readWritePorts++) + "_").str();
      break;
    }
    // Flatten the bundle representing a memory port, name-mangling and adding
    // every field in the bundle to the exter module's port list.
    auto type = op.getResult(i).getType().cast<FIRRTLType>();
    bool shouldFlip = true;
    if (type.isa<FlipType>()) {
      type = type.cast<FlipType>().getElementType();
      // We need to flip each element in the bundle.  If the bundle as a whole
      // was flipped, we can reuse the element type directly without applying a
      // flip.
      shouldFlip = false;
    }
    for (auto bundleElement : type.cast<BundleType>().getElements()) {
      auto name = (prefix + bundleElement.name.getValue()).str();
      auto type = bundleElement.type;
      if (shouldFlip)
        type = FlipType::get(type);
      extPorts.push_back({builder.getStringAttr(name), type});
    }
  }
}

/// Create an external module blackbox representing the memory operation.
/// Returns the port list of the external module.
static FExtModuleOp
createBlackboxModuleForMem(MemOp op,
                           const SmallVectorImpl<ModulePortInfo> &extPorts) {

  OpBuilder builder(op->getContext());

  // The module's name is the name of the memory postfixed with "_ext".
  auto memName = op.name().str();
  if (memName.empty())
    memName = "mem";
  std::string extName = memName + "_ext";

  // Create the blackbox external module.
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
static FModuleOp
createWrapperModule(MemOp op, const MemoryPortList &memPorts,
                    FExtModuleOp extModuleOp,
                    const SmallVectorImpl<ModulePortInfo> &extPorts,
                    SmallVectorImpl<ModulePortInfo> &modPorts) {
  OpBuilder builder(op->getContext());

  // The wrapper module's name is the name of the memory.
  auto memName = op.name();
  if (memName.empty())
    memName = "mem";

  // Create a wrapper module with the same type as the memory
  modPorts.reserve(op.getResults().size());
  for (unsigned i = 0; i < memPorts.size(); ++i) {
    auto name = op.getPortName(i);
    auto type = FlipType::get(op.getPortType(i));
    modPorts.push_back({name, type});
  }
  auto moduleOp = builder.create<FModuleOp>(
      op.getLoc(), builder.getStringAttr(memName), modPorts);

  // Insert the externaml module into the circuit.  This will rename the module
  // if there is a conflict with another module name.
  auto circuitOp = op->getParentOfType<CircuitOp>();
  SymbolTable symbolTable(circuitOp);
  symbolTable.insert(moduleOp);

  // Move the module right after the external module, for readability purposes.
  moduleOp->moveAfter(extModuleOp);

  // Create the module
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());
  auto instanceOp = createInstance(builder, op.getLoc(), extModuleOp.getName(),
                                   op.nameAttr(), extPorts);

  // Connect the ports between the memory module and the instance of the black
  // box memory module. The outer module has a single bundle representing each
  // Memory port, while the inner module has a separate field for each memory
  // port bundle in the memory bundle.
  auto extResultIt = instanceOp.result_begin();
  for (auto memPort : moduleOp.getArguments()) {
    auto memPortType = memPort.getType().cast<FIRRTLType>();
    for (auto field :
         memPortType.getPassiveType().cast<BundleType>().getElements()) {
      auto fieldValue =
          builder.create<SubfieldOp>(op.getLoc(), memPort, field.name);
      // Create the connection between module arguments and the external module,
      // making sure that sinks are on the LHS
      if (fieldValue.getType().cast<FIRRTLType>().isPassive())
        builder.create<ConnectOp>(op.getLoc(), *extResultIt, fieldValue);
      else
        builder.create<ConnectOp>(op.getLoc(), fieldValue, *extResultIt);
      // advance the external module field iterator
      ++extResultIt;
    }
  }

  return moduleOp;
}

/// Create a bundle wire for each memory port.  This takes all the individual
/// fields returned from instantiating the external module, and wraps them in to
/// bundles to match the memory return type.
static void
createWiresForMemoryPorts(OpBuilder builder, Location loc, MemOp op,
                          InstanceOp instanceOp,
                          const SmallVectorImpl<ModulePortInfo> &extPorts,
                          SmallVectorImpl<Value> &results) {

  auto extResultIt = instanceOp.result_begin();

  for (auto memPort : op.getResults()) {
    // Create  a wire bundle for each memory port
    auto wireOp = builder.create<WireOp>(loc, memPort.getType());
    results.push_back(wireOp.getResult());

    // Connect each wire to the corresponding ports in the external module
    auto wireBundle = memPort.getType().cast<FIRRTLType>();
    if (wireBundle.isa<FlipType>())
      wireBundle = wireBundle.cast<FlipType>().getElementType();
    for (auto field : wireBundle.cast<BundleType>().getElements()) {
      auto fieldValue =
          builder.create<SubfieldOp>(op.getLoc(), wireOp, field.name);
      // Create the connection between module arguments and the external module,
      // making sure that sinks are on the LHS
      if ((*extResultIt).getType().cast<FIRRTLType>().isPassive())
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
                            MemOp memOp) {

  // The module we will be replacing the MemOp with.  If we don't have a
  // suitable memory module already created, a new one representing the memory
  // will be created.
  FModuleOp moduleOp;
  SmallVector<ModulePortInfo, 2> modPorts;

  // Check if we have already created a suitable wrapper module. If we have not
  // seen a similar memory, create a new wrapper module.
  auto it = knownMems.find(memOp);
  auto found = it != knownMems.end();
  if (found) {
    // Create an instance of the wrapping module.  We have to retrieve the
    // module port information back from the module.
    moduleOp = it->second;
    getModulePortInfo(moduleOp, modPorts);
  } else {
    // Get the memory port descriptors. This gives us the name and kind of each
    // memory port created by the MemOp.
    auto memPorts = memOp.getPorts();

    // Get the portlist for a module which represents the blackbox memory.
    // Typically has 1R + 1W memory port, which has 4+5=9 fields.
    SmallVector<ModulePortInfo, 9> extPortList;
    getBlackboxPortsForMemOp(memOp, memPorts, extPortList);
    auto extModuleOp = createBlackboxModuleForMem(memOp, extPortList);
    moduleOp = createWrapperModule(memOp, memPorts, extModuleOp, extPortList,
                                   modPorts);
    knownMems[memOp] = moduleOp;
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

static void
replaceMemsWithWrapperModules(CircuitOp circuit,
                              function_ref<bool(MemOp)> shouldReplace) {
  /// A set of replaced memory operations.  When two memory operations
  /// share the same types, they can share the same modules.
  DenseMap<MemOp, FModuleOp, MemOpInfo> knownMems;
  for (auto fmodule : circuit.getOps<FModuleOp>()) {
    for (auto memOp : llvm::make_early_inc_range(fmodule.getOps<MemOp>())) {
      if (shouldReplace(memOp)) {
        replaceMemWithWrapperModule(knownMems, memOp);
      }
    }
  }
  // Erase any of the remaining memory operations.  These were kept around
  // to check if other memory operations were equivalent.
  for (auto it : knownMems)
    it.first.erase();
}

static void
replaceMemWithExtModule(DenseMap<MemOp, FExtModuleOp, MemOpInfo> &knownMems,
                        MemOp memOp) {

  FExtModuleOp extModuleOp;
  SmallVector<ModulePortInfo, 9> extPortList;

  auto it = knownMems.find(memOp);
  auto found = it != knownMems.end();
  if (found) {
    // Create an instance of the wrapping module.  We have to retrieve the
    // module port information back from the module.
    extModuleOp = it->second;
    getModulePortInfo(extModuleOp, extPortList);
  } else {
    // Get the memory port descriptors.  This gives us the name and kind of each
    // memory port created by the MemOp.
    auto memPorts = memOp.getPorts();

    // Get the portlist for a module which represents the blackbox memory.
    // Typically has 1R + 1W memory port, which has 4+5=9 fields.
    getBlackboxPortsForMemOp(memOp, memPorts, extPortList);
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

static void replaceMemsWithExtModules(CircuitOp circuit,
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
}

namespace {
struct BlackboxMemoryPass : public BlackboxMemoryBase<BlackboxMemoryPass> {
  void runOnOperation() override {
    // A memory must have read and write latencies of 1 in order to be
    // blackboxed. In the future this will probably be configurable.
    auto shouldReplace = [](MemOp memOp) -> bool {
      return memOp.readLatency() == 1 && memOp.writeLatency() == 1;
    };
    if (emitWrapper)
      replaceMemsWithWrapperModules(getOperation(), shouldReplace);
    else
      replaceMemsWithExtModules(getOperation(), shouldReplace);
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createBlackBoxMemoryPass() {
  return std::make_unique<BlackboxMemoryPass>();
}
