//===- FirMemLowering.cpp - FirMem lowering utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FirMemLowering.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace hw;
using namespace seq;
using llvm::MapVector;

#define DEBUG_TYPE "lower-seq-firmem"

FirMemLowering::FirMemLowering(ModuleOp circuit)
    : context(circuit.getContext()), circuit(circuit) {
  symbolCache.addDefinitions(circuit);
  globalNamespace.add(symbolCache);

  // For each module, assign an index. Use it to identify the insertion point
  // for the generated ops.
  for (auto [index, module] : llvm::enumerate(circuit.getOps<HWModuleOp>()))
    moduleIndex[module] = index;
}

/// Collect the memories in a list of HW modules.
FirMemLowering::UniqueConfigs
FirMemLowering::collectMemories(ArrayRef<HWModuleOp> modules) {
  // For each module in the list populate a separate vector of `FirMemOp`s in
  // that module. This allows for the traversal of the HW modules to be
  // parallelized.
  using ModuleMemories = SmallVector<std::pair<FirMemConfig, FirMemOp>, 0>;
  SmallVector<ModuleMemories> memories(modules.size());

  mlir::parallelFor(context, 0, modules.size(), [&](auto idx) {
    // TODO: Check if this module is in the DUT hierarchy.
    // bool isInDut = state.isInDUT(module);
    HWModuleOp(modules[idx]).walk([&](seq::FirMemOp op) {
      memories[idx].push_back({collectMemory(op), op});
    });
  });

  // Group the gathered memories by unique `FirMemConfig` details.
  MapVector<FirMemConfig, SmallVector<FirMemOp, 1>> grouped;
  for (auto [module, moduleMemories] : llvm::zip(modules, memories))
    for (auto [summary, memOp] : moduleMemories)
      grouped[summary].push_back(memOp);

  return grouped;
}

/// Trace a value through wires to its original definition.
static Value lookThroughWires(Value value) {
  while (value) {
    if (auto wireOp = value.getDefiningOp<WireOp>()) {
      value = wireOp.getInput();
      continue;
    }
    break;
  }
  return value;
}

/// Determine the exact parametrization of the memory that should be generated
/// for a given `FirMemOp`.
FirMemConfig FirMemLowering::collectMemory(FirMemOp op) {
  FirMemConfig cfg;
  cfg.dataWidth = op.getType().getWidth();
  cfg.depth = op.getType().getDepth();
  cfg.readLatency = op.getReadLatency();
  cfg.writeLatency = op.getWriteLatency();
  cfg.maskBits = op.getType().getMaskWidth().value_or(1);
  cfg.readUnderWrite = op.getRuw();
  cfg.writeUnderWrite = op.getWuw();
  if (auto init = op.getInitAttr()) {
    cfg.initFilename = init.getFilename();
    cfg.initIsBinary = init.getIsBinary();
    cfg.initIsInline = init.getIsInline();
  }
  cfg.outputFile = op.getOutputFileAttr();
  if (auto prefix = op.getPrefixAttr())
    cfg.prefix = prefix.getValue();
  // TODO: Handle modName (maybe not?)
  // TODO: Handle groupID (maybe not?)

  // Count the read, write, and read-write ports, and identify the clocks
  // driving the write ports.
  SmallDenseMap<Value, unsigned> clockValues;
  for (auto *user : op->getUsers()) {
    if (isa<FirMemReadOp>(user))
      ++cfg.numReadPorts;
    else if (isa<FirMemWriteOp>(user))
      ++cfg.numWritePorts;
    else if (isa<FirMemReadWriteOp>(user))
      ++cfg.numReadWritePorts;

    // Assign IDs to the values used as clock. This allows later passes to
    // easily detect which clocks are effectively driven by the same value.
    if (isa<FirMemWriteOp, FirMemReadWriteOp>(user)) {
      auto clock = lookThroughWires(user->getOperand(2));
      cfg.writeClockIDs.push_back(
          clockValues.insert({clock, clockValues.size()}).first->second);
    }
  }

  return cfg;
}

FlatSymbolRefAttr FirMemLowering::getOrCreateSchema() {
  if (!schemaOp) {
    // Create or re-use the generator schema.
    for (auto op : circuit.getOps<hw::HWGeneratorSchemaOp>()) {
      if (op.getDescriptor() == "FIRRTL_Memory") {
        schemaOp = op;
        break;
      }
    }
    if (!schemaOp) {
      auto builder = OpBuilder::atBlockBegin(circuit.getBody());
      std::array<StringRef, 14> schemaFields = {
          "depth",          "numReadPorts",
          "numWritePorts",  "numReadWritePorts",
          "readLatency",    "writeLatency",
          "width",          "maskGran",
          "readUnderWrite", "writeUnderWrite",
          "writeClockIDs",  "initFilename",
          "initIsBinary",   "initIsInline"};
      schemaOp = hw::HWGeneratorSchemaOp::create(
          builder, circuit.getLoc(), "FIRRTLMem", "FIRRTL_Memory",
          builder.getStrArrayAttr(schemaFields));
    }
  }
  return FlatSymbolRefAttr::get(schemaOp);
}

/// Create the `HWModuleGeneratedOp` for a single memory parametrization.
HWModuleGeneratedOp
FirMemLowering::createMemoryModule(FirMemConfig &mem,
                                   ArrayRef<seq::FirMemOp> memOps) {
  auto schemaSymRef = getOrCreateSchema();

  // Identify the first module which uses the memory configuration.
  // Insert the generated module before it.
  HWModuleOp insertPt;
  for (auto memOp : memOps) {
    auto parent = memOp->getParentOfType<HWModuleOp>();
    if (!insertPt || moduleIndex[parent] < moduleIndex[insertPt])
      insertPt = parent;
  }

  OpBuilder builder(context);
  builder.setInsertionPoint(insertPt);

  // Pick a name for the memory. Honor the optional prefix and try to include
  // the common part of the names of the memory instances that use this
  // configuration. The resulting name is of the form:
  //
  //   <prefix>_<commonName>_<depth>x<width>
  //
  StringRef baseName = "";
  bool firstFound = false;
  for (auto memOp : memOps) {
    if (auto memName = memOp.getName()) {
      if (!firstFound) {
        baseName = *memName;
        firstFound = true;
        continue;
      }
      unsigned idx = 0;
      for (; idx < memName->size() && idx < baseName.size(); ++idx)
        if ((*memName)[idx] != baseName[idx])
          break;
      baseName = baseName.take_front(idx);
    }
  }
  baseName = baseName.rtrim('_');

  SmallString<32> nameBuffer;
  nameBuffer += mem.prefix;
  if (!baseName.empty()) {
    nameBuffer += baseName;
  } else {
    nameBuffer += "mem";
  }
  nameBuffer += "_";
  (Twine(mem.depth) + "x" + Twine(mem.dataWidth)).toVector(nameBuffer);
  auto name = builder.getStringAttr(globalNamespace.newName(nameBuffer));

  LLVM_DEBUG(llvm::dbgs() << "Creating " << name << " for " << mem.depth
                          << " x " << mem.dataWidth << " memory\n");

  bool withMask = mem.maskBits > 1;
  SmallVector<hw::PortInfo> ports;

  // Common types used for memory ports.
  Type clkType = ClockType::get(context);
  Type bitType = IntegerType::get(context, 1);
  Type dataType = IntegerType::get(context, std::max((size_t)1, mem.dataWidth));
  Type maskType = IntegerType::get(context, mem.maskBits);
  Type addrType =
      IntegerType::get(context, std::max(1U, llvm::Log2_64_Ceil(mem.depth)));

  // Helper to add an input port.
  size_t inputIdx = 0;
  auto addInput = [&](StringRef prefix, size_t idx, StringRef suffix,
                      Type type) {
    ports.push_back({{builder.getStringAttr(prefix + Twine(idx) + suffix), type,
                      ModulePort::Direction::Input},
                     inputIdx++});
  };

  // Helper to add an output port.
  size_t outputIdx = 0;
  auto addOutput = [&](StringRef prefix, size_t idx, StringRef suffix,
                       Type type) {
    ports.push_back({{builder.getStringAttr(prefix + Twine(idx) + suffix), type,
                      ModulePort::Direction::Output},
                     outputIdx++});
  };

  // Helper to add the ports common to read, read-write, and write ports.
  auto addCommonPorts = [&](StringRef prefix, size_t idx) {
    addInput(prefix, idx, "_addr", addrType);
    addInput(prefix, idx, "_en", bitType);
    addInput(prefix, idx, "_clk", clkType);
  };

  // Add the read ports.
  for (size_t i = 0, e = mem.numReadPorts; i != e; ++i) {
    addCommonPorts("R", i);
    addOutput("R", i, "_data", dataType);
  }

  // Add the read-write ports.
  for (size_t i = 0, e = mem.numReadWritePorts; i != e; ++i) {
    addCommonPorts("RW", i);
    addInput("RW", i, "_wmode", bitType);
    addInput("RW", i, "_wdata", dataType);
    addOutput("RW", i, "_rdata", dataType);
    if (withMask)
      addInput("RW", i, "_wmask", maskType);
  }

  // Add the write ports.
  for (size_t i = 0, e = mem.numWritePorts; i != e; ++i) {
    addCommonPorts("W", i);
    addInput("W", i, "_data", dataType);
    if (withMask)
      addInput("W", i, "_mask", maskType);
  }

  // Mask granularity is the number of data bits that each mask bit can
  // guard. By default it is equal to the data bitwidth.
  auto genAttr = [&](StringRef name, Attribute attr) {
    return builder.getNamedAttr(name, attr);
  };
  auto genAttrUI32 = [&](StringRef name, uint32_t value) {
    return genAttr(name, builder.getUI32IntegerAttr(value));
  };
  NamedAttribute genAttrs[] = {
      genAttr("depth", builder.getI64IntegerAttr(mem.depth)),
      genAttrUI32("numReadPorts", mem.numReadPorts),
      genAttrUI32("numWritePorts", mem.numWritePorts),
      genAttrUI32("numReadWritePorts", mem.numReadWritePorts),
      genAttrUI32("readLatency", mem.readLatency),
      genAttrUI32("writeLatency", mem.writeLatency),
      genAttrUI32("width", mem.dataWidth),
      genAttrUI32("maskGran", mem.dataWidth / mem.maskBits),
      genAttr("readUnderWrite",
              seq::RUWAttr::get(builder.getContext(), mem.readUnderWrite)),
      genAttr("writeUnderWrite",
              seq::WUWAttr::get(builder.getContext(), mem.writeUnderWrite)),
      genAttr("writeClockIDs", builder.getI32ArrayAttr(mem.writeClockIDs)),
      genAttr("initFilename", builder.getStringAttr(mem.initFilename)),
      genAttr("initIsBinary", builder.getBoolAttr(mem.initIsBinary)),
      genAttr("initIsInline", builder.getBoolAttr(mem.initIsInline))};

  // Combine the locations of all actual `FirMemOp`s to be the location of the
  // generated memory.
  Location loc = FirMemOp(memOps.front()).getLoc();
  if (memOps.size() > 1) {
    SmallVector<Location> locs;
    for (auto memOp : memOps)
      locs.push_back(memOp.getLoc());
    loc = FusedLoc::get(context, locs);
  }

  // Create the module.
  auto genOp =
      hw::HWModuleGeneratedOp::create(builder, loc, schemaSymRef, name, ports,
                                      StringRef{}, ArrayAttr{}, genAttrs);
  if (mem.outputFile)
    genOp->setAttr("output_file", mem.outputFile);

  return genOp;
}

/// Replace all `FirMemOp`s in an HW module with an instance of the
/// corresponding generated module.
void FirMemLowering::lowerMemoriesInModule(
    HWModuleOp module,
    ArrayRef<std::tuple<FirMemConfig *, HWModuleGeneratedOp, FirMemOp>> mems) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering " << mems.size() << " memories in "
                          << module.getName() << "\n");

  DenseMap<unsigned, Value> constOneOps;
  auto constOne = [&](unsigned width = 1) {
    auto it = constOneOps.try_emplace(width, Value{});
    if (it.second) {
      auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());
      it.first->second = hw::ConstantOp::create(
          builder, module.getLoc(), builder.getIntegerType(width), 1);
    }
    return it.first->second;
  };
  auto valueOrOne = [&](Value value, unsigned width = 1) {
    return value ? value : constOne(width);
  };

  for (auto [config, genOp, memOp] : mems) {
    LLVM_DEBUG(llvm::dbgs() << "- Lowering " << memOp.getName() << "\n");
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;

    auto addInput = [&](Value value) { inputs.push_back(value); };
    auto addOutput = [&](Value value) { outputs.push_back(value); };

    // Add the read ports.
    for (auto *op : memOp->getUsers()) {
      auto port = dyn_cast<FirMemReadOp>(op);
      if (!port)
        continue;
      addInput(port.getAddress());
      addInput(valueOrOne(port.getEnable()));
      addInput(port.getClk());
      addOutput(port.getData());
    }

    // Add the read-write ports.
    for (auto *op : memOp->getUsers()) {
      auto port = dyn_cast<FirMemReadWriteOp>(op);
      if (!port)
        continue;
      addInput(port.getAddress());
      addInput(valueOrOne(port.getEnable()));
      addInput(port.getClk());
      addInput(port.getMode());
      addInput(port.getWriteData());
      addOutput(port.getReadData());
      if (config->maskBits > 1)
        addInput(valueOrOne(port.getMask(), config->maskBits));
    }

    // Add the write ports.
    for (auto *op : memOp->getUsers()) {
      auto port = dyn_cast<FirMemWriteOp>(op);
      if (!port)
        continue;
      addInput(port.getAddress());
      addInput(valueOrOne(port.getEnable()));
      addInput(port.getClk());
      addInput(port.getData());
      if (config->maskBits > 1)
        addInput(valueOrOne(port.getMask(), config->maskBits));
    }

    // Create the module instance.
    StringRef memName = "mem";
    if (auto name = memOp.getName(); name && !name->empty())
      memName = *name;
    ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
    auto instOp = hw::InstanceOp::create(
        builder, genOp, builder.getStringAttr(memName + "_ext"), inputs,
        ArrayAttr{}, memOp.getInnerSymAttr());
    for (auto [oldOutput, newOutput] : llvm::zip(outputs, instOp.getResults()))
      oldOutput.replaceAllUsesWith(newOutput);

    // Carry attributes over from the `FirMemOp` to the `InstanceOp`.
    auto defaultAttrNames = memOp.getAttributeNames();
    for (auto namedAttr : memOp->getAttrs())
      if (!llvm::is_contained(defaultAttrNames, namedAttr.getName()))
        instOp->setAttr(namedAttr.getName(), namedAttr.getValue());

    // Get rid of the `FirMemOp`.
    for (auto *user : llvm::make_early_inc_range(memOp->getUsers()))
      user->erase();
    memOp.erase();
  }
}
