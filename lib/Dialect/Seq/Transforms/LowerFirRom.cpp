//===- LowerFirRom.cpp - Seq FIRRTL ROM lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq FirRom ops to instances of HW generated modules.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/Namespace.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

#define DEBUG_TYPE "lower-firrom"

using namespace circt;
using namespace seq;
using namespace hw;
using hw::HWModuleGeneratedOp;
using llvm::MapVector;
using llvm::SmallDenseSet;

//===----------------------------------------------------------------------===//
// FIR ROM Parametrization
//===----------------------------------------------------------------------===//

namespace {
/// The configuration of a FIR ROM.
struct FirRomConfig {
  size_t numReadPorts = 0;
  size_t dataWidth = 0;
  size_t depth = 0;
  size_t readLatency = 0;
  StringRef initFilename;
  bool initIsBinary = false;
  bool initIsInline = false;
  Attribute outputFile;
  StringRef prefix;

  llvm::hash_code hashValue() const {
    return llvm::hash_combine(numReadPorts, dataWidth, depth, readLatency,
                              initFilename, initIsBinary, initIsInline,
                              outputFile, prefix);
  }

  auto getTuple() const {
    return std::make_tuple(numReadPorts, dataWidth, depth, readLatency,
                           initFilename, initIsBinary, initIsInline, outputFile,
                           prefix);
  }

  bool operator==(const FirRomConfig &other) const {
    return getTuple() == other.getTuple();
  }
};
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<FirRomConfig> {
  static inline FirRomConfig getEmptyKey() {
    FirRomConfig cfg;
    cfg.depth = DenseMapInfo<size_t>::getEmptyKey();
    return cfg;
  }
  static inline FirRomConfig getTombstoneKey() {
    FirRomConfig cfg;
    cfg.depth = DenseMapInfo<size_t>::getTombstoneKey();
    return cfg;
  }
  static unsigned getHashValue(const FirRomConfig &cfg) {
    return cfg.hashValue();
  }
  static bool isEqual(const FirRomConfig &lhs, const FirRomConfig &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_DEF_LOWERFIRROM
#include "circt/Dialect/Seq/SeqPasses.h.inc"

struct LowerFirRomPass : public impl::LowerFirRomBase<LowerFirRomPass> {
  /// A vector of unique `FirRomConfig`s and all the `FirRomOp`s that use it.
  using UniqueConfig = std::pair<FirRomConfig, SmallVector<FirRomOp, 1>>;
  using UniqueConfigs = SmallVector<UniqueConfig>;

  void runOnOperation() override;

  UniqueConfigs collectRoms(ArrayRef<HWModuleOp> modules);
  FirRomConfig collectRom(FirRomOp op);

  SmallVector<HWModuleGeneratedOp>
  createRomModules(MutableArrayRef<UniqueConfig> configs);
  HWModuleGeneratedOp createRomModule(UniqueConfig &config, OpBuilder &builder,
                                      FlatSymbolRefAttr schemaSymRef,
                                      Namespace &globalNamespace);

  void lowerRomsInModule(
      HWModuleOp module,
      ArrayRef<std::tuple<FirRomConfig *, HWModuleGeneratedOp, FirRomOp>> roms);
};
} // namespace

void LowerFirRomPass::runOnOperation() {
  // Gather all HW modules. We'll parallelize over them.
  SmallVector<HWModuleOp> modules;
  getOperation().walk([&](HWModuleOp op) {
    modules.push_back(op);
    return WalkResult::skip();
  });
  LLVM_DEBUG(llvm::dbgs() << "Lowering ROMs in " << modules.size()
                          << " modules\n");

  // Gather all `FirRomOp`s in the HW modules and group them by configuration.
  auto uniqueRoms = collectRoms(modules);
  LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueRoms.size()
                          << " unique ROM congiurations\n");
  if (uniqueRoms.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  // Create the `HWModuleGeneratedOp`s for each unique configuration. The result
  // is a vector of the same size as `uniqueRoms`, with a `HWModuleGeneratedOp`
  // for every unique rom configuration.
  auto genOps = createRomModules(uniqueRoms);

  // Group the list of roms that we need to update per HW module. This will
  // allow us to parallelize across HW modules.
  MapVector<
      HWModuleOp,
      SmallVector<std::tuple<FirRomConfig *, HWModuleGeneratedOp, FirRomOp>>>
      romsToLowerByModule;

  for (auto [config, genOp] : llvm::zip(uniqueRoms, genOps))
    for (auto romOp : config.second)
      romsToLowerByModule[romOp->getParentOfType<HWModuleOp>()].push_back(
          {&config.first, genOp, romOp});

  // Replace all `FirRomOp`s with instances of the generated module.
  if (getContext().isMultithreadingEnabled()) {
    llvm::parallelForEach(romsToLowerByModule, [&](auto pair) {
      lowerRomsInModule(pair.first, pair.second);
    });
  } else {
    for (auto [module, roms] : romsToLowerByModule)
      lowerRomsInModule(module, roms);
  }
}

/// Collect the roms in a list of HW modules.
LowerFirRomPass::UniqueConfigs
LowerFirRomPass::collectRoms(ArrayRef<HWModuleOp> modules) {
  // For each module in the list populate a separate vector of `FirRomOp`s in
  // that module. This allows for the traversal of the HW modules to be
  // parallelized.
  using ModuleRoms = SmallVector<std::pair<FirRomConfig, FirRomOp>, 0>;
  SmallVector<ModuleRoms> roms(modules.size());

  auto collect = [&](HWModuleOp module, ModuleRoms &roms) {
    // TODO: Check if this module is in the DUT hierarchy.
    // bool isInDut = state.isInDUT(module);
    module.walk([&](seq::FirRomOp op) {
      roms.push_back({collectRom(op), op});
    });
  };

  if (getContext().isMultithreadingEnabled()) {
    llvm::parallelFor(0, modules.size(),
                      [&](auto idx) { collect(modules[idx], roms[idx]); });
  } else {
    for (auto [module, moduleRoms] : llvm::zip(modules, roms))
      collect(module, moduleRoms);
  }

  // Group the gathered roms by unique `FirRomConfig` details.
  MapVector<FirRomConfig, SmallVector<FirRomOp, 1>> grouped;
  for (auto [module, moduleRoms] : llvm::zip(modules, roms))
    for (auto [summary, romOp] : moduleRoms)
      grouped[summary].push_back(romOp);

  return grouped.takeVector();
}

/// Determine the exact parametrization of the ROM that should be generated
/// for a given `FirRomOp`.
FirRomConfig LowerFirRomPass::collectRom(FirRomOp op) {
  FirRomConfig cfg;
  cfg.dataWidth = op.getType().getWidth();
  cfg.depth = op.getType().getDepth();
  cfg.readLatency = op.getReadLatency();
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

  // Count the read ports.
  for (auto *user : op->getUsers()) {
    if (isa<FirRomReadOp>(user))
      ++cfg.numReadPorts;
  }

  return cfg;
}

/// Create the `HWModuleGeneratedOp` for a list of ROM parametrizations.
SmallVector<HWModuleGeneratedOp>
LowerFirRomPass::createRomModules(MutableArrayRef<UniqueConfig> configs) {
  ModuleOp circuit = getOperation();

  // Create or re-use the generator schema.
  hw::HWGeneratorSchemaOp schemaOp;
  for (auto op : circuit.getOps<hw::HWGeneratorSchemaOp>()) {
    if (op.getDescriptor() == "FIRRTL_Rom") {
      schemaOp = op;
      break;
    }
  }
  if (!schemaOp) {
    auto builder = OpBuilder::atBlockBegin(getOperation().getBody());
    std::array<StringRef, 7> schemaFields = {
        "depth",        "numReadPorts", "readLatency", "width",
        "initFilename", "initIsBinary", "initIsInline"};
    schemaOp = builder.create<hw::HWGeneratorSchemaOp>(
        getOperation().getLoc(), "FIRRTLRom", "FIRRTL_Rom",
        builder.getStrArrayAttr(schemaFields));
  }
  auto schemaSymRef = FlatSymbolRefAttr::get(schemaOp);

  // Determine the insertion point for each of the ROM modules. We basically
  // put them ahead of the first module that instantiates that ROM. Do this
  // here in one go such that the `isBeforeInBlock` calls don't have to
  // re-enumerate the entire IR every time we insert one of the ROM modules.
  SmallVector<Operation *> insertionPoints;
  insertionPoints.reserve(configs.size());
  for (auto &config : configs) {
    Operation *op = nullptr;
    for (auto romOp : config.second)
      if (auto parent = romOp->getParentOfType<HWModuleOp>())
        if (!op || parent->isBeforeInBlock(op))
          op = parent;
    insertionPoints.push_back(op);
  }

  // Create the individual ROM modules.
  SymbolCache symbolCache;
  symbolCache.addDefinitions(getOperation());
  Namespace globalNamespace;
  globalNamespace.add(symbolCache);

  SmallVector<HWModuleGeneratedOp> genOps;
  genOps.reserve(configs.size());
  for (auto [config, insertBefore] : llvm::zip(configs, insertionPoints)) {
    OpBuilder builder(circuit.getContext());
    builder.setInsertionPoint(insertBefore);
    genOps.push_back(
        createRomModule(config, builder, schemaSymRef, globalNamespace));
  }

  return genOps;
}

/// Create the `HWModuleGeneratedOp` for a single ROM parametrization.
HWModuleGeneratedOp
LowerFirRomPass::createRomModule(UniqueConfig &config, OpBuilder &builder,
                                 FlatSymbolRefAttr schemaSymRef,
                                 Namespace &globalNamespace) {
  const auto &rom = config.first;
  auto &romOps = config.second;

  // Pick a name for the ROM. Honor the optional prefix and try to include
  // the common part of the names of the ROM instances that use this
  // configuration. The resulting name is of the form:
  //
  //   <prefix>_<commonName>_<depth>x<width>
  //
  StringRef baseName = "";
  bool firstFound = false;
  for (auto romOp : romOps) {
    if (auto romName = romOp.getName()) {
      if (!firstFound) {
        baseName = *romName;
        firstFound = true;
        continue;
      }
      unsigned idx = 0;
      for (; idx < romName->size() && idx < baseName.size(); ++idx)
        if ((*romName)[idx] != baseName[idx])
          break;
      baseName = baseName.take_front(idx);
    }
  }
  baseName = baseName.rtrim('_');

  SmallString<32> nameBuffer;
  nameBuffer += rom.prefix;
  if (!baseName.empty()) {
    nameBuffer += baseName;
  } else {
    nameBuffer += "rom";
  }
  nameBuffer += "_";
  (Twine(rom.depth) + "x" + Twine(rom.dataWidth)).toVector(nameBuffer);
  auto name = builder.getStringAttr(globalNamespace.newName(nameBuffer));

  LLVM_DEBUG(llvm::dbgs() << "Creating " << name << " for " << rom.depth
                          << " x " << rom.dataWidth << " ROM\n");

  SmallVector<hw::PortInfo> ports;

  // Common types used for ROM ports.
  Type bitType = IntegerType::get(&getContext(), 1);
  Type dataType =
      IntegerType::get(&getContext(), std::max((size_t)1, rom.dataWidth));
  Type addrType = IntegerType::get(&getContext(),
                                   std::max(1U, llvm::Log2_64_Ceil(rom.depth)));

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
    addInput(prefix, idx, "_clk", bitType);
  };

  // Add the read ports.
  for (size_t i = 0, e = rom.numReadPorts; i != e; ++i) {
    addCommonPorts("R", i);
    addOutput("R", i, "_data", dataType);
  }

  auto genAttr = [&](StringRef name, Attribute attr) {
    return builder.getNamedAttr(name, attr);
  };
  auto genAttrUI32 = [&](StringRef name, uint32_t value) {
    return genAttr(name, builder.getUI32IntegerAttr(value));
  };
  NamedAttribute genAttrs[] = {
      genAttr("depth", builder.getI64IntegerAttr(rom.depth)),
      genAttrUI32("numReadPorts", rom.numReadPorts),
      genAttrUI32("readLatency", rom.readLatency),
      genAttrUI32("width", rom.dataWidth),
      genAttr("initFilename", builder.getStringAttr(rom.initFilename)),
      genAttr("initIsBinary", builder.getBoolAttr(rom.initIsBinary)),
      genAttr("initIsInline", builder.getBoolAttr(rom.initIsInline))};

  // Combine the locations of all actual `FirRomOp`s to be the location of the
  // generated ROM.
  Location loc = romOps.front().getLoc();
  if (romOps.size() > 1) {
    SmallVector<Location> locs;
    for (auto romOp : romOps)
      locs.push_back(romOp.getLoc());
    loc = FusedLoc::get(&getContext(), locs);
  }

  // Create the module.
  auto genOp = builder.create<hw::HWModuleGeneratedOp>(
      loc, schemaSymRef, name, ports, StringRef{}, ArrayAttr{}, genAttrs);
  if (rom.outputFile)
    genOp->setAttr("output_file", rom.outputFile);

  return genOp;
}

/// Replace all `FirRomOp`s in an HW module with an instance of the
/// corresponding generated module.
void LowerFirRomPass::lowerRomsInModule(
    HWModuleOp module,
    ArrayRef<std::tuple<FirRomConfig *, HWModuleGeneratedOp, FirRomOp>> roms) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering " << roms.size() << " ROMs in "
                          << module.getName() << "\n");

  hw::ConstantOp constOneOp;
  auto constOne = [&] {
    if (!constOneOp) {
      auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());
      constOneOp = builder.create<hw::ConstantOp>(module.getLoc(),
                                                  builder.getI1Type(), 1);
    }
    return constOneOp;
  };
  auto valueOrOne = [&](Value value) { return value ? value : constOne(); };

  for (auto [config, genOp, romOp] : roms) {
    LLVM_DEBUG(llvm::dbgs() << "- Lowering " << romOp.getName() << "\n");
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;

    auto addInput = [&](Value value) { inputs.push_back(value); };
    auto addOutput = [&](Value value) { outputs.push_back(value); };

    // Add the read ports.
    for (auto *op : romOp->getUsers()) {
      auto port = dyn_cast<FirRomReadOp>(op);
      if (!port)
        continue;
      addInput(port.getAddress());
      addInput(valueOrOne(port.getEnable()));
      addInput(port.getClock());
      addOutput(port.getData());
    }

    // Create the module instance.
    StringRef romName = "rom";
    if (auto name = romOp.getName(); name && !name->empty())
      romName = *name;
    ImplicitLocOpBuilder builder(romOp.getLoc(), romOp);
    auto instOp = builder.create<hw::InstanceOp>(
        genOp, builder.getStringAttr(romName + "_ext"), inputs, ArrayAttr{},
        romOp.getInnerSymAttr());
    for (auto [oldOutput, newOutput] : llvm::zip(outputs, instOp.getResults()))
      oldOutput.replaceAllUsesWith(newOutput);

    // Carry attributes over from the `FirRomOp` to the `InstanceOp`.
    auto defaultAttrNames = romOp.getAttributeNames();
    for (auto namedAttr : romOp->getAttrs())
      if (!llvm::is_contained(defaultAttrNames, namedAttr.getName()))
        instOp->setAttr(namedAttr.getName(), namedAttr.getValue());

    // Get rid of the `FirRomOp`.
    for (auto *user : llvm::make_early_inc_range(romOp->getUsers()))
      user->erase();
    romOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::seq::createLowerFirRomPass() {
  return std::make_unique<LowerFirRomPass>();
}
