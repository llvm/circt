//===- MSFTPasses.cpp - Implement MSFT passes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace msft;

namespace circt {
namespace msft {
#define GEN_PASS_CLASSES
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace msft
} // namespace circt

//===----------------------------------------------------------------------===//
// Lower MSFT to HW.
//===----------------------------------------------------------------------===//

namespace {
/// Lower MSFT's InstanceOp to HW's. Currently trivial since `msft.instance` is
/// currently a subset of `hw.instance`.
struct InstanceOpLowering : public OpConversionPattern<InstanceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
InstanceOpLowering::matchAndRewrite(InstanceOp msftInst, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Operation *referencedModule = msftInst.getReferencedModule();
  if (!referencedModule)
    return rewriter.notifyMatchFailure(msftInst,
                                       "Could not find referenced module");
  if (!hw::isAnyModule(referencedModule))
    return rewriter.notifyMatchFailure(
        msftInst, "Referenced module was not an HW module");
  auto hwInst = rewriter.create<hw::InstanceOp>(
      msftInst.getLoc(), referencedModule, msftInst.instanceNameAttr(),
      SmallVector<Value>(adaptor.getOperands().begin(),
                         adaptor.getOperands().end()),
      msftInst.parameters().getValueOr(ArrayAttr()), msftInst.sym_nameAttr());
  rewriter.replaceOp(msftInst, hwInst.getResults());
  return success();
}

namespace {
/// Lower MSFT's ModuleOp to HW's.
struct ModuleOpLowering : public OpConversionPattern<MSFTModuleOp> {
public:
  ModuleOpLowering(MLIRContext *context, StringRef outputFile)
      : OpConversionPattern::OpConversionPattern(context),
        outputFile(outputFile) {}

  LogicalResult
  matchAndRewrite(MSFTModuleOp mod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  StringRef outputFile;
};
} // anonymous namespace

LogicalResult
ModuleOpLowering::matchAndRewrite(MSFTModuleOp mod, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (mod.body().empty()) {
    std::string comment;
    llvm::raw_string_ostream(comment)
        << "// Module not generated: \"" << mod.getName() << "\" params "
        << mod.parameters();
    // TODO: replace this with proper comment op when it's created.
    rewriter.replaceOpWithNewOp<sv::VerbatimOp>(mod, comment);
    return success();
  }

  auto hwmod = rewriter.replaceOpWithNewOp<hw::HWModuleOp>(
      mod, mod.getNameAttr(), mod.getPorts());
  rewriter.eraseBlock(hwmod.getBodyBlock());
  rewriter.inlineRegionBefore(mod.getBody(), hwmod.getBody(),
                              hwmod.getBody().end());

  auto opOutputFile = mod.fileName();
  if (opOutputFile) {
    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        rewriter.getContext(), *opOutputFile, false, true);
    hwmod->setAttr("output_file", outputFileAttr);
  } else if (!outputFile.empty()) {
    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        rewriter.getContext(), outputFile, false, true);
    hwmod->setAttr("output_file", outputFileAttr);
  }

  return success();
}
namespace {

/// Lower MSFT's ModuleExternOp to HW's.
struct ModuleExternOpLowering : public OpConversionPattern<MSFTModuleExternOp> {
public:
  ModuleExternOpLowering(MLIRContext *context, StringRef outputFile)
      : OpConversionPattern::OpConversionPattern(context),
        outputFile(outputFile) {}

  LogicalResult
  matchAndRewrite(MSFTModuleExternOp mod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  StringRef outputFile;
};
} // anonymous namespace

LogicalResult ModuleExternOpLowering::matchAndRewrite(
    MSFTModuleExternOp mod, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto hwMod = rewriter.replaceOpWithNewOp<hw::HWModuleExternOp>(
      mod, mod.getNameAttr(), mod.getPorts(), mod.verilogName().getValueOr(""),
      mod.parameters());

  if (!outputFile.empty()) {
    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        rewriter.getContext(), outputFile, false, true);
    hwMod->setAttr("output_file", outputFileAttr);
  }

  return success();
}

namespace {
/// Lower MSFT's OutputOp to HW's.
struct OutputOpLowering : public OpConversionPattern<OutputOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp out, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(out, out.getOperands());
    return success();
  }
};
} // anonymous namespace

namespace {
/// Simply remove PhysicalRegionOp when done.
struct PhysicalRegionOpLowering : public OpConversionPattern<PhysicalRegionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PhysicalRegionOp physicalRegion, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(physicalRegion);
    return success();
  }
};
} // anonymous namespace

namespace {
struct LowerToHWPass : public LowerToHWBase<LowerToHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void LowerToHWPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  // Traverse MSFT location attributes and export the required Tcl into
  // templated `sv::VerbatimOp`s with symbolic references to the instance paths.
  hw::SymbolCache symCache;
  populateSymbolCache(top, symCache);
  for (auto moduleName : tops) {
    auto hwmod = top.lookupSymbol<msft::MSFTModuleOp>(moduleName);
    if (!hwmod)
      continue;
    if (failed(exportQuartusTcl(hwmod, symCache, tclFile)))
      return signalPassFailure();
  }

  // The `hw::InstanceOp` (which `msft::InstanceOp` lowers to) convenience
  // builder gets its argNames and resultNames from the `hw::HWModuleOp`. So we
  // have to lower `msft::MSFTModuleOp` before we lower `msft::InstanceOp`.

  // Convert everything except instance ops first.

  ConversionTarget target(*ctxt);
  target.addIllegalOp<MSFTModuleOp, MSFTModuleExternOp, OutputOp>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<sv::SVDialect>();

  RewritePatternSet patterns(ctxt);
  patterns.insert<ModuleOpLowering>(ctxt, verilogFile);
  patterns.insert<ModuleExternOpLowering>(ctxt, verilogFile);
  patterns.insert<OutputOpLowering>(ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();

  // Then, convert the InstanceOps
  target.addIllegalOp<msft::InstanceOp>();
  RewritePatternSet instancePatterns(ctxt);
  instancePatterns.insert<InstanceOpLowering>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(instancePatterns))))
    signalPassFailure();

  // Finally, legalize the rest of the MSFT dialect.
  target.addIllegalDialect<MSFTDialect>();
  RewritePatternSet finalPatterns(ctxt);
  finalPatterns.insert<PhysicalRegionOpLowering>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(finalPatterns))))
    signalPassFailure();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerToHWPass() {
  return std::make_unique<LowerToHWPass>();
}
} // namespace msft
} // namespace circt

namespace {
struct PartitionPass : public PartitionBase<PartitionPass> {
  void runOnOperation() override;

private:
  hw::SymbolCache topLevelSyms;

  void partition(MSFTModuleOp mod);
  void partition(DesignPartitionOp part, SmallVectorImpl<Operation *> &users);

  // Find all the modules and use the partial order of the instantiation DAG to
  // sort them. If we use this order when "bubbling" up operations, we guarantee
  // one-pass completeness.
  // Assumption (unchecked): there is not a cycle in the instantiation graph.
  void getAndSortModules(SmallVectorImpl<MSFTModuleOp> &mods);
  void getAndSortModulesVisitor(MSFTModuleOp mod,
                                SmallVectorImpl<MSFTModuleOp> &mods,
                                DenseSet<MSFTModuleOp> &modsSeen);
};
} // anonymous namespace

// Run a post-order DFS.
void PartitionPass::getAndSortModulesVisitor(
    MSFTModuleOp mod, SmallVectorImpl<MSFTModuleOp> &mods,
    DenseSet<MSFTModuleOp> &modsSeen) {
  if (modsSeen.contains(mod))
    return;
  modsSeen.insert(mod);

  mod.walk([&](InstanceOp inst) {
    Operation *modOp = topLevelSyms.getDefinition(inst.moduleNameAttr());
    auto mod = dyn_cast_or_null<MSFTModuleOp>(modOp);
    if (!mod)
      return;
    getAndSortModulesVisitor(mod, mods, modsSeen);
  });

  mods.push_back(mod);
}

void PartitionPass::getAndSortModules(SmallVectorImpl<MSFTModuleOp> &mods) {
  // Add here _before_ we go deeper to prevent infinite recursion.
  DenseSet<MSFTModuleOp> modsSeen;
  getOperation().walk(
      [&](MSFTModuleOp mod) { getAndSortModulesVisitor(mod, mods, modsSeen); });
}

/// Fill a symbol cache with all the top level symbols.
static void populateSymbolCache(mlir::ModuleOp mod, hw::SymbolCache &cache) {
  for (Operation &op : mod.getBody()->getOperations()) {
    StringAttr symName = SymbolTable::getSymbolName(&op);
    if (!symName)
      continue;
    // Add the symbol to the cache.
    cache.addDefinition(symName, &op);
  }
  cache.freeze();
}

void PartitionPass::runOnOperation() {
  ModuleOp outerMod = getOperation();
  ::populateSymbolCache(outerMod, topLevelSyms);

  // Get a properly sorted list, then partition the mods in order.
  SmallVector<MSFTModuleOp, 64> sortedMods;
  getAndSortModules(sortedMods);
  for (auto mod : sortedMods)
    partition(mod);
}

void PartitionPass::partition(MSFTModuleOp mod) {
  DenseMap<SymbolRefAttr, SmallVector<Operation *, 1>> partMembers;
  mod.walk([&partMembers](Operation *op) {
    if (auto part = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition"))
      partMembers[part].push_back(op);
  });

  for (auto part :
       llvm::make_early_inc_range(mod.getOps<DesignPartitionOp>())) {
    SymbolRefAttr partSym = SymbolRefAttr::get(SymbolTable::getSymbolName(mod),
                                               {SymbolRefAttr::get(part)});

    auto usersIter = partMembers.find(partSym);
    if (usersIter != partMembers.end()) {
      this->partition(part, usersIter->second);
      partMembers.erase(usersIter);
    }
    part.erase();
  }

  // TODO: For operations which target partitions not in the same module, bubble
  // them up.
}

void PartitionPass::partition(DesignPartitionOp partOp,
                              SmallVectorImpl<Operation *> &toMove) {

  auto *ctxt = partOp.getContext();
  auto loc = partOp.getLoc();

  //*************
  //   Determine the partition module's interface. Keep some bookkeeping around.
  SmallVector<hw::PortInfo, 128> ports;
  SmallVector<Value, 64> partInstInputs;
  SmallVector<Value, 64> partInstOutputs;

  // Handle module-like operations. Not strictly necessary, but we can base the
  // new portnames on the portnames of the instance being moved and the instance
  // name.
  auto addModuleLike = [&](Operation *inst, hw::ModulePortInfo modPorts) {
    StringAttr name = SymbolTable::getSymbolName(inst);

    for (auto port :
         llvm::concat<hw::PortInfo>(modPorts.inputs, modPorts.outputs)) {
      ports.push_back(
          hw::PortInfo{/*name*/ StringAttr::get(ctxt, name.getValue() + "_" +
                                                          port.name.getValue()),
                       /*direction*/ port.direction,
                       /*type*/ port.type,
                       /*argNum*/ ports.size()});
      if (port.direction == hw::PortDirection::OUTPUT)
        partInstOutputs.push_back(inst->getResult(port.argNum));
      else
        partInstInputs.push_back(inst->getOperand(port.argNum));
    }
  };
  // Handle all other operators.
  auto addOther = [&](Operation *op) { assert(false && "Unimplemented"); };

  // Aggregate the args/results into partition module ports.
  for (Operation *op : toMove) {
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      Operation *modOp = topLevelSyms.getDefinition(inst.moduleNameAttr());
      assert(modOp && "Module instantiated should exist. Verifier should have "
                      "caught this.");

      if (auto mod = dyn_cast<MSFTModuleOp>(modOp)) {
        addModuleLike(inst, mod.getPorts());
      } else if (auto mod = dyn_cast<MSFTModuleExternOp>(modOp)) {
        addModuleLike(inst, mod.getPorts());
      }
    } else {
      addOther(op);
    }
  }

  //*************
  //   Construct the partition module and replace the design partition op.

  // Build the module.
  auto partMod =
      OpBuilder::atBlockEnd(getOperation().getBody())
          .create<hw::HWModuleOp>(loc, partOp.verilogNameAttr(), ports);
  Block *partBlock = partMod.getBodyBlock();
  partBlock->clear();
  auto partBuilder = OpBuilder::atBlockEnd(partBlock);

  // Replace partOp with an instantion of the partition.
  auto partInst = OpBuilder(partOp).create<hw::InstanceOp>(
      loc, partMod, partOp.getNameAttr(), partInstInputs, ArrayAttr(),
      SymbolTable::getSymbolName(partOp));

  // Replace original ops' outputs with partition outputs.
  assert(partInstOutputs.size() == partInst.getNumResults());
  for (size_t resNum = 0, e = partInstOutputs.size(); resNum < e; ++resNum)
    partInstOutputs[resNum].replaceAllUsesWith(partInst.getResult(resNum));

  //*************
  //   Move the operations!

  // Map the original operation's inputs to block arguments.
  mlir::BlockAndValueMapping mapping;
  assert(partInstInputs.size() == partBlock->getNumArguments());
  for (size_t argNum = 0, e = partInstInputs.size(); argNum < e; ++argNum) {
    mapping.map(partInstInputs[argNum], partBlock->getArgument(argNum));
  }

  // Since the same value can map to multiple outputs, compute the 1-N mapping
  // here.
  DenseMap<Value, SmallVector<int, 1>> resultOutputConnections;
  for (size_t outNum = 0, e = partInstOutputs.size(); outNum < e; ++outNum)
    resultOutputConnections[partInstOutputs[outNum]].push_back(outNum);

  // Hold the block outputs.
  SmallVector<Value, 64> outputs(partInstOutputs.size(), Value());

  // Clone the ops into the partition block. Map the results into the module
  // outputs.
  for (Operation *op : toMove) {
    auto *newOp = partBuilder.insert(op->clone(mapping));
    newOp->removeAttr("targetDesignPartition");
    for (size_t resNum = 0, e = op->getNumResults(); resNum < e; ++resNum)
      for (int outputNum : resultOutputConnections[op->getResult(resNum)])
        outputs[outputNum] = newOp->getResult(resNum);
    op->erase();
  }
  partBuilder.create<hw::OutputOp>(loc, outputs);
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createPartitionPass() {
  return std::make_unique<PartitionPass>();
}
} // namespace msft
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace

void circt::msft::registerMSFTPasses() { registerPasses(); }
