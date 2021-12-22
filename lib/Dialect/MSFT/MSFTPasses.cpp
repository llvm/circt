//===- MSFTPasses.cpp - Implement MSFT passes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
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
/// Simply remove hw::GlobalRefOps for placement when done.
struct GlobalRefOpLowering : public OpConversionPattern<hw::GlobalRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::GlobalRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    for (auto attr : op->getAttrs()) {
      if (attr.getName().getValue().startswith("loc")) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
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
  for (auto moduleName : tops) {
    auto hwmod = top.lookupSymbol<msft::MSFTModuleOp>(moduleName);
    if (!hwmod)
      continue;
    if (failed(exportQuartusTcl(hwmod, tclFile)))
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
  target.addDynamicallyLegalOp<hw::GlobalRefOp>([](hw::GlobalRefOp op) {
    for (auto attr : op->getAttrs())
      if (attr.getName().getValue().startswith("loc"))
        return false;
    return true;
  });

  RewritePatternSet patterns(ctxt);
  patterns.insert<ModuleOpLowering>(ctxt, verilogFile);
  patterns.insert<ModuleExternOpLowering>(ctxt, verilogFile);
  patterns.insert<OutputOpLowering>(ctxt);
  patterns.insert<GlobalRefOpLowering>(ctxt);

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
  DenseMap<MSFTModuleOp, SmallVector<InstanceOp, 1>> moduleInstantiations;

  void partition(MSFTModuleOp mod);
  void partition(DesignPartitionOp part, SmallVectorImpl<Operation *> &users);

  void bubbleUp(MSFTModuleOp mod, ArrayRef<Operation *> ops);

  // Find all the modules and use the partial order of the instantiation DAG
  // to sort them. If we use this order when "bubbling" up operations, we
  // guarantee one-pass completeness. As a side-effect, populate the module to
  // instantiation sites mapping.
  //
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
    moduleInstantiations[mod].push_back(inst);
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
  DenseMap<StringAttr, SmallVector<Operation *, 1>> localPartMembers;
  SmallVector<Operation *, 64> nonLocalTaggedOps;
  mod.walk([&](Operation *op) {
    auto partRef = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
    if (!partRef)
      return;

    if (partRef.getRootReference() == SymbolTable::getSymbolName(mod))
      localPartMembers[partRef.getLeafReference()].push_back(op);
    else
      nonLocalTaggedOps.push_back(op);
  });

  bubbleUp(mod, nonLocalTaggedOps);

  for (auto part :
       llvm::make_early_inc_range(mod.getOps<DesignPartitionOp>())) {
    auto usersIter = localPartMembers.find(part.sym_nameAttr());
    if (usersIter != localPartMembers.end())
      this->partition(part, usersIter->second);
    part.erase();
  }
}

/// TODO: Migrate these to some sort of OpInterface shared with hw.
static bool isAnyModule(Operation *module) {
  return isa<MSFTModuleOp>(module) || isa<MSFTModuleExternOp>(module) ||
         hw::isAnyModule(module);
}
hw::ModulePortInfo getModulePortInfo(Operation *op) {
  if (auto mod = dyn_cast<MSFTModuleOp>(op))
    return mod.getPorts();
  if (auto mod = dyn_cast<MSFTModuleExternOp>(op))
    return mod.getPorts();
  return hw::getModulePortInfo(op);
}

/// Heuristics to get the entity name.
static StringRef getOpName(Operation *op) {
  StringAttr name;
  if ((name = op->getAttrOfType<StringAttr>("name")) && name.size())
    return name.getValue();
  if ((name = op->getAttrOfType<StringAttr>("sym_name")) && name.size())
    return name.getValue();
  return op->getName().getStringRef();
}
/// Try to set the entity name.
/// TODO: this needs to be more complex to deal with renaming symbols.
static void setEntityName(Operation *op, Twine name) {
  StringAttr nameAttr = StringAttr::get(op->getContext(), name);
  if (op->hasAttrOfType<StringAttr>("name"))
    op->setAttr("name", nameAttr);
  if (op->hasAttrOfType<StringAttr>("sym_name"))
    op->setAttr("sym_name", nameAttr);
}
/// Heuristics to get the output name.
static StringRef getResultName(OpResult res, const hw::SymbolCache &syms,
                               std::string &buff) {
  Operation *op = res.getDefiningOp();
  if (auto inst = dyn_cast<InstanceOp>(op)) {
    Operation *modOp = syms.getDefinition(inst.moduleNameAttr());
    assert(modOp && "Invalid IR");
    assert(isAnyModule(modOp) && "Instance must point to a module");
    hw::ModulePortInfo ports = getModulePortInfo(modOp);
    return ports.outputs[res.getResultNumber()].name;
  }
  if (auto asmInterface = dyn_cast<mlir::OpAsmOpInterface>(op)) {
    StringRef retName;
    asmInterface.getAsmResultNames([&](Value v, StringRef name) {
      if (v == res && !name.empty())
        retName = StringAttr::get(op->getContext(), name).getValue();
    });
    if (!retName.empty())
      return retName;
  }

  // Fallback. Not ideal.
  buff.clear();
  llvm::raw_string_ostream(buff) << "out" << res.getResultNumber();
  return buff;
}

/// Heuristics to get the input name.
static StringRef getOperandName(OpOperand &oper, const hw::SymbolCache &syms,
                                std::string &buff) {
  Operation *op = oper.getOwner();
  if (auto inst = dyn_cast<InstanceOp>(op)) {
    Operation *modOp = syms.getDefinition(inst.moduleNameAttr());
    assert(modOp && "Invalid IR");
    assert(isAnyModule(modOp) && "Instance must point to a module");
    hw::ModulePortInfo ports = getModulePortInfo(modOp);
    return ports.inputs[oper.getOperandNumber()].name;
  }

  // Fallback. Not ideal.
  buff.clear();
  llvm::raw_string_ostream(buff) << "in" << oper.getOperandNumber();
  return buff;
}

void PartitionPass::bubbleUp(MSFTModuleOp mod, ArrayRef<Operation *> ops) {
  // This particular implementation is _very_ sensitive to iteration order. It
  // assumes that the order in which the ops, operands, and results are the same
  // _every_ time it runs through them. Doing this saves on bookkeeping.
  auto *ctxt = mod.getContext();
  std::string nameBuffer;

  //*************
  //   Figure out all the new ports 'mod' is going to need. The outputs need to
  //   know where they're being driven from, which'll be the outputs of 'ops'.
  SmallVector<std::pair<StringAttr, Type>, 64> newInputs;
  SmallVector<std::pair<StringAttr, Value>, 64> newOutputs;
  SmallVector<Type, 64> newResTypes;

  for (Operation *op : ops) {
    StringRef opName = ::getOpName(op);
    for (OpResult res : op->getOpResults()) {
      StringRef name = getResultName(res, topLevelSyms, nameBuffer);
      newInputs.push_back(std::make_pair(
          StringAttr::get(ctxt, opName + "." + name), res.getType()));
    }
    for (OpOperand &oper : op->getOpOperands()) {
      StringRef name = getOperandName(oper, topLevelSyms, nameBuffer);
      newOutputs.push_back(std::make_pair(
          StringAttr::get(ctxt, opName + "." + name), oper.get()));
      newResTypes.push_back(oper.get().getType());
    }
  }

  //*************
  //   Add them to 'mod' and replace all of the old 'ops' results with the new
  //   ports.
  SmallVector<BlockArgument> newBlockArgs = mod.addPorts(newInputs, newOutputs);
  size_t blockArgNum = 0;
  for (Operation *op : ops)
    for (OpResult res : op->getResults())
      res.replaceAllUsesWith(newBlockArgs[blockArgNum++]);

  //*************
  //   For all of the instantiation sites (for 'mod'):
  //     - Create a new instance with the correct result types.
  //     - Clone in 'ops'.
  //     - Construct the new instance operands from the old ones + the cloned
  //     ops' results.
  for (InstanceOp inst : moduleInstantiations[mod]) {
    OpBuilder b(inst);

    // Since we only have to add result types, just copy most everything.
    SmallVector<Type, 64> resTypes(inst.getResultTypes());
    resTypes.append(newResTypes);
    auto newInst = cast<InstanceOp>(b.insert(Operation::create(
        OperationState(inst->getLoc(), inst->getName().getStringRef(),
                       inst->getOperands(), resTypes, inst->getAttrs()))));
    size_t resultNum = 0;
    for (Value origRes : inst.getResults())
      origRes.replaceAllUsesWith(newInst->getResult(resultNum++));

    SmallVector<Value, 64> newOperands(inst->getOperands());
    for (Operation *op : ops) {
      BlockAndValueMapping map;
      for (Value oper : op->getOperands())
        map.map(oper, newInst->getResult(resultNum++));
      Operation *newOp = b.insert(op->clone(map));
      for (Value res : newOp->getResults())
        newOperands.push_back(res);
      setEntityName(newOp, inst.getName() + "." + ::getOpName(op));
    }
    newInst->setOperands(newOperands);
    inst.erase();
  }

  //*************
  //   Done.
  for (Operation *op : ops)
    op->erase();
}

void PartitionPass::partition(DesignPartitionOp partOp,
                              SmallVectorImpl<Operation *> &toMove) {

  auto *ctxt = partOp.getContext();
  auto loc = partOp.getLoc();
  std::string nameBuffer;

  //*************
  //   Determine the partition module's interface. Keep some bookkeeping around.
  SmallVector<hw::PortInfo, 128> ports;
  SmallVector<Value, 64> partInstInputs;
  SmallVector<Value, 64> partInstOutputs;

  // Handle module-like operations. Not strictly necessary, but we can base the
  // new portnames on the portnames of the instance being moved and the instance
  // name.
  auto addModuleLike = [&](Operation *inst, Operation *modOp) {
    hw::ModulePortInfo modPorts = getModulePortInfo(modOp);
    StringRef name = ::getOpName(inst);

    for (auto port :
         llvm::concat<hw::PortInfo>(modPorts.inputs, modPorts.outputs)) {
      ports.push_back(hw::PortInfo{
          /*name*/ StringAttr::get(ctxt, name + "." + port.name.getValue()),
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
  auto addOther = [&](Operation *op) {
    StringRef name = ::getOpName(op);

    for (OpOperand &oper : op->getOpOperands()) {
      ports.push_back(hw::PortInfo{
          /*name*/ StringAttr::get(
              ctxt,
              name + "." + getOperandName(oper, topLevelSyms, nameBuffer)),
          /*direction*/ hw::PortDirection::INPUT,
          /*type*/ oper.get().getType(),
          /*argNum*/ ports.size()});
      partInstInputs.push_back(oper.get());
    }

    for (OpResult res : op->getOpResults()) {
      ports.push_back(hw::PortInfo{
          /*name*/ StringAttr::get(
              ctxt, name + "." + getResultName(res, topLevelSyms, nameBuffer)),
          /*direction*/ hw::PortDirection::OUTPUT,
          /*type*/ res.getType(),
          /*argNum*/ ports.size()});
      partInstOutputs.push_back(res);
    }
  };

  // Aggregate the args/results into partition module ports.
  for (Operation *op : toMove) {
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      Operation *modOp = topLevelSyms.getDefinition(inst.moduleNameAttr());
      assert(modOp && "Module instantiated should exist. Verifier should have "
                      "caught this.");

      if (isAnyModule(modOp))
        addModuleLike(inst, modOp);
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
    Operation *newOp = partBuilder.insert(op->clone(mapping));
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
