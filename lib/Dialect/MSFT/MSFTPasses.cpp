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

/// TODO: Migrate these to some sort of OpInterface shared with hw.
static bool isAnyModule(Operation *module) {
  return isa<MSFTModuleOp, MSFTModuleExternOp>(module) ||
         hw::isAnyModule(module);
}
hw::ModulePortInfo getModulePortInfo(Operation *op) {
  if (auto mod = dyn_cast<MSFTModuleOp>(op))
    return mod.getPorts();
  if (auto mod = dyn_cast<MSFTModuleExternOp>(op))
    return mod.getPorts();
  return hw::getModulePortInfo(op);
}

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
      if (isa<MSFTDialect>(attr.getValue().getDialect())) {
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
      if (isa<MSFTDialect>(attr.getValue().getDialect()))
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
struct PassCommon {
protected:
  hw::SymbolCache topLevelSyms;
  DenseMap<MSFTModuleOp, SmallVector<InstanceOp, 1>> moduleInstantiations;

  void populateSymbolCache(ModuleOp topMod);
  LogicalResult verifyInstances(ModuleOp topMod);

  // Find all the modules and use the partial order of the instantiation DAG
  // to sort them. If we use this order when "bubbling" up operations, we
  // guarantee one-pass completeness. As a side-effect, populate the module to
  // instantiation sites mapping.
  //
  // Assumption (unchecked): there is not a cycle in the instantiation graph.
  void getAndSortModules(ModuleOp topMod, SmallVectorImpl<MSFTModuleOp> &mods);
  void getAndSortModulesVisitor(MSFTModuleOp mod,
                                SmallVectorImpl<MSFTModuleOp> &mods,
                                DenseSet<MSFTModuleOp> &modsSeen);

  void updateInstances(
      MSFTModuleOp mod, ArrayRef<unsigned> newToOldResultMap,
      llvm::function_ref<void(InstanceOp, InstanceOp, SmallVectorImpl<Value> &)>
          getOperandsFunc);

  static bool isWireManipulationOp(Operation *op) {
    return isa<hw::ArrayConcatOp, hw::ArrayCreateOp, hw::ArrayGetOp,
               hw::ArraySliceOp, hw::StructCreateOp, hw::StructExplodeOp,
               hw::StructExtractOp, hw::StructInjectOp, hw::StructCreateOp,
               hw::ConstantOp>(op);
  }
};
} // anonymous namespace

/// Update all the instantiations of 'mod' to match the port list. For any
/// output ports which survived, automatically map the result according to
/// `newToOldResultMap`. Calls 'getOperandsFunc' with the new instance op, the
/// old instance op, and expects the operand vector to return filled.
/// `getOperandsFunc` can (and often does) modify other operations. The update
/// call deletes the original instance op, so all references are invalidated
/// after this call.
void PassCommon::updateInstances(
    MSFTModuleOp mod, ArrayRef<unsigned> newToOldResultMap,
    llvm::function_ref<void(InstanceOp, InstanceOp, SmallVectorImpl<Value> &)>
        getOperandsFunc) {
  SmallVector<InstanceOp, 1> newInstances;
  for (InstanceOp inst : moduleInstantiations[mod]) {
    assert(inst->getParentOp());
    OpBuilder b(inst);
    auto newInst =
        b.create<InstanceOp>(inst.getLoc(), mod.getType().getResults(),
                             inst.getOperands(), inst->getAttrs());
    for (size_t portNum = 0, e = newToOldResultMap.size(); portNum < e;
         ++portNum) {
      assert(portNum < newInst.getNumResults());
      assert(newToOldResultMap[portNum] < inst.getNumResults());
      inst.getResult(newToOldResultMap[portNum])
          .replaceAllUsesWith(newInst.getResult(portNum));
    }

    SmallVector<Value> newOperands;
    getOperandsFunc(newInst, inst, newOperands);
    newInst->setOperands(newOperands);

    newInstances.push_back(newInst);
    inst->dropAllUses();
    inst->erase();
  }
  moduleInstantiations[mod].swap(newInstances);
}

// Run a post-order DFS.
void PassCommon::getAndSortModulesVisitor(MSFTModuleOp mod,
                                          SmallVectorImpl<MSFTModuleOp> &mods,
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

void PassCommon::getAndSortModules(ModuleOp topMod,
                                   SmallVectorImpl<MSFTModuleOp> &mods) {
  // Add here _before_ we go deeper to prevent infinite recursion.
  DenseSet<MSFTModuleOp> modsSeen;
  mods.clear();
  moduleInstantiations.clear();
  topMod.walk(
      [&](MSFTModuleOp mod) { getAndSortModulesVisitor(mod, mods, modsSeen); });
}

/// Fill a symbol cache with all the top level symbols.
void PassCommon::populateSymbolCache(mlir::ModuleOp mod) {
  for (Operation &op : mod.getBody()->getOperations()) {
    StringAttr symName = SymbolTable::getSymbolName(&op);
    if (!symName)
      continue;
    // Add the symbol to the cache.
    topLevelSyms.addDefinition(symName, &op);
  }
  topLevelSyms.freeze();
}

LogicalResult PassCommon::verifyInstances(mlir::ModuleOp mod) {
  WalkResult r = mod.walk([&](InstanceOp inst) {
    Operation *modOp = topLevelSyms.getDefinition(inst.moduleNameAttr());
    if (!isAnyModule(modOp))
      return WalkResult::interrupt();

    hw::ModulePortInfo ports = getModulePortInfo(modOp);
    return succeeded(inst.verifySignatureMatch(ports))
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return failure(r.wasInterrupted());
}

namespace {
struct PartitionPass : public PartitionBase<PartitionPass>, PassCommon {
  void runOnOperation() override;

private:
  void partition(MSFTModuleOp mod);
  void partition(DesignPartitionOp part, ArrayRef<Operation *> users);

  void bubbleUp(MSFTModuleOp mod, ArrayRef<Operation *> ops);
  void bubbleUpGlobalRefs(Operation *op);
  void pushDownGlobalRefs(Operation *op, DesignPartitionOp partOp,
                          DenseSet<Attribute> &newGlobalRefs);

  // Tag wire manipulation ops connected to this potentially tagged op.
  static void markWireOps(Operation *op);
};
} // anonymous namespace

void PartitionPass::runOnOperation() {
  ModuleOp outerMod = getOperation();
  populateSymbolCache(outerMod);
  if (failed(verifyInstances(outerMod))) {
    signalPassFailure();
    return;
  }

  // Get a properly sorted list, then partition the mods in order.
  SmallVector<MSFTModuleOp, 64> sortedMods;
  getAndSortModules(outerMod, sortedMods);
  for (auto mod : sortedMods)
    partition(mod);
}

void PartitionPass::markWireOps(Operation *taggedOp) {
  auto partRef =
      taggedOp->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
  if (!partRef)
    return;
  SmallVector<Operation *, 8> opQueue;
  opQueue.push_back(taggedOp);

  while (!opQueue.empty()) {
    Operation *op = opQueue.back();
    opQueue.pop_back();

    for (auto *user : op->getUsers()) {
      if (!isWireManipulationOp(user) || user->hasAttr("targetDesignPartition"))
        continue;
      user->setAttr("targetDesignPartition", partRef);
      opQueue.push_back(user);
    }

    for (auto operValue : op->getOperands()) {
      Operation *defOp = operValue.getDefiningOp();
      if (!defOp || !isWireManipulationOp(defOp) ||
          defOp->hasAttr("targetDesignPartition"))
        continue;
      defOp->setAttr("targetDesignPartition", partRef);
      opQueue.push_back(defOp);
    }
  }
}

void PartitionPass::partition(MSFTModuleOp mod) {
  DenseMap<StringAttr, SmallVector<Operation *, 1>> localPartMembers;
  SmallVector<Operation *, 64> nonLocalTaggedOps;

  mod.walk(&markWireOps);

  mod.walk([&](Operation *op) {
    auto partRef = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
    if (!partRef)
      return;

    if (partRef.getRootReference() == SymbolTable::getSymbolName(mod))
      localPartMembers[partRef.getLeafReference()].push_back(op);
    else
      nonLocalTaggedOps.push_back(op);
  });

  if (!nonLocalTaggedOps.empty())
    bubbleUp(mod, nonLocalTaggedOps);

  for (auto part :
       llvm::make_early_inc_range(mod.getOps<DesignPartitionOp>())) {
    auto usersIter = localPartMembers.find(part.sym_nameAttr());
    if (usersIter != localPartMembers.end())
      this->partition(part, usersIter->second);
    part.erase();
  }
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
  if (auto constOp = dyn_cast<hw::ConstantOp>(op)) {
    buff.clear();
    llvm::raw_string_ostream(buff) << "c" << constOp.value();
    return buff;
  }

  if (res.getDefiningOp()->getNumResults() == 1)
    return {};

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

  if (oper.getOwner()->getNumOperands() == 1)
    return "in";

  // Fallback. Not ideal.
  buff.clear();
  llvm::raw_string_ostream(buff) << "in" << oper.getOperandNumber();
  return buff;
}

/// Helper to get the circt.globalRef attribute.
static ArrayAttr getGlobalRefs(Operation *op) {
  return op->getAttrOfType<ArrayAttr>("circt.globalRef");
}

/// Helper to update GlobalRefOps after referenced ops bubble up.
void PartitionPass::bubbleUpGlobalRefs(Operation *op) {
  auto globalRefs = getGlobalRefs(op);
  if (!globalRefs)
    return;

  for (auto globalRef : globalRefs.getAsRange<hw::GlobalRefAttr>()) {
    // Resolve the GlobalRefOp and get its path.
    auto refSymbol = globalRef.getGlblSym();
    auto globalRefOp = dyn_cast_or_null<hw::GlobalRefOp>(
        topLevelSyms.getDefinition(refSymbol));
    assert(globalRefOp && "symbol must reference a GlobalRefOp");
    auto oldPath = globalRefOp.namepath().getAsRange<hw::InnerRefAttr>();

    // Construct a new path starting from the old path.
    SmallVector<Attribute> newPath(oldPath);

    // If there aren't two elements in the path, this GlobalRefOp has already
    // been updated.
    if (newPath.size() < 2)
      return;

    // Splice the last two elements in the path, using the combined name from
    // the op's attributes, and taking the module from the second to last
    // element.
    newPath.pop_back_val().cast<hw::InnerRefAttr>();
    auto nextElement = newPath.pop_back_val().cast<hw::InnerRefAttr>();
    auto newModule = nextElement.getModule();
    auto newNameAttr = StringAttr::get(op->getContext(), ::getOpName(op));
    auto newLeaf = hw::InnerRefAttr::get(newModule, newNameAttr);
    newPath.push_back(newLeaf);

    // GlobalRefs use the inner_sym attribute, so keep it up to date.
    op->setAttr("inner_sym", newNameAttr);

    // Update the path on the GlobalRefOp.
    auto newPathAttr = ArrayAttr::get(op->getContext(), newPath);
    globalRefOp.namepathAttr(newPathAttr);
  }
}

/// Helper to update GlobalRefops after referenced ops are pushed down.
void PartitionPass::pushDownGlobalRefs(Operation *op, DesignPartitionOp partOp,
                                       DenseSet<Attribute> &newGlobalRefs) {
  auto globalRefs = getGlobalRefs(op);
  if (!globalRefs)
    return;

  for (auto globalRef : globalRefs.getAsRange<hw::GlobalRefAttr>()) {
    // Resolve the GlobalRefOp and get its path.
    auto refSymbol = globalRef.getGlblSym();
    auto globalRefOp = dyn_cast_or_null<hw::GlobalRefOp>(
        topLevelSyms.getDefinition(refSymbol));
    assert(globalRefOp && "symbol must reference a GlobalRefOp");
    auto oldPath = globalRefOp.namepath().getAsRange<hw::InnerRefAttr>();

    // Get the module containing the partition and the partition's name.
    auto partAttr = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
    auto partMod = partAttr.getRootReference();
    auto partName = partAttr.getLeafReference();
    auto partModName = partOp.verilogNameAttr();
    assert(partModName);

    // Construct a new path starting from the old path.
    SmallVector<Attribute> newPath(oldPath);
    auto lastElement = newPath.pop_back_val().cast<hw::InnerRefAttr>();

    // If the last element in the path is not pointing at the module
    // containing the partition, this GlobalRefOp has already been updated.
    if (lastElement.getModule() != partMod)
      continue;

    // Split the last element in the path to add the partition.
    auto partRef = hw::InnerRefAttr::get(partMod, partName);
    auto leafRef = hw::InnerRefAttr::get(partModName, lastElement.getName());
    newPath.push_back(partRef);
    newPath.push_back(leafRef);

    // Update the path on the GlobalRefOp.
    auto newPathAttr = ArrayAttr::get(op->getContext(), newPath);
    globalRefOp.namepathAttr(newPathAttr);

    // Ensure the part instance will have this GlobalRefAttr.
    // global refs if not.
    newGlobalRefs.insert(globalRef);
  }
}

void PartitionPass::bubbleUp(MSFTModuleOp mod, ArrayRef<Operation *> ops) {
  // This particular implementation is _very_ sensitive to iteration order. It
  // assumes that the order in which the ops, operands, and results are the same
  // _every_ time it runs through them. Doing this saves on bookkeeping.
  auto *ctxt = mod.getContext();
  FunctionType origType = mod.getType();
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
          StringAttr::get(ctxt, opName + (name.empty() ? "" : "." + name)),
          res.getType()));
    }
    for (OpOperand &oper : op->getOpOperands()) {
      StringRef name = getOperandName(oper, topLevelSyms, nameBuffer);
      newOutputs.push_back(std::make_pair(
          StringAttr::get(ctxt, opName + (name.empty() ? "" : "." + name)),
          oper.get()));
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
  SmallVector<unsigned> resValues;
  for (size_t i = 0, e = origType.getNumResults(); i < e; ++i)
    resValues.push_back(i);
  auto cloneOpsGetOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                                 SmallVectorImpl<Value> &newOperands) {
    OpBuilder b(newInst);

    size_t resultNum = origType.getNumResults();
    auto oldOperands = oldInst->getOperands();
    newOperands.append(oldOperands.begin(), oldOperands.end());
    for (Operation *op : ops) {
      BlockAndValueMapping map;
      for (Value oper : op->getOperands())
        map.map(oper, newInst->getResult(resultNum++));
      Operation *newOp = b.insert(op->clone(map));
      for (Value res : newOp->getResults())
        newOperands.push_back(res);
      setEntityName(newOp, oldInst.getName() + "." + ::getOpName(op));
      bubbleUpGlobalRefs(newOp);
    }
  };
  updateInstances(mod, resValues, cloneOpsGetOperands);

  //*************
  //   Done.
  for (Operation *op : ops)
    op->erase();
}

void PartitionPass::partition(DesignPartitionOp partOp,
                              ArrayRef<Operation *> toMove) {

  auto *ctxt = partOp.getContext();
  auto loc = partOp.getLoc();
  std::string nameBuffer;

  //*************
  //   Determine the partition module's interface. Keep some bookkeeping around.
  SmallVector<hw::PortInfo> inputPorts;
  SmallVector<hw::PortInfo> outputPorts;
  SmallVector<Value, 64> partInstInputs;
  SmallVector<Value, 64> partInstOutputs;

  // Handle module-like operations. Not strictly necessary, but we can base the
  // new portnames on the portnames of the instance being moved and the instance
  // name.
  auto addModuleLike = [&](Operation *inst, Operation *modOp) {
    hw::ModulePortInfo modPorts = getModulePortInfo(modOp);
    StringRef name = ::getOpName(inst);

    for (auto port : modPorts.inputs) {
      partInstInputs.push_back(inst->getOperand(port.argNum));
      inputPorts.push_back(hw::PortInfo{
          /*name*/ StringAttr::get(ctxt, name + "." + port.name.getValue()),
          /*direction*/ hw::PortDirection::INPUT,
          /*type*/ port.type,
          /*argNum*/ inputPorts.size()});
    }

    for (auto port : modPorts.outputs) {
      partInstOutputs.push_back(inst->getResult(port.argNum));
      outputPorts.push_back(hw::PortInfo{
          /*name*/ StringAttr::get(ctxt, name + "." + port.name.getValue()),
          /*direction*/ hw::PortDirection::OUTPUT,
          /*type*/ port.type,
          /*argNum*/ outputPorts.size()});
    }
  };
  // Handle all other operators.
  auto addOther = [&](Operation *op) {
    StringRef name = ::getOpName(op);

    for (OpOperand &oper : op->getOpOperands()) {
      inputPorts.push_back(hw::PortInfo{
          /*name*/ StringAttr::get(
              ctxt,
              name + "." + getOperandName(oper, topLevelSyms, nameBuffer)),
          /*direction*/ hw::PortDirection::INPUT,
          /*type*/ oper.get().getType(),
          /*argNum*/ inputPorts.size()});
      partInstInputs.push_back(oper.get());
    }

    for (OpResult res : op->getOpResults()) {
      StringRef resName = getResultName(res, topLevelSyms, nameBuffer);
      outputPorts.push_back(
          hw::PortInfo{/*name*/ StringAttr::get(
                           ctxt, name + (resName.empty() ? "" : "." + resName)),
                       /*direction*/ hw::PortDirection::OUTPUT,
                       /*type*/ res.getType(),
                       /*argNum*/ outputPorts.size()});
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
  hw::ModulePortInfo modPortInfo(inputPorts, outputPorts);
  auto partMod =
      OpBuilder::atBlockEnd(getOperation().getBody())
          .create<MSFTModuleOp>(loc, partOp.verilogNameAttr(), modPortInfo,
                                ArrayRef<NamedAttribute>{});
  Block *partBlock = partMod.getBodyBlock();
  partBlock->clear();
  auto partBuilder = OpBuilder::atBlockEnd(partBlock);

  // Replace partOp with an instantion of the partition.
  SmallVector<Type> instRetTypes(
      llvm::map_range(partInstOutputs, [](Value v) { return v.getType(); }));
  auto partInst = OpBuilder(partOp).create<InstanceOp>(
      loc, instRetTypes, partOp.getNameAttr(),
      SymbolTable::getSymbolName(partMod), partInstInputs, ArrayAttr(),
      SymbolRefAttr());

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
  // outputs. Push down any global refs to include the partition. Update the
  // partition to include the new set of global refs, and set its inner_sym.
  DenseSet<Attribute> newGlobalRefs;
  for (Operation *op : toMove) {
    Operation *newOp = partBuilder.insert(op->clone(mapping));
    newOp->removeAttr("targetDesignPartition");
    for (size_t resNum = 0, e = op->getNumResults(); resNum < e; ++resNum)
      for (int outputNum : resultOutputConnections[op->getResult(resNum)])
        outputs[outputNum] = newOp->getResult(resNum);
    pushDownGlobalRefs(op, partOp, newGlobalRefs);
  }
  SmallVector<Attribute> newGlobalRefVec(newGlobalRefs.begin(),
                                         newGlobalRefs.end());
  auto newRefsAttr = ArrayAttr::get(partInst->getContext(), newGlobalRefVec);
  partInst->setAttr("circt.globalRef", newRefsAttr);
  partInst->setAttr("inner_sym", partInst.sym_nameAttr());

  partBuilder.create<OutputOp>(loc, outputs);

  // Replace original ops' outputs with partition outputs.
  assert(partInstOutputs.size() == partInst.getNumResults());
  for (size_t resNum = 0, e = partInstOutputs.size(); resNum < e; ++resNum)
    partInstOutputs[resNum].replaceAllUsesWith(partInst.getResult(resNum));

  for (Operation *op : toMove)
    op->erase();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createPartitionPass() {
  return std::make_unique<PartitionPass>();
}
} // namespace msft
} // namespace circt

namespace {
struct WireCleanupPass : public WireCleanupBase<WireCleanupPass>, PassCommon {
  void runOnOperation() override;

private:
  void bubbleWiresUp(MSFTModuleOp mod);
  void sinkWiresDown(MSFTModuleOp mod);
};
} // anonymous namespace

void WireCleanupPass::runOnOperation() {
  ModuleOp topMod = getOperation();
  populateSymbolCache(topMod);
  if (failed(verifyInstances(topMod))) {
    signalPassFailure();
    return;
  }

  SmallVector<MSFTModuleOp> sortedMods;
  getAndSortModules(topMod, sortedMods);

  for (auto mod : sortedMods)
    bubbleWiresUp(mod);

  for (auto mod : llvm::reverse(sortedMods))
    sinkWiresDown(mod);
}

/// Push up any wires which are simply passed-through.
void WireCleanupPass::bubbleWiresUp(MSFTModuleOp mod) {
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();
  hw::ModulePortInfo ports = mod.getPorts();

  // Find all "passthough" internal wires, filling 'inputPortsToRemove' as a
  // side-effect.
  DenseMap<Value, hw::PortInfo> passThroughs;
  llvm::BitVector inputPortsToRemove(ports.inputs.size());
  for (hw::PortInfo inputPort : ports.inputs) {
    BlockArgument portArg = body->getArgument(inputPort.argNum);
    bool removePort = true;
    for (OpOperand user : portArg.getUsers()) {
      if (user.getOwner() == terminator)
        passThroughs[portArg] = inputPort;
      else
        removePort = false;
    }
    if (removePort)
      inputPortsToRemove.set(inputPort.argNum);
  }

  // Find all output ports which we can remove. Fill in 'outputToInputIdx' to
  // help rewire instantiations later on.
  DenseMap<unsigned, unsigned> outputToInputIdx;
  llvm::BitVector outputPortsToRemove(ports.outputs.size());
  for (hw::PortInfo outputPort : ports.outputs) {
    assert(outputPort.argNum < terminator->getNumOperands() && "Invalid IR");
    Value outputValue = terminator->getOperand(outputPort.argNum);
    auto inputNumF = passThroughs.find(outputValue);
    if (inputNumF == passThroughs.end())
      continue;
    hw::PortInfo inputPort = inputNumF->second;
    outputToInputIdx[outputPort.argNum] = inputPort.argNum;
    outputPortsToRemove.set(outputPort.argNum);
  }

  // Use MSFTModuleOp's `removePorts` method to remove the ports. It returns a
  // mapping of the new output port to old output port indices to assist in
  // updating the instantiations later on.
  auto newToOldResult =
      mod.removePorts(inputPortsToRemove, outputPortsToRemove);

  // Update the instantiations.
  auto setPassthroughsGetOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                                        SmallVectorImpl<Value> &newOperands) {
    // Re-map the passthrough values around the instance.
    for (auto idxPair : outputToInputIdx) {
      size_t outputPortNum = idxPair.first;
      assert(outputPortNum <= oldInst.getNumResults());
      size_t inputPortNum = idxPair.second;
      assert(inputPortNum <= oldInst.getNumOperands());
      oldInst.getResult(outputPortNum)
          .replaceAllUsesWith(oldInst.getOperand(inputPortNum));
    }
    // Use a sort-merge-join approach to figure out the operand mapping on the
    // fly.
    for (size_t operNum = 0, e = oldInst.getNumOperands(); operNum < e;
         ++operNum)
      if (!inputPortsToRemove.test(operNum))
        newOperands.push_back(oldInst.getOperand(operNum));
  };
  updateInstances(mod, newToOldResult, setPassthroughsGetOperands);
}

/// Sink all the instance connections which are loops.
void WireCleanupPass::sinkWiresDown(MSFTModuleOp mod) {
  auto instantiations = moduleInstantiations[mod];
  // TODO: remove this limitation. This would involve looking at the common
  // loopbacks for all the instances.
  if (instantiations.size() != 1)
    return;
  InstanceOp inst = instantiations[0];

  // Find all the "loopback" connections in the instantiation. Populate
  // 'inputToOutputLoopback' with a mapping of input port to output port which
  // drives it. Populate 'resultsToErase' with output ports which only drive
  // input ports.
  DenseMap<unsigned, unsigned> inputToOutputLoopback;
  llvm::BitVector resultsToErase(inst.getNumResults());
  for (unsigned resNum = 0, e = inst.getNumResults(); resNum < e; ++resNum) {
    bool allLoops = true;
    for (auto &use : inst.getResult(resNum).getUses()) {
      if (use.getOwner() != inst.getOperation())
        allLoops = false;
      else
        inputToOutputLoopback[use.getOperandNumber()] = resNum;
    }
    if (allLoops)
      resultsToErase.set(resNum);
  }

  // Add internal connections to replace the instantiation's loop back
  // connections.
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();
  llvm::BitVector argsToErase(body->getNumArguments());
  for (auto resOper : inputToOutputLoopback) {
    body->getArgument(resOper.first)
        .replaceAllUsesWith(terminator->getOperand(resOper.second));
    argsToErase.set(resOper.first);
  }

  // Remove the ports.
  SmallVector<unsigned> newToOldResultMap =
      mod.removePorts(argsToErase, resultsToErase);
  // and update the instantiations.
  auto getOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                         SmallVectorImpl<Value> &newOperands) {
    // Use sort-merge-join to compute the new operands;
    for (unsigned argNum = 0, e = oldInst.getNumOperands(); argNum < e;
         ++argNum)
      if (!argsToErase.test(argNum))
        newOperands.push_back(oldInst.getOperand(argNum));
  };
  updateInstances(mod, newToOldResultMap, getOperands);
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createWireCleanupPass() {
  return std::make_unique<WireCleanupPass>();
}
} // namespace msft
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace

void circt::msft::registerMSFTPasses() { registerPasses(); }
