//===- LowerAnnotations.cpp - Lower Annotations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerAnnotations pass.  This pass processes FIRRTL
// annotations, rewriting them, scattering them, and dealing with non-local
// annotations.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-annos"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

/// Get annotations or an empty set of annotations.
static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annots = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annots;
  return ArrayAttr::get(op->getContext(), {});
}

/// Construct the annotation array with a new thing appended.
static ArrayAttr appendArrayAttr(ArrayAttr array, Attribute a) {
  if (!array)
    return ArrayAttr::get(a.getContext(), ArrayRef<Attribute>{a});
  SmallVector<Attribute> old(array.begin(), array.end());
  old.push_back(a);
  return ArrayAttr::get(a.getContext(), old);
}

/// Update an ArrayAttribute by replacing one entry.
static ArrayAttr replaceArrayAttrElement(ArrayAttr array, size_t elem,
                                         Attribute newVal) {
  SmallVector<Attribute> old(array.begin(), array.end());
  old[elem] = newVal;
  return ArrayAttr::get(array.getContext(), old);
}

/// Apply a new annotation to a resolved target.  This handles ports,
/// aggregates, modules, wires, etc.
static void addAnnotation(AnnoTarget ref, unsigned fieldIdx,
                          ArrayRef<NamedAttribute> anno) {
  auto *context = ref.getOp()->getContext();
  DictionaryAttr annotation;
  if (fieldIdx) {
    SmallVector<NamedAttribute> annoField(anno.begin(), anno.end());
    annoField.emplace_back(
        StringAttr::get(context, "circt.fieldID"),
        IntegerAttr::get(IntegerType::get(context, 32, IntegerType::Signless),
                         fieldIdx));
    annotation = DictionaryAttr::get(context, annoField);
  } else {
    annotation = DictionaryAttr::get(context, anno);
  }

  if (ref.isa<OpAnnoTarget>()) {
    auto newAnno = appendArrayAttr(getAnnotationsFrom(ref.getOp()), annotation);
    ref.getOp()->setAttr(getAnnotationAttrName(), newAnno);
    return;
  }

  auto portRef = ref.cast<PortAnnoTarget>();
  auto portAnnoRaw = ref.getOp()->getAttr(getPortAnnotationAttrName());
  ArrayAttr portAnno = portAnnoRaw.dyn_cast_or_null<ArrayAttr>();
  if (!portAnno || portAnno.size() != getNumPorts(ref.getOp())) {
    SmallVector<Attribute> emptyPortAttr(
        getNumPorts(ref.getOp()),
        ArrayAttr::get(ref.getOp()->getContext(), {}));
    portAnno = ArrayAttr::get(ref.getOp()->getContext(), emptyPortAttr);
  }
  portAnno = replaceArrayAttrElement(
      portAnno, portRef.getPortNo(),
      appendArrayAttr(portAnno[portRef.getPortNo()].dyn_cast<ArrayAttr>(),
                      annotation));
  ref.getOp()->setAttr("portAnnotations", portAnno);
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
static FlatSymbolRefAttr buildNLA(const AnnoPathValue &target,
                                  ApplyState &state) {
  OpBuilder b(state.circuit.getBodyRegion());
  SmallVector<Attribute> insts;
  for (auto inst : target.instances) {
    insts.push_back(OpAnnoTarget(inst).getNLAReference(
        state.getNamespace(inst->getParentOfType<FModuleLike>())));
  }

  insts.push_back(
      FlatSymbolRefAttr::get(target.ref.getModule().moduleNameAttr()));

  auto instAttr = ArrayAttr::get(state.circuit.getContext(), insts);

  // Re-use NLA for this path if already created.
  auto it = state.instPathToNLAMap.find(instAttr);
  if (it != state.instPathToNLAMap.end()) {
    ++state.numReusedHierPaths;
    return it->second;
  }

  // Create the NLA
  auto nla = b.create<hw::HierPathOp>(state.circuit.getLoc(), "nla", instAttr);
  state.symTbl.insert(nla);
  nla.setVisibility(SymbolTable::Visibility::Private);
  auto sym = FlatSymbolRefAttr::get(nla);
  state.instPathToNLAMap.insert({instAttr, sym});
  return sym;
}

/// Scatter breadcrumb annotations corresponding to non-local annotations
/// along the instance path.  Returns symbol name used to anchor annotations to
/// path.
// FIXME: uniq annotation chain links
static FlatSymbolRefAttr scatterNonLocalPath(const AnnoPathValue &target,
                                             ApplyState &state) {

  FlatSymbolRefAttr sym = buildNLA(target, state);
  return sym;
}

//===----------------------------------------------------------------------===//
// Standard Utility Resolvers
//===----------------------------------------------------------------------===//

/// Always resolve to the circuit, ignoring the annotation.
static std::optional<AnnoPathValue> noResolve(DictionaryAttr anno,
                                              ApplyState &state) {
  return AnnoPathValue(state.circuit);
}

/// Implementation of standard resolution.  First parses the target path, then
/// resolves it.
static std::optional<AnnoPathValue> stdResolveImpl(StringRef rawPath,
                                                   ApplyState &state) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  auto tokens = tokenizePath(path);
  if (!tokens) {
    mlir::emitError(state.circuit.getLoc())
        << "Cannot tokenize annotation path " << rawPath;
    return {};
  }

  return resolveEntities(*tokens, state.circuit, state.symTbl,
                         state.targetCaches);
}

/// (SFC) FIRRTL SingleTargetAnnotation resolver.  Uses the 'target' field of
/// the annotation with standard parsing to resolve the path.  This requires
/// 'target' to exist and be normalized (per docs/FIRRTLAnnotations.md).
static std::optional<AnnoPathValue> stdResolve(DictionaryAttr anno,
                                               ApplyState &state) {
  auto target = anno.getNamed("target");
  if (!target) {
    mlir::emitError(state.circuit.getLoc())
        << "No target field in annotation " << anno;
    return {};
  }
  if (!target->getValue().isa<StringAttr>()) {
    mlir::emitError(state.circuit.getLoc())
        << "Target field in annotation doesn't contain string " << anno;
    return {};
  }
  return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                        state);
}

/// Resolves with target, if it exists.  If not, resolves to the circuit.
static std::optional<AnnoPathValue> tryResolve(DictionaryAttr anno,
                                               ApplyState &state) {
  auto target = anno.getNamed("target");
  if (target)
    return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                          state);
  return AnnoPathValue(state.circuit);
}

//===----------------------------------------------------------------------===//
// Standard Utility Appliers
//===----------------------------------------------------------------------===//

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotation.  Optionally handles non-local annotations.
static LogicalResult applyWithoutTargetImpl(const AnnoPathValue &target,
                                            DictionaryAttr anno,
                                            ApplyState &state,
                                            bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal()) {
    Annotation annotation(anno);
    auto diag = mlir::emitError(target.ref.getOp()->getLoc())
                << "is targeted by a non-local annotation \""
                << annotation.getClass() << "\" with target "
                << annotation.getMember("target")
                << ", but this annotation cannot be non-local";
    diag.attachNote() << "see current annotation: " << anno << "\n";
    return failure();
  }
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno) {
    if (na.getName().getValue() != "target") {
      newAnnoAttrs.push_back(na);
    } else if (!target.isLocal()) {
      auto sym = scatterNonLocalPath(target, state);
      newAnnoAttrs.push_back(
          {StringAttr::get(anno.getContext(), "circt.nonlocal"), sym});
    }
  }
  addAnnotation(target.ref, target.fieldIdx, newAnnoAttrs);
  return success();
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotation.  Optionally handles non-local annotations.
/// Ensures the target resolves to an expected type of operation.
template <bool allowNonLocal, bool allowPortAnnoTarget, typename T,
          typename... Tr>
static LogicalResult applyWithoutTarget(const AnnoPathValue &target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  if (target.ref.isa<PortAnnoTarget>()) {
    if (!allowPortAnnoTarget)
      return failure();
  } else if (!target.isOpOfType<T, Tr...>())
    return failure();

  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

template <bool allowNonLocal, typename T, typename... Tr>
static LogicalResult applyWithoutTarget(const AnnoPathValue &target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  return applyWithoutTarget<allowNonLocal, false, T, Tr...>(target, anno,
                                                            state);
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
template <bool allowNonLocal = false>
static LogicalResult applyWithoutTarget(const AnnoPathValue &target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

/// Just drop the annotation.  This is intended for Annotations which are known,
/// but can be safely ignored.
static LogicalResult drop(const AnnoPathValue &target, DictionaryAttr anno,
                          ApplyState &state) {
  return success();
}

//===----------------------------------------------------------------------===//
// Driving table
//===----------------------------------------------------------------------===//

namespace {
struct AnnoRecord {
  llvm::function_ref<std::optional<AnnoPathValue>(DictionaryAttr, ApplyState &)>
      resolver;
  llvm::function_ref<LogicalResult(const AnnoPathValue &, DictionaryAttr,
                                   ApplyState &)>
      applier;
};

/// Resolution and application of a "firrtl.annotations.NoTargetAnnotation".
/// This should be used for any Annotation which does not apply to anything in
/// the FIRRTL Circuit, i.e., an Annotation which has no target.  Historically,
/// NoTargetAnnotations were used to control the Scala FIRRTL Compiler (SFC) or
/// its passes, e.g., to set the output directory or to turn on a pass.
/// Examples of these in the SFC are "firrtl.options.TargetDirAnnotation" to set
/// the output directory or "firrtl.stage.RunFIRRTLTransformAnnotation" to
/// cause the SFC to schedule a specified pass.  Instead of leaving these
/// floating or attaching them to the top-level MLIR module (which is a purer
/// interpretation of "no target"), we choose to attach them to the Circuit even
/// they do not "apply" to the Circuit.  This gives later passes a common place,
/// the Circuit, to search for these control Annotations.
static AnnoRecord NoTargetAnnotation = {noResolve,
                                        applyWithoutTarget<false, CircuitOp>};

} // end anonymous namespace

static const llvm::StringMap<AnnoRecord> annotationRecords{{

    // Testing Annotation
    {"circt.test", {stdResolve, applyWithoutTarget<true>}},
    {"circt.testLocalOnly", {stdResolve, applyWithoutTarget<>}},
    {"circt.testNT", {noResolve, applyWithoutTarget<>}},
    {"circt.missing", {tryResolve, applyWithoutTarget<true>}},
    {"circt.Intrinsic", {stdResolve, applyWithoutTarget<false, FExtModuleOp>}},
    // Grand Central Views/Interfaces Annotations
    {extractGrandCentralClass, NoTargetAnnotation},
    {grandCentralHierarchyFileAnnoClass, NoTargetAnnotation},
    {serializedViewAnnoClass, {noResolve, applyGCTView}},
    {viewAnnoClass, {noResolve, applyGCTView}},
    {companionAnnoClass, {stdResolve, applyWithoutTarget<>}},
    {augmentedGroundTypeClass, {stdResolve, applyWithoutTarget<true>}},
    // Grand Central Data Tap Annotations
    {dataTapsClass, {noResolve, applyGCTDataTaps}},
    {dataTapsBlackboxClass, {stdResolve, applyWithoutTarget<true>}},
    {referenceKeySourceClass, {stdResolve, applyWithoutTarget<true>}},
    {referenceKeyPortClass, {stdResolve, applyWithoutTarget<true>}},
    {internalKeySourceClass, {stdResolve, applyWithoutTarget<true>}},
    {internalKeyPortClass, {stdResolve, applyWithoutTarget<true>}},
    {deletedKeyClass, {stdResolve, applyWithoutTarget<true>}},
    {literalKeyClass, {stdResolve, applyWithoutTarget<true>}},
    // Grand Central Mem Tap Annotations
    {memTapClass, {noResolve, applyGCTMemTaps}},
    {memTapSourceClass, {stdResolve, applyWithoutTarget<true>}},
    {memTapPortClass, {stdResolve, applyWithoutTarget<true>}},
    {memTapBlackboxClass, {stdResolve, applyWithoutTarget<true>}},
    // OMIR Annotations
    {omirAnnoClass, {noResolve, applyOMIR}},
    {omirTrackerAnnoClass, {stdResolve, applyWithoutTarget<true>}},
    {omirFileAnnoClass, NoTargetAnnotation},
    // Miscellaneous Annotations
    {dontTouchAnnoClass,
     {stdResolve, applyWithoutTarget<true, true, WireOp, NodeOp, RegOp,
                                     RegResetOp, InstanceOp, MemOp, CombMemOp,
                                     MemoryPortOp, SeqMemOp>}},
    {prefixModulesAnnoClass,
     {stdResolve,
      applyWithoutTarget<true, FModuleOp, FExtModuleOp, InstanceOp>}},
    {dutAnnoClass, {stdResolve, applyWithoutTarget<false, FModuleOp>}},
    {extractSeqMemsAnnoClass, NoTargetAnnotation},
    {injectDUTHierarchyAnnoClass, NoTargetAnnotation},
    {convertMemToRegOfVecAnnoClass, NoTargetAnnotation},
    {excludeMemToRegAnnoClass,
     {stdResolve, applyWithoutTarget<true, MemOp, CombMemOp>}},
    {sitestBlackBoxAnnoClass, NoTargetAnnotation},
    {enumComponentAnnoClass, {noResolve, drop}},
    {enumDefAnnoClass, {noResolve, drop}},
    {enumVecAnnoClass, {noResolve, drop}},
    {forceNameAnnoClass,
     {stdResolve, applyWithoutTarget<true, FModuleOp, FExtModuleOp>}},
    {flattenAnnoClass, {stdResolve, applyWithoutTarget<false, FModuleOp>}},
    {inlineAnnoClass, {stdResolve, applyWithoutTarget<false, FModuleOp>}},
    {noDedupAnnoClass,
     {stdResolve, applyWithoutTarget<false, FModuleOp, FExtModuleOp>}},
    {blackBoxInlineAnnoClass,
     {stdResolve, applyWithoutTarget<false, FExtModuleOp>}},
    {dontObfuscateModuleAnnoClass,
     {stdResolve, applyWithoutTarget<false, FModuleOp>}},
    {verifBlackBoxAnnoClass,
     {stdResolve, applyWithoutTarget<false, FExtModuleOp>}},
    {elaborationArtefactsDirectoryAnnoClass, NoTargetAnnotation},
    {subCircuitsTargetDirectoryAnnoClass, NoTargetAnnotation},
    {retimeModulesFileAnnoClass, NoTargetAnnotation},
    {retimeModuleAnnoClass,
     {stdResolve, applyWithoutTarget<false, FModuleOp, FExtModuleOp>}},
    {metadataDirectoryAttrName, NoTargetAnnotation},
    {moduleHierAnnoClass, NoTargetAnnotation},
    {sitestTestHarnessBlackBoxAnnoClass, NoTargetAnnotation},
    {testBenchDirAnnoClass, NoTargetAnnotation},
    {testHarnessHierAnnoClass, NoTargetAnnotation},
    {testHarnessPathAnnoClass, NoTargetAnnotation},
    {prefixInterfacesAnnoClass, NoTargetAnnotation},
    {subCircuitDirAnnotation, NoTargetAnnotation},
    {extractAssertAnnoClass, NoTargetAnnotation},
    {extractAssumeAnnoClass, NoTargetAnnotation},
    {extractCoverageAnnoClass, NoTargetAnnotation},
    {dftTestModeEnableAnnoClass, {stdResolve, applyWithoutTarget<true>}},
    {runFIRRTLTransformAnnoClass, {noResolve, drop}},
    {mustDedupAnnoClass, NoTargetAnnotation},
    {addSeqMemPortAnnoClass, NoTargetAnnotation},
    {addSeqMemPortsFileAnnoClass, NoTargetAnnotation},
    {extractClockGatesAnnoClass, NoTargetAnnotation},
    {extractBlackBoxAnnoClass, {stdResolve, applyWithoutTarget<false>}},
    {fullAsyncResetAnnoClass, {stdResolve, applyWithoutTarget<true>}},
    {ignoreFullAsyncResetAnnoClass,
     {stdResolve, applyWithoutTarget<true, FModuleOp>}},
    {decodeTableAnnotation, {noResolve, drop}},
    {blackBoxTargetDirAnnoClass, NoTargetAnnotation},
    {traceNameAnnoClass, {stdResolve, applyTraceName}},
    {traceAnnoClass, {stdResolve, applyWithoutTarget<true>}},

}};

/// Lookup a record for a given annotation class.  Optionally, returns the
/// record for "circuit.missing" if the record doesn't exist.
static const AnnoRecord *getAnnotationHandler(StringRef annoStr,
                                              bool ignoreUnhandledAnno) {
  auto ii = annotationRecords.find(annoStr);
  if (ii != annotationRecords.end())
    return &ii->second;
  if (ignoreUnhandledAnno)
    return &annotationRecords.find("circt.missing")->second;
  return nullptr;
}

bool firrtl::isAnnoClassLowered(StringRef className) {
  return annotationRecords.count(className);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerAnnotationsPass
    : public LowerFIRRTLAnnotationsBase<LowerAnnotationsPass> {
  void runOnOperation() override;
  LogicalResult applyAnnotation(DictionaryAttr anno, ApplyState &state);
  LogicalResult solveWiringProblems(ApplyState &state);

  bool ignoreUnhandledAnno = false;
  bool ignoreClasslessAnno = false;
  bool noRefTypePorts = false;
  SmallVector<DictionaryAttr> worklistAttrs;
};
} // end anonymous namespace

LogicalResult LowerAnnotationsPass::applyAnnotation(DictionaryAttr anno,
                                                    ApplyState &state) {
  LLVM_DEBUG(llvm::dbgs() << "  - anno: " << anno << "\n";);

  // Lookup the class
  StringRef annoClassVal;
  if (auto annoClass = anno.getNamed("class"))
    annoClassVal = annoClass->getValue().cast<StringAttr>().getValue();
  else if (ignoreClasslessAnno)
    annoClassVal = "circt.missing";
  else
    return mlir::emitError(state.circuit.getLoc())
           << "Annotation without a class: " << anno;

  // See if we handle the class
  auto *record = getAnnotationHandler(annoClassVal, false);
  if (!record) {
    ++numUnhandled;
    if (!ignoreUnhandledAnno)
      return mlir::emitError(state.circuit.getLoc())
             << "Unhandled annotation: " << anno;

    // Try again, requesting the fallback handler.
    record = getAnnotationHandler(annoClassVal, ignoreUnhandledAnno);
    assert(record);
  }

  // Try to apply the annotation
  auto target = record->resolver(anno, state);
  if (!target)
    return mlir::emitError(state.circuit.getLoc())
           << "Unable to resolve target of annotation: " << anno;
  if (record->applier(*target, anno, state).failed())
    return mlir::emitError(state.circuit.getLoc())
           << "Unable to apply annotation: " << anno;
  return success();
}

/// Modify the circuit to solve and apply all Wiring Problems in the circuit.  A
/// Wiring Problem is a mapping from a source to a sink that should be connected
/// via a RefType.  This uses a two-step approach.  First, all Wiring Problems
/// are analyzed to compute pending modifications to modules.  Second, modules
/// are visited from leaves to roots to apply module modifications.  Module
/// modifications include addings ports and connecting things up.
LogicalResult LowerAnnotationsPass::solveWiringProblems(ApplyState &state) {
  // Utility function to extract the defining module from a value which may be
  // either a BlockArgument or an Operation result.
  auto getModule = [](Value value) {
    if (BlockArgument blockArg = value.dyn_cast<BlockArgument>())
      return cast<FModuleLike>(blockArg.getParentBlock()->getParentOp());
    return value.getDefiningOp()->getParentOfType<FModuleLike>();
  };

  // Utility function to determine where to insert connection operations.
  auto findInsertionBlock = [&getModule](Value src, Value dest) -> Block * {
    // Check for easy case: both are in the same block.
    if (src.getParentBlock() == dest.getParentBlock())
      return src.getParentBlock();

    // If connecting across blocks, figure out where to connect.
    assert(getModule(src) == getModule(dest));
    // Helper to determine if 'a' is available at 'b's block.
    auto safelyDoms = [&](Value a, Value b) {
      if (a.isa<BlockArgument>())
        return true;
      if (b.isa<BlockArgument>())
        return false;
      // Handle cases where 'b' is in child op after 'a'.
      auto *ancestor =
          a.getParentBlock()->findAncestorOpInBlock(*b.getDefiningOp());
      return ancestor && a.getDefiningOp()->isBeforeInBlock(ancestor);
    };
    if (safelyDoms(src, dest))
      return dest.getParentBlock();
    if (safelyDoms(dest, src))
      return src.getParentBlock();
    return {};
  };

  auto getNoopCast = [](Value v) -> mlir::UnrealizedConversionCastOp {
    auto op =
        dyn_cast_or_null<mlir::UnrealizedConversionCastOp>(v.getDefiningOp());
    if (op && op.getNumResults() == 1 && op.getNumOperands() == 1 &&
        op.getResultTypes()[0] == op.getOperandTypes()[0])
      return op;
    return {};
  };

  // Utility function to connect a destination to a source.  Always use a
  // ConnectOp as the widths may be uninferred.
  SmallVector<Operation *> opsToErase;
  auto connect = [&](Value src, Value dest,
                     ImplicitLocOpBuilder &builder) -> LogicalResult {
    // Strip away noop unrealized_conversion_cast's, used as placeholders.
    // In the future, these should be created/managed as part of creating WP's.
    if (auto op = getNoopCast(dest)) {
      dest = op.getOperand(0);
      opsToErase.push_back(op);
      std::swap(src, dest);
    } else if (auto op = getNoopCast(src)) {
      src = op.getOperand(0);
      opsToErase.push_back(op);
    }

    if (foldFlow(dest) == Flow::Source)
      std::swap(src, dest);

    // Figure out where to insert operations.
    auto *insertBlock = findInsertionBlock(src, dest);
    if (!insertBlock)
      return emitError(src.getLoc())
          .append("This value is involved with a Wiring Problem where the "
                  "destination is in the same module but neither dominates the "
                  "other, which is not supported.")
          .attachNote(dest.getLoc())
          .append("The destination is here.");

    // Insert at end, past invalidation in same block.
    builder.setInsertionPointToEnd(insertBlock);

    // Create RefSend/RefResolve if necessary.
    if (isa<RefType>(dest.getType()) != isa<RefType>(src.getType())) {
      if (isa<RefType>(dest.getType()))
        src = builder.create<RefSendOp>(src);
      else
        src = builder.create<RefResolveOp>(src);
    }

    // If the sink is a wire with no users, then convert this to a node.
    auto destOp = dyn_cast_or_null<WireOp>(dest.getDefiningOp());
    if (destOp && dest.getUses().empty()) {
      builder.create<NodeOp>(src.getType(), src, destOp.getName())
          .setAnnotationsAttr(destOp.getAnnotations());
      opsToErase.push_back(destOp);
      return success();
    }
    // Otherwise, just connect to the source.
    builder.create<ConnectOp>(dest, src);
    return success();
  };

  auto &instanceGraph = state.instancePathCache.instanceGraph;
  auto *context = state.circuit.getContext();

  // Examine all discovered Wiring Problems to determine modifications that need
  // to be made per-module.
  LLVM_DEBUG({ llvm::dbgs() << "Analyzing wiring problems:\n"; });
  DenseMap<FModuleLike, ModuleModifications> moduleModifications;
  DenseSet<Value> visitedSinks;
  for (auto &e : llvm::enumerate(state.wiringProblems)) {
    auto index = e.index();
    auto problem = e.value();
    // This is a unique index that is assigned to this specific wiring problem
    // and is used as a key during wiring to know which Values (ports, sources,
    // or sinks) should be connected.
    auto source = problem.source;
    auto sink = problem.sink;

    // Check that no WiringProblems are trying to use the same sink.  This
    // should never happen.
    if (!visitedSinks.insert(sink).second) {
      auto diag = mlir::emitError(source.getLoc())
                  << "This sink is involved with a Wiring Problem which is "
                     "targeted by a source used by another Wiring Problem. "
                     "(This is both illegal and should be impossible.)";
      diag.attachNote(source.getLoc()) << "The source is here";
      return failure();
    }
    FModuleLike sourceModule = getModule(source);
    FModuleLike sinkModule = getModule(sink);
    if (isa<FExtModuleOp>(sourceModule) || isa<FExtModuleOp>(sinkModule)) {
      auto diag = mlir::emitError(source.getLoc())
                  << "This source is involved with a Wiring Problem which "
                     "includes an External Module port and External Module "
                     "ports anre not supported.";
      diag.attachNote(sink.getLoc()) << "The sink is here.";
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  - index: " << index << "\n"
                   << "    source:\n"
                   << "      module: " << sourceModule.moduleName() << "\n"
                   << "      value: " << source << "\n"
                   << "    sink:\n"
                   << "      module: " << sinkModule.moduleName() << "\n"
                   << "      value: " << sink << "\n"
                   << "    newNameHint: " << problem.newNameHint << "\n";
    });

    // If the source and sink are in the same block, just wire them up.
    if (sink.getParentBlock() == source.getParentBlock()) {
      auto builder = ImplicitLocOpBuilder::atBlockEnd(UnknownLoc::get(context),
                                                      sink.getParentBlock());
      if (failed(connect(source, sink, builder)))
        return failure();
      continue;
    }
    // If both are in the same module but not same block, U-turn.
    // We may not be able to handle this, but that is checked below while
    // connecting.
    if (sourceModule == sinkModule) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    LCA: " << sourceModule.moduleName() << "\n");
      moduleModifications[sourceModule].connectionMap[index] = source;
      moduleModifications[sourceModule].uturns.push_back({index, sink});
      continue;
    }

    // Otherwise, get instance paths for source/sink, and compute LCA.
    auto sourcePaths = state.instancePathCache.getAbsolutePaths(
        cast<hw::HWModuleLike>(*sourceModule));
    auto sinkPaths = state.instancePathCache.getAbsolutePaths(
        cast<hw::HWModuleLike>(*sinkModule));

    if (sourcePaths.size() != 1 || sinkPaths.size() != 1) {
      auto diag =
          mlir::emitError(source.getLoc())
          << "This source is involved with a Wiring Problem where the source "
             "or the sink are multiply instantiated and this is not supported.";
      diag.attachNote(sink.getLoc()) << "The sink is here.";
      return failure();
    }

    FModuleOp lca =
        cast<FModuleOp>(instanceGraph.getTopLevelNode()->getModule());
    auto sources = sourcePaths[0];
    auto sinks = sinkPaths[0];
    while (!sources.empty() && !sinks.empty()) {
      if (sources[0] != sinks[0])
        break;
      auto newLCA = sources[0];
      lca = cast<FModuleOp>(instanceGraph.getReferencedModule(newLCA));
      sources = sources.drop_front();
      sinks = sinks.drop_front();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "    LCA: " << lca.moduleName() << "\n"
                   << "    sourcePaths:\n";
      for (auto inst : sourcePaths[0])
        llvm::dbgs() << "      - " << inst.instanceName() << " of "
                     << inst.referencedModuleName() << "\n";
      llvm::dbgs() << "    sinkPaths:\n";
      for (auto inst : sinkPaths[0])
        llvm::dbgs() << "      - " << inst.instanceName() << " of "
                     << inst.referencedModuleName() << "\n";
    });

    // Pre-populate the connectionMap of the module with the source and sink.
    moduleModifications[sourceModule].connectionMap[index] = source;
    moduleModifications[sinkModule].connectionMap[index] = sink;

    // Record ports that should be added to each module along the LCA path.
    RefType refType = TypeSwitch<Type, RefType>(problem.source.getType())
                          .Case<FIRRTLBaseType>([](FIRRTLBaseType base) {
                            return RefType::get(base);
                          })
                          .Case<RefType>([](RefType ref) { return ref; });

    // Record module modifications related to adding ports to modules.
    auto addPorts = [&](ArrayRef<hw::HWInstanceLike> insts, Value val, Type tpe,
                        Direction dir) {
      StringRef name, instName;
      for (auto inst : llvm::reverse(insts)) {
        auto mod = cast<FModuleOp>(instanceGraph.getReferencedModule(inst));
        if (name.empty()) {
          if (problem.newNameHint.empty())
            name = state.getNamespace(mod).newName(
                getFieldName(getFieldRefFromValue(val), /*nameSafe=*/true)
                    .first +
                "__bore");
          else
            name = state.getNamespace(mod).newName(problem.newNameHint);
        } else {
          assert(!instName.empty());
          name = state.getNamespace(mod).newName(instName + "_" + name);
        }
        moduleModifications[mod].portsToAdd.push_back(
            {index, {StringAttr::get(context, name), tpe, dir}});
        instName = inst.instanceName();
      }
    };

    // Record the addition of ports.
    if (noRefTypePorts)
      addPorts(sources, source, refType.getType(), Direction::Out);
    else
      addPorts(sources, source, refType, Direction::Out);
    addPorts(sinks, sink, refType.getType(), Direction::In);
  }

  // Iterate over modules from leaves to roots, applying ModuleModifications to
  // each module.
  LLVM_DEBUG({ llvm::dbgs() << "Updating modules:\n"; });
  for (auto *op : llvm::post_order(instanceGraph.getTopLevelNode())) {
    auto fmodule = dyn_cast<FModuleOp>(*op->getModule());
    // Skip external modules and modules that have no modifications.
    if (!fmodule || !moduleModifications.count(fmodule))
      continue;

    auto modifications = moduleModifications[fmodule];
    LLVM_DEBUG({
      llvm::dbgs() << "  - module: " << fmodule.moduleName() << "\n";
      llvm::dbgs() << "    ports:\n";
      for (auto [index, port] : modifications.portsToAdd) {
        llvm::dbgs() << "      - name: " << port.getName() << "\n"
                     << "        id: " << index << "\n"
                     << "        type: " << port.type << "\n"
                     << "        direction: "
                     << (port.direction == Direction::In ? "in" : "out")
                     << "\n";
      }
    });

    // Add ports to the module after all other existing ports.
    SmallVector<std::pair<unsigned, PortInfo>> newPorts;
    SmallVector<unsigned> problemIndices;
    for (auto [problemIdx, portInfo] : modifications.portsToAdd) {
      // Create the port.
      newPorts.push_back({fmodule.getNumPorts(), portInfo});
      problemIndices.push_back(problemIdx);
    }
    auto originalNumPorts = fmodule.getNumPorts();
    auto portIdx = fmodule.getNumPorts();
    fmodule.insertPorts(newPorts);

    auto builder = ImplicitLocOpBuilder::atBlockBegin(UnknownLoc::get(context),
                                                      fmodule.getBodyBlock());

    // Connect each port to the value stored in the connectionMap for this
    // wiring problem index.
    for (auto [problemIdx, portPair] : llvm::zip(problemIndices, newPorts)) {
      Value src = moduleModifications[fmodule].connectionMap[problemIdx];
      assert(src && "there did not exist a driver for the port");
      Value dest = fmodule.getArgument(portIdx++);
      if (failed(connect(src, dest, builder)))
        return failure();
    }

    // If a U-turn exists, this is an LCA and we need a U-turn connection. These
    // are the last connections made for this module.
    for (auto [problemIdx, dest] : moduleModifications[fmodule].uturns) {
      Value src = moduleModifications[fmodule].connectionMap[problemIdx];
      assert(src && "there did not exist a connection for the u-turn");
      if (failed(connect(src, dest, builder)))
        return failure();
    }

    // Update the connectionMap of all modules for which we created a port.
    for (auto *inst : instanceGraph.lookup(fmodule)->uses()) {
      InstanceOp useInst = cast<InstanceOp>(inst->getInstance());
      auto enclosingModule = useInst->getParentOfType<FModuleOp>();
      auto clonedInst = useInst.cloneAndInsertPorts(newPorts);
      state.instancePathCache.replaceInstance(useInst, clonedInst);
      // When RAUW-ing, ignore the new ports that we added when replacing (they
      // cannot have uses).
      useInst->replaceAllUsesWith(
          clonedInst.getResults().drop_back(newPorts.size()));
      useInst->erase();
      // Record information in the moduleModifications strucutre for the module
      // _where this is instantiated_.  This is done so that when that module is
      // visited later, there will be information available for it to find ports
      // it needs to wire up.  If there is already an existing connection, then
      // this is a U-turn.
      for (auto [newPortIdx, problemIdx] : llvm::enumerate(problemIndices)) {
        auto &modifications = moduleModifications[enclosingModule];
        auto newPort = clonedInst.getResult(newPortIdx + originalNumPorts);
        if (modifications.connectionMap.count(problemIdx)) {
          modifications.uturns.push_back({problemIdx, newPort});
          continue;
        }
        modifications.connectionMap[problemIdx] = newPort;
      }
    }
  }

  // Delete unused WireOps created by producers of WiringProblems.
  for (auto *op : opsToErase)
    op->erase();

  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerAnnotationsPass::runOnOperation() {
  CircuitOp circuit = getOperation();
  SymbolTable modules(circuit);

  LLVM_DEBUG(llvm::dbgs() << "===- Running LowerAnnotations Pass "
                             "------------------------------------------===\n");

  // Grab the annotations from a non-standard attribute called "rawAnnotations".
  // This is a temporary location for all annotations that are earmarked for
  // processing by this pass as we migrate annotations from being handled by
  // FIRAnnotations/FIRParser into this pass.  While we do this, this pass is
  // not supposed to touch _other_ annotations to enable this pass to be run
  // after FIRAnnotations/FIRParser.
  auto annotations = circuit->getAttrOfType<ArrayAttr>(rawAnnotations);
  if (!annotations)
    return;
  circuit->removeAttr(rawAnnotations);

  // Populate the worklist in reverse order.  This has the effect of causing
  // annotations to be processed in the order in which they appear in the
  // original JSON.
  for (auto anno : llvm::reverse(annotations.getValue()))
    worklistAttrs.push_back(anno.cast<DictionaryAttr>());

  size_t numFailures = 0;
  size_t numAdded = 0;
  auto addToWorklist = [&](DictionaryAttr anno) {
    ++numAdded;
    worklistAttrs.push_back(anno);
  };
  InstancePathCache instancePathCache(getAnalysis<InstanceGraph>());
  ApplyState state{circuit, modules, addToWorklist, instancePathCache};
  LLVM_DEBUG(llvm::dbgs() << "Processing annotations:\n");
  while (!worklistAttrs.empty()) {
    auto attr = worklistAttrs.pop_back_val();
    if (applyAnnotation(attr, state).failed())
      ++numFailures;
  }

  if (failed(solveWiringProblems(state)))
    ++numFailures;

  // Update statistics
  numRawAnnotations += annotations.size();
  numAddedAnnos += numAdded;
  numAnnos += numAdded + annotations.size();
  numReusedHierPathOps += state.numReusedHierPaths;

  if (numFailures)
    signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerFIRRTLAnnotationsPass(bool ignoreUnhandledAnnotations,
                                                bool ignoreClasslessAnnotations,
                                                bool noRefTypePorts) {
  auto pass = std::make_unique<LowerAnnotationsPass>();
  pass->ignoreUnhandledAnno = ignoreUnhandledAnnotations;
  pass->ignoreClasslessAnno = ignoreClasslessAnnotations;
  pass->noRefTypePorts = noRefTypePorts;
  return pass;
}
