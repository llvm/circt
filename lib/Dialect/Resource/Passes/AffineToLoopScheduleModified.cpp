//===- AffineToLoopSchedule.cpp--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MODIFIED VERSION. Changes relative to the original conversion:
//
//   1. affine.apply support: AffineApplyOp is now marked illegal in
//      lowerAffineStructures(), so the AffineApplyLowering pattern from
//      AffineToStandard (already pulled in via
//      populateAffineToStdConversionPatterns) expands it into arith ops
//      before scheduling. The scheduler therefore never sees affine.apply.
//
//   2. Expanded arithmetic coverage in populateOperatorTypes():
//      - MulIOp / DivSIOp / DivUIOp / RemSIOp / RemUIOp -> "multicycle".
//        (div/rem appear when affine.apply maps containing floordiv/mod are
//        expanded, e.g. the delinearization introduced by loop coalescing.)
//      - SelectOp, SubIOp, AndI/OrI/XOrI, shifts, trunc/ext, etc. -> "comb".
//        These also appear in floordiv/mod expansion (cmpi + select fixups).
//      - The original dead `.Case<AddIOp, CmpIOp>` (unreachable because both
//        ops matched the first Case) has been removed.
//
//   3. Nested loop support, in two layers:
//      a) Perfectly nested bands are coalesced (flattened) into a single
//         loop with affine::coalesceLoops() before any analysis runs. The
//         delinearization it introduces (floordiv/mod affine.apply ops) is
//         handled by (1) and (2).
//      b) Any remaining nesting (imperfect nests, or bands whose inner
//         bounds depend on outer IVs so coalescing fails) falls back to
//         pipelining each *innermost* loop in place. Outer affine.for loops
//         are left intact as sequential control around the pipeline; the
//         pipeline op is not IsolatedFromAbove, so it may freely reference
//         outer induction variables and loop-invariant values.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Resource/Passes/AffineToLoopScheduleModified.h"
#include "circt/Dialect/Resource/Passes/MemrefBankClassification.h"
#include "circt/Dialect/Resource/Passes/CyclicSchedulingAnalysisModified.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/Resource/HLS/HLSDialect.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Dialect/Resource/HLS/HLSOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"   // mlir::createCSEPass
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <limits>
#include <mlir/IR/IntegerSet.h>

#define DEBUG_TYPE "affine-to-loopschedule"
// https://circt.llvm.org/docs/Scheduling/
namespace circt {
#define GEN_PASS_DEF_AFFINETOLOOPSCHEDULE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace mlir::affine;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::loopschedule;
using namespace circt::hls_analysis;

namespace {


struct AffineToLoopSchedule
    : public circt::impl::AffineToLoopScheduleBase<AffineToLoopSchedule> {
  void unrollInnerReductions();
  void runOnOperation() override;

private:
  ModuloProblem getModuloProblem(CyclicProblem &prob);
  LogicalResult
  lowerAffineStructures(MemoryDependenceAnalysis &dependenceAnalysis);

  // MODIFIED: these now operate on a single (innermost) loop instead of a
  // perfectly nested band, since nests are either coalesced into one loop
  // up front, or only their innermost loops are pipelined in place.
  LogicalResult populateOperatorTypes(AffineForOp forOp, ModuloProblem &problem,
                                      const BRAMClassification &bramClass);
  
  LogicalResult solveSchedulingProblem(AffineForOp forOp,
                                       ModuloProblem &problem);
  LogicalResult createLoopSchedulePipeline(AffineForOp forOp,             
                                           ModuloProblem &problem);

  // MODIFIED: pre-analysis structural transform. Flattens perfectly nested
  // bands into single loops where legal.
  void coalescePerfectNests();

  CyclicSchedulingAnalysis *schedulingAnalysis;
};

} // namespace

/// Returns true if `forOp` contains no nested affine.for anywhere in its
/// body (including inside ifs).
static bool isInnermostAffineFor(AffineForOp forOp) {
  bool hasNested = false;
  forOp.getBody()->walk([&](AffineForOp) {
    hasNested = true;
    return WalkResult::interrupt();
  });
  return !hasNested;
}

/// MODIFIED: Returns true if the perfect band has a rectangular iteration
/// space, i.e. every non-outermost loop's bounds are invariant with respect
/// to all enclosing IVs in the band. Flattening via coalesceLoops is only
/// valid in that case (trip count = product of per-loop trip counts);
/// affine::coalesceLoops does NOT reject non-rectangular nests gracefully -
/// it hoists the upper-bound product above the outermost loop, where an
/// outer-IV-dependent bound operand is undefined, and corrupts the IR
/// (null operands) instead of returning failure. So we must check here.
static bool isRectangularBand(ArrayRef<AffineForOp> band) {
  // Op handles are value-like wrappers around Operation*, but their
  // generated accessors are not const-qualified, so copy out of the
  // ArrayRef before calling getRegion().
  AffineForOp outermost = band.front();
  Region &outerRegion = outermost.getRegion();
  for (AffineForOp loop : band.drop_front()) {
    if (!areValuesDefinedAbove(loop.getLowerBoundOperands(), outerRegion) ||
        !areValuesDefinedAbove(loop.getUpperBoundOperands(), outerRegion))
      return false;
  }
  return true;
}

ModuloProblem AffineToLoopSchedule::getModuloProblem(CyclicProblem &prob) {
  ModuloProblem modProb(prob.getContainingOp());
  for (auto *op : prob.getOperations()) {
    auto opr = prob.getLinkedOperatorType(op);
    if (opr.has_value()) {
      modProb.setLinkedOperatorType(op, opr.value());
      auto latency = prob.getLatency(opr.value());
      if (latency.has_value())
        modProb.setLatency(opr.value(), latency.value());
    }
    auto rsrc = prob.getLinkedResourceTypes(op);
    if (rsrc.has_value())
      modProb.setLinkedResourceTypes(op, rsrc.value());
    modProb.insertOperation(op);
  }

  for (auto *op : prob.getOperations()) {
    for (auto dep : prob.getDependences(op)) {
      if (dep.isAuxiliary()) {
        auto depInserted = modProb.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
      }
      auto distance = prob.getDistance(dep);
      if (distance.has_value())
        modProb.setDistance(dep, distance.value());
    }
  }

  return modProb;
}

/// MODIFIED: Flatten perfectly nested loop bands into single loops using
/// affine::coalesceLoops. This runs *before* dependence analysis so all
/// later analyses see the flattened IR.
///
/// coalesceLoops introduces delinearization of the flat IV via affine.apply
/// ops containing floordiv/mod; those are expanded to arith ops in
/// lowerAffineStructures and scheduled as multicycle/comb ops.
///
/// Bands that cannot be coalesced (e.g. inner bounds depending on outer
/// IVs) are simply left alone; their innermost loops are pipelined in place
/// later, with the outer loops remaining as sequential affine.for control.
void AffineToLoopSchedule::coalescePerfectNests() {
  // Collect maximal perfect bands first (pre-order so outer bands are found
  // before any inner siblings), then mutate after the walk completes.
  SmallVector<SmallVector<AffineForOp>> bands;
  DenseSet<Operation *> visited;
  getOperation().walk<WalkOrder::PreOrder>([&](AffineForOp root) {
    if (visited.contains(root))
      return;
    SmallVector<AffineForOp> band;
    getPerfectlyNestedLoops(band, root);
    for (AffineForOp loop : band)
      visited.insert(loop);
    // Only worth coalescing if the band actually ends in an innermost loop;
    // otherwise the flattened loop still would not be pipelineable.
    // MODIFIED: additionally require a rectangular iteration space, since
    // coalesceLoops silently miscompiles triangular/trapezoidal nests
    // instead of failing (see isRectangularBand). Non-rectangular nests
    // fall back to innermost-loop pipelining.
    if (band.size() > 1 && isInnermostAffineFor(band.back()) &&
        isRectangularBand(band))
      bands.push_back(std::move(band));
  });

  for (auto &band : bands) {
    if (failed(coalesceLoops(MutableArrayRef<AffineForOp>(band)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Could not coalesce perfect nest rooted at "
                 << *band.front()
                 << "; falling back to innermost-loop pipelining\n");
    }
  }
}

// Anchor memref-level write-effecting ops (e.g. hls.store_enable) that
// MemoryDependenceAnalysis can't see. Without an outgoing edge the scheduler
// treats them as sinks; give each an auxiliary dependence to the loop
// terminator so the dependence DAG has a single sink.
static void anchorUntrackedSideEffects(AffineForOp forOp,
                                       ModuloProblem &problem) {
  Operation *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](Operation *op) {
    if (!isa<hls_analysis::StoreEnableOp>(op))
      return;
    // Aux (control) dependence: op must be scheduled before the terminator.
    Problem::Dependence dep(op, anchor);
    auto inserted = problem.insertDependence(dep);
    (void)inserted;
    assert(succeeded(inserted) && "failed to anchor store_enable");
  });
}

/// MODIFIED: Fully unroll constant-trip innermost reduction loops so the
/// loop *above* them becomes the pipelineable innermost loop, matching
/// Vitis HLS behavior ("pipeline 2nd-level loop, unroll lower loop"). The
/// unrolled inner body exposes one memref.load per (bank, iteration) so the
/// partitioned arrays are actually read in parallel; otherwise the inner
/// reduction would serialize and the array_partition would be wasted.
///
/// Targets only innermost affine.for loops with a small constant trip count
/// that carry a reduction (iter_args)
void AffineToLoopSchedule::unrollInnerReductions() {
  SmallVector<AffineForOp> targets;
  // TODO: automatically unroll loop count with trips < 64
  //       paramterize later (Vitis specific behavior)
  
  unsigned kMaxUnrollTrip = 64;
  
  getOperation().walk([&](AffineForOp forOp) {
    if (!isInnermostAffineFor(forOp))
      return;
    // Reduction: carries iter_args (the accumulator).
    if (forOp.getNumIterOperands() == 0)
      return;
    auto tc = getConstantTripCount(forOp);
    if (!tc || *tc == 0)
      return;
    // Guard against unrolling something huge; the HLS inner dim is small.
    if (*tc > kMaxUnrollTrip)   // e.g. 64
      return;
    targets.push_back(forOp);
  });

  for (AffineForOp forOp : targets) {
    if (failed(loopUnrollFull(forOp))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "could not fully unroll inner reduction: " << *forOp << "\n");
    }
  }
}


void AffineToLoopSchedule::runOnOperation() {
  // Phase 0 — structural transforms. Both mutate the loop nest and MUST run
  // before any analysis is constructed.
  coalescePerfectNests();      // flatten perfect rectangular bands
  // unrollInnerReductions();     // unroll innermost reduction loops; the loop
  // above becomes the new innermost/pipelined loop
  
  // Get dependence analysis for the whole function.
  auto dependenceAnalysis = getAnalysis<MemoryDependenceAnalysis>();

  // After dependence analysis, materialize affine structures.
  if (failed(lowerAffineStructures(dependenceAnalysis)))
    return signalPassFailure();


  // MODIFIED: collapse the redundant index arithmetic introduced by
  // per-access affine-map expansion. Each affine.load/store expands its own
  // copy of the floordiv/mod delinearization, so a banked access used N
  // times yields N identical divsi/select chains. Running CSE here — before
  // the scheduling problem is constructed — means the scheduler sees one
  // shared computation, so its result occupies a single pipeline register
  // slot instead of one per access (otherwise the redundancy is frozen into
  // the loopschedule.register tuple, where later CSE can merge the def but
  // not the duplicated register operands).
  {
    OpPassManager cleanup(getOperation()->getName());
    cleanup.addPass(createCSEPass());
    if (failed(runPipeline(cleanup, getOperation())))
      return signalPassFailure();
  }

  // Get scheduling analysis for the whole function.
  schedulingAnalysis = &getAnalysis<CyclicSchedulingAnalysis>();

  auto &bramClass = getAnalysis<BRAMClassification>();
  
  // MODIFIED: Instead of bailing on nests, collect every *innermost* loop
  // (post-coalescing) and pipeline each one in place. Outer loops of
  // non-coalescable nests remain as affine.for wrapping the pipeline.
  SmallVector<AffineForOp> innermostLoops;
  getOperation().walk([&](AffineForOp forOp) {
    if (isInnermostAffineFor(forOp))
      innermostLoops.push_back(forOp);
  });

  for (AffineForOp loop : innermostLoops) {
    ModuloProblem moduloProblem =
        getModuloProblem(schedulingAnalysis->getProblem(loop));

    // Populate the target operator types.
    if (failed(populateOperatorTypes(loop, moduloProblem, bramClass)))
      return signalPassFailure();
    
    anchorUntrackedSideEffects(loop, moduloProblem);

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveSchedulingProblem(loop, moduloProblem)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createLoopSchedulePipeline(loop, moduloProblem)))
      return signalPassFailure();
  }
}

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
/// Also replaces the affine load with the memref load in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineLoadLowering : public OpConversionPattern<AffineLoadOp> {
public:
  AffineLoadLowering(MLIRContext *context,
                     MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build memref.load memref[expandedMap.results].
    auto memrefLoad = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, op.getMemRef(), *resultOperands);

    dependenceAnalysis.replaceOp(op, memrefLoad);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which
/// replaces the original 'affine.store').
/// Also replaces the affine store with the memref store in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineStoreLowering : public OpConversionPattern<AffineStoreOp> {
public:
  AffineStoreLowering(MLIRContext *context,
                      MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    auto memrefStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);

    dependenceAnalysis.replaceOp(op, memrefStore);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

// Lower an IntegerSet predicate to one i1: AND of one comparison per
// constraint. Equality (expr == 0) -> cmpi eq; inequality (expr >= 0) -> sge.
// Empty set is vacuously true.
static Value materializeIntegerSet(OpBuilder &b, Location loc, IntegerSet set,
                                   ValueRange setOperands) {
  if (set.getNumConstraints() == 0)
    return arith::ConstantIntOp::create(b, loc, /*value=*/1, /*width=*/1);

  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value acc;
  for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
    AffineExpr c = set.getConstraint(i);
    bool isEq = set.isEq(i);
    auto cMap = AffineMap::get(set.getNumDims(), set.getNumSymbols(), c);
    Value lhs = affine::AffineApplyOp::create(b, loc, cMap,
                                              llvm::to_vector(setOperands));
    Value cmp = arith::CmpIOp::create(
        b, loc, isEq ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge,
        lhs, zero);
    acc = acc ? arith::AndIOp::create(b, loc, acc, cmp).getResult() : cmp;
  }
  return acc;
}

struct AffineStoreEnableLowering
    : public OpConversionPattern<hls_analysis::AffineStoreEnableOp> {
  AffineStoreEnableLowering(MLIRContext *ctx,
                            MemoryDependenceAnalysis &da)
      : OpConversionPattern(ctx), dependenceAnalysis(da) {}

  LogicalResult
  matchAndRewrite(AffineStoreEnableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // (a) predicate set -> i1
    Value en = materializeIntegerSet(rewriter, loc, op.getCondition(),
                                     adaptor.getSetOperands());
    if (!en)
      return rewriter.notifyMatchFailure(op, "could not materialize predicate");

    // (b) affine map -> index SSA values
    auto idx = expandAffineMap(rewriter, loc, op.getMap(),
                               llvm::to_vector(adaptor.getMapOperands()));
    if (!idx)
      return rewriter.notifyMatchFailure(op, "could not expand access map");

    // (c) flat predicated store
    auto lowered = rewriter.replaceOpWithNewOp<hls_analysis::StoreEnableOp>(
        op, adaptor.getValue(), en, adaptor.getMemref(), *idx);

    // (d) hand the scheduler the dependence edges the affine op carried
    dependenceAnalysis.replaceOp(op, lowered);
    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Helper to hoist computation out of scf::IfOp branches, turning it into a
/// mux-like operation, and exposing potentially concurrent execution of its
/// branches.
struct IfOpHoisting : OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        rewriter.inlineBlockBefore(&op.getThenRegion().front(), op);
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        rewriter.inlineBlockBefore(&op.getElseRegion().front(), op);
      }
    });

    return success();
  }
};

/// Helper to determine if an scf::IfOp is in mux-like form.
static bool ifOpLegalityCallback(IfOp op) {
  return op.thenBlock()->without_terminator().empty() &&
         (!op.elseBlock() || op.elseBlock()->without_terminator().empty());
}

/// Helper to mark AffineYieldOp legal, unless it is inside a partially
/// converted scf::IfOp.
static bool yieldOpLegalityCallback(AffineYieldOp op) {
  return !op->getParentOfType<IfOp>();
}

/// After analyzing memory dependences, and before creating the schedule, we
/// want to materialize affine operations with arithmetic, scf, and memref
/// operations, which make the condition computation of addresses, etc.
/// explicit. This is important so the schedule can consider potentially
/// complex computations in the condition of ifs, or the addresses of loads
/// and stores. The dependence analysis will be updated so the dependences
/// from the affine loads and stores are now on the memref loads and stores.
LogicalResult AffineToLoopSchedule::lowerAffineStructures(
    MemoryDependenceAnalysis &dependenceAnalysis) {
  auto *context = &getContext();
  auto op = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, ArithDialect, MemRefDialect,
                         SCFDialect, HLSDialect>();
  // MODIFIED: AffineApplyOp is now illegal too. The AffineApplyLowering
  // pattern registered by populateAffineToStdConversionPatterns expands it
  // into explicit arith ops (muli/addi/divsi/remsi/cmpi/select for
  // floordiv/mod semantics), which the scheduler can then handle. 
  target.addIllegalOp<AffineApplyOp, AffineStoreEnableOp,
                      AffineIfOp, AffineLoadOp, AffineStoreOp>();
  target.addDynamicallyLegalOp<IfOp>(ifOpLegalityCallback);
  target.addDynamicallyLegalOp<AffineYieldOp>(yieldOpLegalityCallback);

  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);
  patterns.add<AffineLoadLowering>(context, dependenceAnalysis);
  patterns.add<AffineStoreLowering>(context, dependenceAnalysis);
  patterns.add<AffineStoreEnableLowering>(context, dependenceAnalysis);
  patterns.add<IfOpHoisting>(context);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  return success();
}


//===----------------------------------------------------------------------===//
// Operator type (latency-1 "mem_<hash>") is unchanged and shared per memref —
// it governs latency, not port count. Only the *resource* construction varies
// by mode:
//   1P  -> one pool, limit 1
//   TDP -> one pool, limit 2
//   SDP -> two pools, limit 1 each: loads -> _rd, stores -> _wr (dedicated)
//
//===----------------------------------------------------------------------===//
 
static void linkMemoryResource(Operation *memOp, Value memRef, bool isLoad,
                               ModuloProblem &problem,
                               const BRAMClassification &bramClass) {
  auto key = std::to_string(hash_value(memRef));
  // Operator type: latency 1, shared per memref (unchanged).
  Problem::OperatorType memOpr =
      problem.getOrInsertOperatorType("mem_" + key);
  problem.setLatency(memOpr, 1);
  problem.setLinkedOperatorType(memOp, memOpr);
 
  using hls_analysis::StorageMode;
  const BankProfile *prof = bramClass.getProfile(memRef);
 
  SmallVector<Problem::ResourceType> rsrcs;
  // Fallback: untracked memref -> preserve original single-pool limit-2.
  StorageMode mode = prof ? prof->mode : StorageMode::TDP;
  Problem::ResourceType r;
  
  switch (mode) {
  case StorageMode::OnePort: {
    // 1 r/w port single
    r = problem.getOrInsertResourceType("mem_" + key + "_rsrc");
    problem.setLimit(r, 1);
    rsrcs.push_back(r);
    break;
  }
  case StorageMode::TDP: {
    // 2 r/w port single
    r = problem.getOrInsertResourceType("mem_" + key + "_rsrc");
    problem.setLimit(r, 2);
    rsrcs.push_back(r);
    break;
  }
  case StorageMode::SDP: {
    // Dedicated read and write ports: split pools, each limit 1. A load only
    // consumes the read port; a store only the write port. A shared pool of 2
    // would wrongly let two reads issue in one cycle, which SDP cannot do.
    if (isLoad) {
      r = problem.getOrInsertResourceType("mem_" + key + "_rd_rsrc");
      problem.setLimit(r, 1);
    } else {
      r = problem.getOrInsertResourceType("mem_" + key + "_wr_rsrc");
      problem.setLimit(r, 1);
    }
    rsrcs.push_back(r);
    break;
  }
  }
  problem.setLinkedResourceTypes(memOp, rsrcs);
 
  LLVM_DEBUG(llvm::dbgs()
             << "linked " << (isLoad ? "load" : "store") << " to "
             << stringifyStorageMode(mode) << " resource for "
             << *memOp << "\n");
}

/// Populate the scheduling problem operator types for the dialect we are
/// targeting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
///
/// MODIFIED: now takes a single innermost loop, and covers the full set of
/// arith ops produced by affine map expansion (including the floordiv/mod
/// delinearization introduced by loop coalescing): mul/div/rem are
/// multicycle, while select/sub/logical/shift/ext/trunc are combinational.
LogicalResult AffineToLoopSchedule::populateOperatorTypes(
    AffineForOp forOp, ModuloProblem &problem,
    const BRAMClassification &bramClass) {
  
  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);
 
  Problem::OperatorType fpMulOpr = problem.getOrInsertOperatorType("fpmul");
  problem.setLatency(fpMulOpr, 4);
  Problem::OperatorType fpAddOpr = problem.getOrInsertOperatorType("fpaddsub");
  problem.setLatency(fpAddOpr, 5);
  Problem::OperatorType fpDivOpr = problem.getOrInsertOperatorType("fpdiv");
  problem.setLatency(fpDivOpr, 12);
 
  Operation *unsupported;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<IfOp, AffineYieldOp, arith::ConstantOp, CmpIOp, IndexCastOp,
              AllocaOp, AllocOp, YieldOp, AddIOp, SubIOp,
              SelectOp, AndIOp, OrIOp, XOrIOp, ShLIOp, ShRSIOp, ShRUIOp,
              TruncIOp, ExtSIOp, ExtUIOp, NegFOp, CmpFOp, ExtFOp, TruncFOp,
              SIToFPOp, UIToFPOp, FPToSIOp, FPToUIOp, BitcastOp>(
            [&](Operation *combOp) {
              problem.setLinkedOperatorType(combOp, combOpr);
              return WalkResult::advance();
            })
        // CHANGED: store now goes through the mode-keyed helper.
        .Case<AffineStoreOp, memref::StoreOp>([&](Operation *memOp) {
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<memref::StoreOp>(*memOp).getMemRef();
          linkMemoryResource(memOp, memRef, /*isLoad=*/false, problem,
                             bramClass);
          return WalkResult::advance();
        })
        // CHANGED: load now goes through the mode-keyed helper.
        .Case<AffineLoadOp, memref::LoadOp>([&](Operation *memOp) {
          Value memRef = isa<AffineLoadOp>(*memOp)
                             ? cast<AffineLoadOp>(*memOp).getMemRef()
                             : cast<memref::LoadOp>(*memOp).getMemRef();
          linkMemoryResource(memOp, memRef, /*isLoad=*/true, problem,
                             bramClass);
          return WalkResult::advance();
        })
        .Case<StoreEnableOp>([&](Operation *memOp) {
          Value memRef = cast<StoreEnableOp>(*memOp).getMemref();
          linkMemoryResource(memOp, memRef, /*isLoad=*/false, problem,
                             bramClass);
          return WalkResult::advance();
        })
        .Case<MulIOp, DivSIOp, DivUIOp, RemSIOp, RemUIOp>(
            [&](Operation *mcOp) {
              problem.setLinkedOperatorType(mcOp, mcOpr);
              return WalkResult::advance();
            })
        .Case<MulFOp>([&](Operation *fpOp) {
          problem.setLinkedOperatorType(fpOp, fpMulOpr);
          return WalkResult::advance();
        })
        .Case<AddFOp, SubFOp, MaximumFOp, MinimumFOp, MaxNumFOp, MinNumFOp>(
            [&](Operation *fpOp) {
              problem.setLinkedOperatorType(fpOp, fpAddOpr);
              return WalkResult::advance();
            })
        .Case<DivFOp, RemFOp>([&](Operation *fpOp) {
          problem.setLinkedOperatorType(fpOp, fpDivOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });
 
  if (result.wasInterrupted())
    return forOp.emitError("unsupported operation ") << *unsupported;
 
  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult
AffineToLoopSchedule::solveSchedulingProblem(AffineForOp forOp,
                                             ModuloProblem &problem) {
  // Optionally debug problem inputs.
  LLVM_DEBUG(forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::dbgs() << "Scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr->getAttr();
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = forOp.getBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });

  return success();
}

/// Create the loopschedule pipeline op for a single innermost loop.
///
/// MODIFIED: the pipeline is created *in place of* `forOp` rather than at
/// the root of a loop nest, and only `forOp` itself is erased. Any
/// enclosing affine.for loops survive and act as sequential control around
/// the pipeline. Because LoopSchedulePipelineOp is not IsolatedFromAbove,
/// cloned ops may keep references to outer induction variables and other
/// dominating values without extra plumbing.
LogicalResult
AffineToLoopSchedule::createLoopSchedulePipeline(AffineForOp forOp,
                                                 ModuloProblem &problem) {
  ImplicitLocOpBuilder builder(forOp.getLoc(), forOp);

  // Create Values for the loop's lower and upper bounds. If the bounds
  // depend on outer induction variables (non-coalesced nests), the expanded
  // bound computation is inserted right before the pipeline, inside the
  // enclosing loop, so dominance is preserved.
  Value lowerBound = lowerAffineLowerBound(forOp, builder);
  Value upperBound = lowerAffineUpperBound(forOp, builder);
  int64_t stepValue = forOp.getStep().getSExtValue();
  auto step = arith::ConstantOp::create(
      builder, IntegerAttr::get(builder.getIndexType(), stepValue));

  // Create the pipeline op, with the same result types as the loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = forOp.getResultTypes();

  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  SmallVector<Value> iterArgs;
  iterArgs.push_back(lowerBound);
  iterArgs.append(forOp.getInits().begin(), forOp.getInits().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an
  // AffineMap. For loops whose bounds depend on outer IVs this is simply
  // omitted.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount = getConstantTripCount(forOp))
    tripCountAttr = builder.getI64IntegerAttr(*tripCount);

  auto pipeline = LoopSchedulePipelineOp::create(builder, resultTypes, ii,
                                                 tripCountAttr, iterArgs);

  // Create the condition, which currently just compares the induction
  // variable to the upper bound.
  Block &condBlock = pipeline.getCondBlock();
  builder.setInsertionPointToStart(&condBlock);
  auto cmpResult = arith::CmpIOp::create(builder, builder.getI1Type(),
                                         arith::CmpIPredicate::ult,
                                         condBlock.getArgument(0), upperBound);
  condBlock.getTerminator()->insertOperands(0, {cmpResult});

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  // MODIFIED: ensure a stage exists for *every* time slot up to the last
  // scheduled operation, even if no op starts there. With multi-cycle
  // operators (e.g. latency-3 remsi/divsi from floordiv/mod expansion, or
  // latency-4/5 FP ops), the schedule can have gaps: a value produced at
  // time t and consumed at time t+k must be registered through every
  // intermediate stage, and that pass-through chain breaks if intermediate
  // stages are missing. When a chain breaks, builder.clone() silently falls
  // back to the original (soon-to-be-erased) value, producing null operands
  // after loop erasure. Empty groups become pure register-forwarding stages;
  // any stage with nothing to execute and nothing to forward is skipped at
  // creation time below.
  unsigned maxStartTime = 0;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp>(op))
      continue;
    maxStartTime = std::max(maxStartTime, *problem.getStartTime(op));
  }
  for (unsigned t = 0; t <= maxStartTime; ++t)
    if (!startGroups.count(t))
      startGroups.try_emplace(t, SmallVector<Operation *>());

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args. Values defined *outside* the
  // loop (outer IVs, loop-invariant values) are intentionally not mapped:
  // clones keep referencing them directly.
  IRMapping valueMap;
  assert(iterArgs.size() == forOp.getBody()->getNumArguments());
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(forOp.getBody()->getArgument(i),
                 pipeline.getStagesBlock().getArgument(i));

  // Create the stages.
  Block &stagesBlock = pipeline.getStagesBlock();
  builder.setInsertionPointToStart(&stagesBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());

  // Keys for translating values in each stage
  SmallVector<SmallVector<Value>> registerValues;
  SmallVector<SmallVector<Type>> registerTypes;

  // The maps that ensure a stage uses the correct version of a value
  SmallVector<IRMapping> stageValueMaps;

  // For storing the range of stages an operation's results need to be valid
  // for
  DenseMap<Operation *, std::pair<unsigned, unsigned>> pipeTimes;

  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];

    // Collect the return types for this stage. Operations whose results are
    // not used within this stage are returned.
    auto isLoopTerminator = [forOp](Operation *op) {
      return isa<AffineYieldOp>(op) && op->getParentOp() == forOp;
    };

    // Initialize set of registers up until this point in time
    for (unsigned i = registerValues.size(); i <= startTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());

    // Check each operation to see if its results need plumbing
    for (auto *op : group) {
      if (op->getUsers().empty())
        continue;

      unsigned pipeEndTime = 0;
      for (auto *user : op->getUsers()) {
        unsigned userStartTime = *problem.getStartTime(user);
        if (*problem.getStartTime(user) > startTime)
          pipeEndTime = std::max(pipeEndTime, userStartTime);
        else if (isLoopTerminator(user))
          // Manually forward the value into the terminator's valueMap
          pipeEndTime = std::max(pipeEndTime, userStartTime + 1);
      }

      // Insert the range of pipeline stages the value needs to be valid for
      pipeTimes[op] = std::pair(startTime, pipeEndTime);

      // Add register stages for each time slice we need to pipe to
      for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
        registerValues.push_back(SmallVector<Value>());

      // Keep a collection of this stages results as keys to our valueMaps
      for (auto result : op->getResults())
        registerValues[startTime].push_back(result);

      // Other stages that use the value will need these values as keys too
      unsigned firstUse = std::max(
          startTime + 1,
          startTime + *problem.getLatency(*problem.getLinkedOperatorType(op)));
      for (unsigned i = firstUse; i < pipeEndTime; ++i) {
        for (auto result : op->getResults())
          registerValues[i].push_back(result);
      }
    }
  }

  // Now make register Types and stageValueMaps
  for (unsigned i = 0; i < registerValues.size(); ++i) {
    SmallVector<mlir::Type> types;
    for (auto val : registerValues[i])
      types.push_back(val.getType());

    registerTypes.push_back(types);
    stageValueMaps.push_back(valueMap);
  }

  // One more map is needed for the pipeline stages terminator
  stageValueMaps.push_back(valueMap);

  // Create stages along with maps
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    llvm::sort(group, [&](Operation *a, Operation *b) {
      return dom.properlyDominates(a, b);
    });
    auto stageTypes = registerTypes[startTime];

    // MODIFIED: a gap-filling stage that has no ops to run and no values to
    // forward is pointless; skip it. (Stage 0 is always created since it
    // carries the induction variable increment.)
    if (group.empty() && stageTypes.empty() && startTime != 0)
      continue;
    // Add the induction variable increment in the first stage.
    if (startTime == 0)
      stageTypes.push_back(lowerBound.getType());

    // Create the stage itself.
    builder.setInsertionPoint(stagesBlock.getTerminator());
    auto startTimeAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), startTime);
    auto stage =
        LoopSchedulePipelineStageOp::create(builder, stageTypes, startTimeAttr);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    for (auto *op : group) {
      auto *newOp = builder.clone(*op, stageValueMaps[startTime]);

      // All further uses in this stage should used the cloned-version of
      // values So we update the mapping in this stage
      for (auto result : op->getResults())
        stageValueMaps[startTime].map(
            result, newOp->getResult(result.getResultNumber()));
    }

    // Register all values in the terminator, using their mapped value
    SmallVector<Value> stageOperands;
    unsigned resIndex = 0;
    for (auto res : registerValues[startTime]) {
      stageOperands.push_back(stageValueMaps[startTime].lookup(res));
      // Additionally, update the map of the stage that will consume the
      // registered value
      unsigned destTime = startTime + 1;
      unsigned latency = *problem.getLatency(
          *problem.getLinkedOperatorType(res.getDefiningOp()));
      // Multi-cycle case
      if (*problem.getStartTime(res.getDefiningOp()) == startTime &&
          latency > 1)
        destTime = startTime + latency;
      destTime = std::min((unsigned)(stageValueMaps.size() - 1), destTime);
      stageValueMaps[destTime].map(res, stage.getResult(resIndex++));
    }
    // Add these mapped values to pipeline.register
    stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                    stageOperands);

    // Add the induction variable increment to the first stage.
    if (startTime == 0) {
      auto incResult =
          arith::AddIOp::create(builder, stagesBlock.getArgument(0), step);
      stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                      incResult->getResults());
    }
  }

  // Add the iter args and results to the terminator.
  auto stagesTerminator =
      cast<LoopScheduleTerminatorOp>(stagesBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and
  // any mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  termIterArgs.push_back(
      stagesBlock.front().getResult(stagesBlock.front().getNumResults() - 1));

  for (auto value : forOp.getBody()->getTerminator()->getOperands()) {
    unsigned lookupTime = std::min((unsigned)(stageValueMaps.size() - 1),
                                   pipeTimes[value.getDefiningOp()].second);

    termIterArgs.push_back(stageValueMaps[lookupTime].lookup(value));
    termResults.push_back(stageValueMaps[lookupTime].lookup(value));
  }

  stagesTerminator.getIterArgsMutable().append(termIterArgs);
  stagesTerminator.getResultsMutable().append(termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  // MODIFIED: remove only this loop from the IR; enclosing loops (if any)
  // are kept as sequential control around the new pipeline.
  forOp.walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}
namespace circt::hls_analysis {

std::unique_ptr<mlir::Pass> createAffineToLoopSchedule() {
  return std::make_unique<AffineToLoopSchedule>();
}

void registerAffineToLoopScheduleAnalysisPass() {
  PassRegistration<AffineToLoopSchedule>();
}

}
