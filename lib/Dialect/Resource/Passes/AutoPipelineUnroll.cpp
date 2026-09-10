//===- AutoPipelineUnroll.cpp - model Vitis auto-pipeline + implied unroll -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reproduces Vitis HLS's *automatic* pipeline decision on affine loop nests so
// downstream II / replication / BRAM estimation sees the same loop structure the
// tool schedules against.
//
// Rule, calibrated against gemm + the perfect copy-nests on a Zynq-7020:
//
//   THRESHOLD = config_compile -pipeline_loops (default 64).
//
//   For the innermost loop L of a perfect band (parent loop = L's enclosing
//   affine.for):
//
//     1. L has no parent loop          -> leave. Pipelined in place downstream,
//                                          regardless of trip count.
//     2. L carries a reduction:
//          TC(L) <= THRESHOLD          -> UNROLL L (and everything strictly
//                                          below the pipelined parent); the
//                                          recurrence becomes an add tree and
//                                          the parent becomes the pipelined loop.
//          TC(L) >  THRESHOLD          -> leave. L stays a pipelined loop; the
//                                          recurrence sets RecMII downstream
//                                          (matches N=65 gemm: II=6, no unroll).
//     3. L carries no reduction        -> leave. The perfect nest folds into the
//                                          flatten; nothing is unrolled.
//
//   "Pipelining a loop fully unrolls every loop strictly nested inside it"
//   (Vitis semantics) is modeled by unrollStrictlyInside() on the pipelined
//   parent in the reduction-unroll case.
//
// This pass performs ONLY the structural mutation (the implied unroll). It does
// not insert pipeline ops: in this flow, "pipeline L" = a downstream
// AffineToLoopSchedule pass turning the surviving innermost loop into a
// LoopSchedulePipelineOp. Run order:
//
//     auto-pipeline-unroll  ->  coalesce-perfect-nests  ->  affine-to-loopschedule
//
// Unrolling the reduction here turns lp1/lp2/lp3 into a dependency-free lp1/lp2
// band that the coalescer then flattens, matching csynth:
//   - lpsum_3 (tmp1[i][j] += prod[i][j][k]) : reduction, TC=64 -> UNROLL
//   - lp3     (prod[i][j][k] = ...)         : no reduction     -> LEAVE (flatten)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "auto-pipeline-unroll"

using namespace mlir;
using namespace mlir::affine;

namespace {

//===----------------------------------------------------------------------===//
// Reduction detection (clause 2)
//===----------------------------------------------------------------------===//

/// Does `from` flow (transitively, within `forOp`'s body) into `into`?
/// Used to confirm the loaded value actually feeds the stored value -- i.e. a
/// genuine read-modify-write, not an unrelated load and store that happen to
/// touch the same address.
static bool valueFeeds(Value from, Value into, AffineForOp forOp) {
  if (from == into)
    return true;
  llvm::SmallVector<Value, 8> worklist{into};
  llvm::SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (v == from)
      return true;
    Operation *def = v.getDefiningOp();
    if (!def || !forOp->isProperAncestor(def))
      continue;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

/// True iff `iv` appears among the access map operands of `avm`. If the IV is
/// not an operand the address is provably invariant in this loop's IV.
static bool accessUsesIV(const AffineValueMap &avm, Value iv) {
  return llvm::is_contained(avm.getOperands(), iv);
}

static AffineValueMap getAccessMap(Operation *op) {
  if (auto ld = dyn_cast<AffineLoadOp>(op))
    return AffineValueMap(ld.getAffineMap(), ld.getMapOperands());
  auto st = cast<AffineStoreOp>(op);
  return AffineValueMap(st.getAffineMap(), st.getMapOperands());
}

/// True iff `forOp` carries a reduction recurrence across its own IV.
///
/// Two forms:
///   (i)  SSA reduction: the loop carries iter_args (the accumulator).
///   (ii) Memref reduction (post-Polygeist): an affine.load and affine.store to
///        the SAME memref with the SAME access map and operands, whose address
///        is INVARIANT in this loop's IV, where the loaded value transitively
///        FEEDS the stored value. That is `mem[invariant] = f(mem[invariant],...)`
///        -- e.g. tmp1[i][j] += prod[i][j][k] over k.
///
/// Deliberately does NOT fire for:
///   - prod[i][j][k] = ...   (store address varies with IV -> not invariant)
///   - pure read / pure write (no load feeding the store)
///   - load+store of one memref where the load does not feed the store
static bool isReductionLoop(AffineForOp forOp) {
  // (i) SSA reduction.
  if (forOp.getNumIterOperands() != 0)
    return true;

  Value iv = forOp.getInductionVar();
  Block *body = forOp.getBody();

  llvm::SmallVector<AffineLoadOp, 4> loads;
  llvm::SmallVector<AffineStoreOp, 4> stores;
  body->walk([&](Operation *op) {
    if (auto ld = dyn_cast<AffineLoadOp>(op))
      loads.push_back(ld);
    else if (auto st = dyn_cast<AffineStoreOp>(op))
      stores.push_back(st);
  });

  for (AffineStoreOp st : stores) {
    AffineValueMap stMap = getAccessMap(st);
    if (accessUsesIV(stMap, iv))
      continue;
    for (AffineLoadOp ld : loads) {
      if (ld.getMemRef() != st.getMemRef())
        continue;
      AffineValueMap ldMap = getAccessMap(ld);
      if (accessUsesIV(ldMap, iv))
        continue;
      if (ldMap.getAffineMap() != stMap.getAffineMap())
        continue;
      if (ldMap.getOperands() != stMap.getOperands())
        continue;
      if (valueFeeds(ld.getResult(), st.getValueToStore(), forOp))
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Unroll helper
//===----------------------------------------------------------------------===//

/// Fully unroll every affine.for strictly nested inside `p`, innermost-first.
/// Models "pipelining p unrolls all loops below it".
static void unrollStrictlyInside(AffineForOp p) {
  SmallVector<AffineForOp> inners;
  // Post-order walk: innermost loops first, so a child is unrolled before its
  // parent is touched.
  p.getBody()->walk([&](AffineForOp f) { inners.push_back(f); });
  for (AffineForOp f : inners)
    if (failed(loopUnrollFull(f)))
      LLVM_DEBUG(llvm::dbgs() << "  loopUnrollFull failed on nested loop\n");
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct AutoPipelineUnrollPass
    : public PassWrapper<AutoPipelineUnrollPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoPipelineUnrollPass)

  AutoPipelineUnrollPass() = default;
  AutoPipelineUnrollPass(const AutoPipelineUnrollPass &other)
      : PassWrapper(other) {}

  Option<unsigned> threshold{
      *this, "threshold",
      llvm::cl::desc("Auto-pipeline trip-count threshold (Vitis "
                     "config_compile -pipeline_loops, default 64)"),
      llvm::cl::init(64)};

  StringRef getArgument() const final { return "auto-pipeline-unroll"; }
  StringRef getDescription() const final {
    return "Model Vitis auto-pipeline loop selection and apply the implied full "
           "unroll of small innermost reductions. Run before "
           "coalesce-perfect-nests.";
  }

/// Count ops in `loop`'s body that are NOT the given nested loop and not the
  /// terminator. >0 means real inter-level code (the imperfect-nest signal).
  static unsigned interLevelOpCount(AffineForOp loop, AffineForOp nestedToIgnore) {
    unsigned n = 0;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (&op == nestedToIgnore.getOperation())
        continue;
      ++n;
    }
    return n;
  }

  void processNest(AffineForOp root) {
    SmallVector<AffineForOp> band;
    getPerfectlyNestedLoops(band, root);

    // Direct children loops of root.
    SmallVector<AffineForOp> childLoops;
    for (Operation &op : root.getBody()->without_terminator())
      if (auto f = dyn_cast<AffineForOp>(&op))
        childLoops.push_back(f);

    // --- Case B: IMPERFECT nest = root has a nested loop AND real inter-level
    //     code sitting beside it (e.g. bicg lprd_1: scalar buff_p/buff_r/...
    //     stores next to the lprd_2 copy loop). Vitis pipelines root at that
    //     level, so each inner loop with TC <= threshold unrolls -- reduction
    //     or not (lprd_2 is a plain copy and still unrolls).
    //
    //     CRITICAL: only when there is GENUINE inter-level code. A clean nest
    //     whose body is *only* the inner loop (lp1 -> lp2) is NOT imperfect;
    //     it flattens, and its inner unrolls only if it is a reduction
    //     (handled by Case A). Without this guard Case B wrongly unrolls
    //     non-reduction inners like bicg lp2.
    if (!childLoops.empty()) {
      // Inter-level code exists if any child loop has sibling ops in root.
      bool hasInterLevelCode = false;
      for (AffineForOp child : childLoops)
        if (interLevelOpCount(root, child) > 0) {
          hasInterLevelCode = true;
          break;
        }
      // Multiple sibling loops also make root imperfect (can't be one perfect
      // band), and each is pipelined-from-root -> unroll.
      bool multipleChildLoops = childLoops.size() > 1;

      if (hasInterLevelCode || multipleChildLoops) {
        for (AffineForOp f : childLoops) {
          std::optional<uint64_t> tc = getConstantTripCount(f);
          if (!tc || *tc == 0 || *tc > threshold)
            continue;
          LLVM_DEBUG(llvm::dbgs()
                     << "imperfect nest: pipeline @ " << root.getLoc()
                     << ", unroll inner (trip " << *tc << ") @ " << f.getLoc()
                     << "\n");
          if (failed(loopUnrollFull(f)))
            LLVM_DEBUG(llvm::dbgs() << "  loopUnrollFull failed\n");
        }
        return;
      }
    }

    // --- Case A: PERFECT nest (root's body is exactly one inner loop, no
    //     inter-level code). Reduction-unroll logic on the innermost loop. ---
    if (band.empty())
      return;
    AffineForOp inner = band.back();

    if (band.size() < 2)
      return; // no parent loop -> pipelined in place.

    std::optional<uint64_t> innerTrip = getConstantTripCount(inner);
    if (!innerTrip || *innerTrip == 0)
      return;

    // Non-reduction perfect inner -> flattens. Leave. (bicg lp2: s_out[j]
    // address varies with inner IV j -> not a reduction -> flatten, no unroll.)
    if (!isReductionLoop(inner))
      return;

    // Reduction, TC > threshold -> in-place pipeline (RecMII). Leave.
    if (*innerTrip > threshold) {
      LLVM_DEBUG(llvm::dbgs()
                 << "reduction, trip " << *innerTrip << " > " << threshold
                 << ": leaving for in-place pipeline @ " << inner.getLoc()
                 << "\n");
      return;
    }

    // Reduction, TC <= threshold -> pipeline parent, unroll reduction axis.
    // (bicg lp4: q_out[i] address invariant in inner IV j -> reduction ->
    //  unroll. gemm lp3 likewise.)
    AffineForOp parent = band[band.size() - 2];
    LLVM_DEBUG(llvm::dbgs()
               << "reduction, trip " << *innerTrip << " <= " << threshold
               << ": pipeline parent @ " << parent.getLoc()
               << ", unroll below\n");
    unrollStrictlyInside(parent);
  }


  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect outermost affine.for nests first (parent is not an affine.for);
    // processing mutates bodies via unroll. A single pass suffices: unrolling a
    // reduction cannot create a new innermost reduction Vitis would also unroll
    // (Vitis itself unrolls one level and pipelines the parent).
    SmallVector<AffineForOp> roots;
    func.walk([&](AffineForOp f) {
      if (!isa<AffineForOp>(f->getParentOp()))
        roots.push_back(f);
    });
    for (AffineForOp root : roots)
      processNest(root);
  }
};

} // namespace

namespace circt::hls_analysis {

std::unique_ptr<mlir::Pass> createAutoPipelineUnrollPass() {
  return std::make_unique<AutoPipelineUnrollPass>();
}

void registerAutoPipelineUnrollPass() {
  ::mlir::PassRegistration<AutoPipelineUnrollPass>();
}

} // namespace circt::hls_analysis
