//===- LatencyRetiming.cpp - Implement LatencyRetiming Pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "arc-latency-retiming"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LATENCYRETIMING
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
struct LatencyRetimingStatistics {
  unsigned numOpsRemoved = 0;
  unsigned latencyUnitsSaved = 0;
};

/// Absorb the latencies from predecessor states to collapse shift registers and
/// reduce the overall amount of latency units in the design.
struct LatencyRetimingPattern
    : mlir::OpInterfaceRewritePattern<ClockedOpInterface> {
  LatencyRetimingPattern(MLIRContext *context, SymbolCache &symCache,
                         LatencyRetimingStatistics &statistics)
      : OpInterfaceRewritePattern<ClockedOpInterface>(context),
        symCache(symCache), statistics(statistics) {}

  LogicalResult matchAndRewrite(ClockedOpInterface op,
                                PatternRewriter &rewriter) const final;

private:
  SymbolCache &symCache;
  LatencyRetimingStatistics &statistics;
};

} // namespace

LogicalResult
LatencyRetimingPattern::matchAndRewrite(ClockedOpInterface op,
                                        PatternRewriter &rewriter) const {
  uint32_t minPrevLatency = UINT_MAX;
  SetVector<ClockedOpInterface> predecessors;
  Value clock;

  auto hasEnableOrReset = [](Operation *op) -> bool {
    if (auto stateOp = dyn_cast<StateOp>(op))
      if (stateOp.getReset() || stateOp.getEnable())
        return true;
    return false;
  };

  // Restrict this pattern to call and state ops only. In the future we could
  // also add support for memory write operations.
  if (!isa<CallOp, StateOp>(op.getOperation()))
    return failure();

  // In principle we could support enables and resets but would have to check
  // that all involved states have the same.
  if (hasEnableOrReset(op))
    return failure();

  assert(isa<mlir::CallOpInterface>(op.getOperation()) &&
         "state and call operations call arcs and thus have to implement the "
         "CallOpInterface");
  auto callOp = cast<mlir::CallOpInterface>(op.getOperation());

  for (auto input : callOp.getArgOperands()) {
    auto predOp = input.getDefiningOp<ClockedOpInterface>();

    // Only support call and state ops for the predecessors as well.
    if (!predOp || !isa<CallOp, StateOp>(predOp.getOperation()))
      return failure();

    // Conditions for both StateOp and CallOp
    if (predOp->hasAttr("name") || predOp->hasAttr("names"))
      return failure();

    // Check for a use-def cycle since we can be in a graph region.
    if (predOp == op)
      return failure();

    if (predOp.getClock() && op.getClock() &&
        predOp.getClock() != op.getClock())
      return failure();

    if (predOp->getParentRegion() != op->getParentRegion())
      return failure();

    if (hasEnableOrReset(predOp))
      return failure();

    // Check that the predecessor state does not have another user since then
    // we cannot change its latency attribute without also changing it for the
    // other users. This is not supported yet and thus we just fail.
    if (llvm::any_of(predOp->getUsers(),
                     [&](auto *user) { return user != op; }))
      return failure();

    // We check that all clocks are the same if present. Here we remember that
    // clock. If none of the involved operations have a clock, they must have
    // latency 0 and thus `minPrevLatency = 0` leading to early failure below.
    if (!clock) {
      if (predOp.getClock())
        clock = predOp.getClock();
      if (auto clockDomain = predOp->getParentOfType<ClockDomainOp>())
        clock = clockDomain.getClock();
    }

    predecessors.insert(predOp);
    minPrevLatency = std::min(minPrevLatency, predOp.getLatency());
  }

  if (minPrevLatency == 0 || minPrevLatency == UINT_MAX)
    return failure();

  auto setLatency = [&](Operation *op, uint64_t newLatency, Value clock) {
    assert((isa<StateOp, CallOp>(op)) && "must be a state or call op");
    bool isInClockDomain = op->getParentOfType<ClockDomainOp>();

    if (auto stateOp = dyn_cast<StateOp>(op)) {
      if (newLatency == 0) {
        if (cast<DefineOp>(symCache.getDefinition(stateOp.getArcAttr()))
                .isPassthrough()) {
          rewriter.replaceOp(stateOp, stateOp.getInputs());
          ++statistics.numOpsRemoved;
          return;
        }
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<CallOp>(op, stateOp.getOutputs().getTypes(),
                                            stateOp.getArcAttr(),
                                            stateOp.getInputs());
        return;
      }

      rewriter.modifyOpInPlace(op, [&]() {
        stateOp.setLatency(newLatency);
        if (!stateOp.getClock() && !isInClockDomain)
          stateOp.getClockMutable().assign(clock);
      });
      return;
    }

    if (auto callOp = dyn_cast<CallOp>(op); callOp && newLatency > 0)
      rewriter.replaceOpWithNewOp<StateOp>(
          op, callOp.getArcAttr(), callOp->getResultTypes(),
          isInClockDomain ? Value{} : clock, Value{}, newLatency,
          callOp.getInputs());
  };

  setLatency(op, op.getLatency() + minPrevLatency, clock);
  for (auto prevOp : predecessors) {
    statistics.latencyUnitsSaved += minPrevLatency;
    auto newLatency = prevOp.getLatency() - minPrevLatency;
    setLatency(prevOp, newLatency, {});
  }
  statistics.latencyUnitsSaved -= minPrevLatency;

  return success();
}

//===----------------------------------------------------------------------===//
// LatencyRetiming pass
//===----------------------------------------------------------------------===//

namespace {
struct LatencyRetimingPass
    : arc::impl::LatencyRetimingBase<LatencyRetimingPass> {
  void runOnOperation() override;
};
} // namespace

void LatencyRetimingPass::runOnOperation() {
  SymbolCache cache;
  cache.addDefinitions(getOperation());

  LatencyRetimingStatistics statistics;

  RewritePatternSet patterns(&getContext());
  patterns.add<LatencyRetimingPattern>(&getContext(), cache, statistics);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();

  numOpsRemoved = statistics.numOpsRemoved;
  latencyUnitsSaved = statistics.latencyUnitsSaved;
}

std::unique_ptr<Pass> arc::createLatencyRetimingPass() {
  return std::make_unique<LatencyRetimingPass>();
}
