//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-hoist-signals"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_HOISTSIGNALSPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

//===----------------------------------------------------------------------===//
// Probe Hoisting
//===----------------------------------------------------------------------===//

namespace {
/// The struct performing the hoisting in a single region.
struct Hoister {
  Hoister(Region &region) : region(region) {}
  void hoist();

  void findValuesLiveAcrossWait(Liveness &liveness);
  void hoistProbes();

  /// The region we are hoisting ops out of.
  Region &region;

  /// The set of values that are alive across wait ops themselves, or that have
  /// transitive users that are live across wait ops.
  DenseSet<Value> liveAcrossWait;

  /// A lookup table of probes we have already hoisted, for deduplication.
  DenseMap<Value, PrbOp> hoistedProbes;
};
} // namespace

void Hoister::hoist() {
  Liveness liveness(region.getParentOp());
  findValuesLiveAcrossWait(liveness);
  hoistProbes();
}

/// Find all values in the region that are alive across `llhd.wait` operations,
/// or that have transitive uses that are alive across waits. We can only hoist
/// probes that do not feed data flow graphs that are alive across such wait
/// ops. Since control flow edges in `cf.br` and `cf.cond_br` ops are
/// side-effect free, we have no guarantee that moving a probe out of a process
/// could potentially cause other ops to become eligible for a move out of the
/// process. Therefore, if such ops are moved outside of the process, they are
/// effectively moved across the waits and thus sample their operands at
/// different points in time. Only values that are explicitly carried across
/// `llhd.wait`, where the LLHD dialect has control over the control flow
/// semantics, may have probes in their fan-in cone hoisted out.
void Hoister::findValuesLiveAcrossWait(Liveness &liveness) {
  // First find all values that are live across `llhd.wait` operations. We are
  // only interested in values defined in the current region.
  SmallVector<Value> worklist;
  for (auto &block : region)
    if (isa<WaitOp>(block.getTerminator()))
      for (auto value : liveness.getLiveOut(&block))
        if (value.getParentRegion() == &region)
          if (liveAcrossWait.insert(value).second)
            worklist.push_back(value);

  // Propagate liveness information along the use-def chain and across control
  // flow. This will allow us to check `liveAcrossWait` to know if a value
  // escapes across a wait along its use-def chain that isn't an explicit
  // successor operand of the wait op.
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (auto *defOp = value.getDefiningOp()) {
      for (auto operand : defOp->getOperands())
        if (operand.getParentRegion() == &region)
          if (liveAcrossWait.insert(operand).second)
            worklist.push_back(operand);
    } else {
      auto blockArg = cast<BlockArgument>(value);
      for (auto &use : blockArg.getOwner()->getUses()) {
        auto branch = dyn_cast<BranchOpInterface>(use.getOwner());
        if (!branch)
          continue;
        auto operand = branch.getSuccessorOperands(
            use.getOperandNumber())[blockArg.getArgNumber()];
        if (operand.getParentRegion() == &region)
          if (liveAcrossWait.insert(operand).second)
            worklist.push_back(operand);
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << liveAcrossWait.size()
                          << " values live across wait\n");
}

/// Hoist any probes at the beginning of resuming blocks out of the process if
/// their values do not leak across wait ops. Resuming blocks are blocks where
/// all predecessors are `llhd.wait` ops, and the entry block. Only waits
/// without any side-effecting op in between themselves and the beginning of the
/// block can be hoisted.
void Hoister::hoistProbes() {
  for (auto &block : region) {
    // We can only hoist probes in blocks where all predecessors have wait
    // terminators.
    if (!llvm::all_of(block.getPredecessors(), [](auto *predecessor) {
          return isa<WaitOp>(predecessor->getTerminator());
        }))
      continue;

    for (auto &op : llvm::make_early_inc_range(block)) {
      auto probeOp = dyn_cast<PrbOp>(op);

      // We can only hoist probes that have no side-effecting ops between
      // themselves and the beginning of a block. If we see a side-effecting op,
      // give up on this block.
      if (!probeOp) {
        if (isMemoryEffectFree(&op))
          continue;
        else
          break;
      }

      // Only hoist probes that don't leak across wait ops.
      if (liveAcrossWait.contains(probeOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping (live across wait) " << probeOp << "\n");
        continue;
      }

      // We can only hoist probes of signals that are declared outside the
      // process.
      if (!probeOp.getSignal().getParentRegion()->isProperAncestor(&region)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping (local signal) " << probeOp << "\n");
        continue;
      }

      // Move the probe out of the process, trying to reuse any previous probe
      // that we've already hoisted.
      auto &hoistedOp = hoistedProbes[probeOp.getSignal()];
      if (hoistedOp) {
        LLVM_DEBUG(llvm::dbgs() << "- Replacing " << probeOp << "\n");
        probeOp.replaceAllUsesWith(hoistedOp.getResult());
        probeOp.erase();
      } else {
        LLVM_DEBUG(llvm::dbgs() << "- Hoisting " << probeOp << "\n");
        probeOp->moveBefore(region.getParentOp());
        hoistedOp = probeOp;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct HoistSignalsPass
    : public llhd::impl::HoistSignalsPassBase<HoistSignalsPass> {
  void runOnOperation() override;
};
} // namespace

void HoistSignalsPass::runOnOperation() {
  SmallVector<Region *> regions;
  getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ProcessOp, FinalOp>(op)) {
      auto &region = op->getRegion(0);
      if (!region.empty())
        regions.push_back(&region);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  for (auto *region : regions)
    Hoister(*region).hoist();
}
