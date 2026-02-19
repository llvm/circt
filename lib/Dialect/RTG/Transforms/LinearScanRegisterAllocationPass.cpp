//===- LinearScanRegisterAllocationPass.cpp - Register Allocation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass allocates registers using a simple linear scan algorithm.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_LINEARSCANREGISTERALLOCATIONPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "rtg-linear-scan-register-allocation"

namespace {

/// Represents a register and its live range.
struct RegisterLiveRange {
  rtg::RegisterAttrInterface fixedReg;
  rtg::VirtualRegisterOp regOp;
  unsigned start;
  unsigned end;
};

class LinearScanRegisterAllocationPass
    : public circt::rtg::impl::LinearScanRegisterAllocationPassBase<
          LinearScanRegisterAllocationPass> {
public:
  void runOnOperation() override;
};

} // end namespace

static void expireOldInterval(SmallVector<RegisterLiveRange *> &active,
                              RegisterLiveRange *reg) {
  // TODO: use a better datastructure for 'active'
  llvm::sort(active, [](auto *a, auto *b) { return a->end < b->end; });

  for (auto *iter = active.begin(); iter != active.end(); ++iter) {
    auto *a = *iter;
    if (a->end >= reg->start)
      return;

    active.erase(iter--);
  }
}

static LogicalResult allocateRegistersForSegment(rtg::SegmentOp segOp) {
  DenseMap<Operation *, unsigned> opIndices;
  DenseMap<unsigned, Operation *> opByIndex;
  unsigned maxIdx;
  for (auto [i, op] : llvm::enumerate(*segOp.getBody())) {
    // TODO: ideally check that the IR is already fully elaborated
    opIndices[&op] = i;
    opByIndex[i] = &op;
    maxIdx = i;
  }

  // Collect all the register intervals we have to consider.
  SmallVector<std::unique_ptr<RegisterLiveRange>> regRanges;
  SmallVector<RegisterLiveRange *> active;
  for (auto &op : *segOp.getBody()) {
    if (!isa<rtg::ConstantOp, rtg::VirtualRegisterOp>(&op) ||
        !isa<rtg::RegisterTypeInterface>(op.getResult(0).getType()))
      continue;

    RegisterLiveRange lr;
    lr.start = maxIdx;
    lr.end = 0;

    if (auto regOp = dyn_cast<rtg::VirtualRegisterOp>(&op))
      lr.regOp = regOp;

    if (auto regOp = dyn_cast<rtg::ConstantOp>(&op)) {
      auto reg = dyn_cast<rtg::RegisterAttrInterface>(regOp.getValue());
      if (!reg) {
        op.emitError("expected register attribute");
        return failure();
      }
      lr.fixedReg = reg;
    }

    for (auto *user : op.getUsers()) {
      if (!isa<rtg::InstructionOpInterface, rtg::ValidateOp>(user)) {
        user->emitError("only operations implementing 'InstructionOpInterface' "
                        "and 'rtg.validate' are allowed to use registers");
        return failure();
      }

      // TODO: support labels and control-flow loops (jumps in general)
      unsigned idx = opIndices.at(user);
      lr.start = std::min(lr.start, idx);
      lr.end = std::max(lr.end, idx);
    }

    regRanges.emplace_back(std::make_unique<RegisterLiveRange>(lr));

    // Reserve fixed registers from the start. It will be made available again
    // past the interval end. Not reserving it from the start can lead to the
    // same register being chosen for a virtual register that overlaps with the
    // fixed register interval.
    // TODO: don't overapproximate that much
    if (!lr.regOp)
      active.push_back(regRanges.back().get());
  }

  // Sort such that we can process registers by increasing interval start.
  llvm::sort(regRanges, [](const auto &a, const auto &b) {
    return a->start < b->start || (a->start == b->start && !a->regOp);
  });

  for (auto &lr : regRanges) {
    // Make registers out of live range available again.
    expireOldInterval(active, lr.get());

    // Handle already fixed registers.
    if (!lr->regOp)
      continue;

    // Handle virtual registers.
    auto configAttr =
        cast<rtg::VirtualRegisterConfigAttr>(lr->regOp.getAllowedRegsAttr());
    rtg::RegisterAttrInterface availableReg;
    for (auto reg : configAttr.getAllowedRegs()) {
      if (llvm::none_of(active, [&](auto *r) { return r->fixedReg == reg; })) {
        availableReg = cast<rtg::RegisterAttrInterface>(reg);
        break;
      }
    }

    if (!availableReg) {
      auto err = lr->regOp->emitError(
          "need to spill this register, but not supported yet");
      for (auto *a : active)
        err.attachNote(a->regOp->getLoc())
            << "overlapping live-range with this register that is set to '"
            << a->fixedReg.getRegisterAssembly() << "'";
      err.attachNote(opByIndex[lr->start]->getLoc())
          << "register live-range starts here";
      err.attachNote(opByIndex[lr->end]->getLoc())
          << "register live-range ends here";
      return failure();
    }

    lr->fixedReg = availableReg;
    active.push_back(lr.get());
  }

  LLVM_DEBUG({
    for (auto &regRange : regRanges) {
      llvm::dbgs() << "Start: " << regRange->start << ", End: " << regRange->end
                   << ", Selected: " << regRange->fixedReg << "\n";
    }
    llvm::dbgs() << "\n";
  });

  for (auto &reg : regRanges) {
    // No need to fix already fixed registers.
    if (!reg->regOp)
      continue;

    IRRewriter rewriter(reg->regOp);
    rewriter.replaceOpWithNewOp<rtg::ConstantOp>(reg->regOp, reg->fixedReg);
  }

  return success();
}

void LinearScanRegisterAllocationPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Processing "
                          << OpWithFlags(getOperation(),
                                         OpPrintingFlags().skipRegions())
                          << "\n\n");

  if (getOperation()->getNumRegions() != 1 ||
      getOperation()->getRegion(0).getBlocks().size() != 1) {
    getOperation()->emitError("expected a single region with a single block");
    return signalPassFailure();
  }

  for (auto segOp : getOperation()->getRegion(0).getOps<rtg::SegmentOp>()) {
    if (failed(allocateRegistersForSegment(segOp)))
      return signalPassFailure();
  }
}
