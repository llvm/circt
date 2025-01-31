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

void LinearScanRegisterAllocationPass::runOnOperation() {
  auto testOp = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "=== Processing test @" << testOp.getSymName()
                          << "\n\n");

  DenseMap<Operation *, unsigned> opIndices;
  unsigned maxIdx;
  for (auto [i, op] : llvm::enumerate(*testOp.getBody())) {
    // TODO: ideally check that the IR is already fully elaborated
    opIndices[&op] = i;
    maxIdx = i;
  }

  // Collect all the register intervals we have to consider.
  SmallVector<std::unique_ptr<RegisterLiveRange>> regRanges;
  SmallVector<RegisterLiveRange *> active;
  for (auto &op : *testOp.getBody()) {
    if (!isa<rtg::FixedRegisterOp, rtg::VirtualRegisterOp>(&op))
      continue;

    RegisterLiveRange lr;
    lr.start = maxIdx;
    lr.end = 0;

    if (auto regOp = dyn_cast<rtg::VirtualRegisterOp>(&op))
      lr.regOp = regOp;

    if (auto regOp = dyn_cast<rtg::FixedRegisterOp>(&op))
      lr.fixedReg = regOp.getReg();

    for (auto *user : op.getUsers()) {
      if (!isa<rtg::InstructionOpInterface>(user)) {
        user->emitError("only operations implementing 'InstructionOpInterface "
                        "are allowed to use registers");
        return signalPassFailure();
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
    rtg::RegisterAttrInterface availableReg;
    for (auto reg : lr->regOp.getAllowedRegs()) {
      if (llvm::none_of(active, [&](auto *r) { return r->fixedReg == reg; })) {
        availableReg = cast<rtg::RegisterAttrInterface>(reg);
        break;
      }
    }

    if (!availableReg) {
      ++numRegistersSpilled;
      lr->regOp->emitError(
          "need to spill this register, but not supported yet");
      return signalPassFailure();
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
    rewriter.replaceOpWithNewOp<rtg::FixedRegisterOp>(reg->regOp,
                                                      reg->fixedReg);
  }
}
