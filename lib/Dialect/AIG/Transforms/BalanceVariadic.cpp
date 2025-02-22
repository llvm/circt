//===- BalanceVariadic.cpp - Lowering Variadic to Binary Ops ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers variadic AndInverter operations to balanced binary
// AndInverter operations.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/PriorityQueue.h"

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/AIG/Analysis/OpDepthAnalysis.h"

#define DEBUG_TYPE "aig-balance-variadic"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_BALANCEVARIADIC
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

namespace {
/// For wrapping Value and complement information into one object
struct Signal {
  Value value;
  bool complement;

  Signal() = default;
  Signal(Value v, bool complement) : value(v), complement(complement) {}

  bool isComplement() const { return complement; }
  Value getValue() const { return value; }
};

struct BalanceVariadicDriver {
  BalanceVariadicDriver(mlir::IRRewriter &rewriter,
                        aig::analysis::OpDepthAnalysis *opDepthAnalysis)
      : rewriter(rewriter), opDepthAnalysis(opDepthAnalysis) {}

  struct PairSorter {
    bool operator()(const std::pair<size_t, Signal> &lhs,
                    const std::pair<size_t, Signal> &rhs) const {
      // First compare by level (higher level = lower priority)
      if (lhs.first != rhs.first)
        return lhs.first > rhs.first;

      // If levels are equal, compare by argnumber or result number for
      // deterministic ordering
      auto *lop = lhs.second.getValue().getDefiningOp();
      auto *rop = rhs.second.getValue().getDefiningOp();
      if (lop && rop) {
        return lop->isBeforeInBlock(rop);
      }

      if (!lop && rop)
        return false;

      if (lop && !rop)
        return true;

      BlockArgument larg = cast<BlockArgument>(lhs.second.getValue());
      BlockArgument rarg = cast<BlockArgument>(rhs.second.getValue());
      return larg.getArgNumber() > rarg.getArgNumber();
    }
  };

  using NodeLevelHeap =
      llvm::PriorityQueue<std::pair<size_t, Signal>,
                          std::vector<std::pair<size_t, Signal>>, PairSorter>;

  void balanceVariadicAndInverterOp(AndInverterOp op) {
    rewriter.setInsertionPoint(op);

    NodeLevelHeap sortByLevel;
    for (auto [fanin, inverted] :
         llvm::zip(op.getOperands(), op.getInverted())) {
      auto faninOp = fanin.getDefiningOp<AndInverterOp>();
      size_t level = faninOp ? opDepthAnalysis->updateLevel(faninOp, true) : 0;
      sortByLevel.push({level, Signal(fanin, inverted)});
    }

    // extract the top two elements with minimum level
    // and replace them with a new AndInverterOp
    while (sortByLevel.size() > 2) {
      auto [llv, lhs] = sortByLevel.top();
      sortByLevel.pop();
      auto [rlv, rhs] = sortByLevel.top();
      sortByLevel.pop();

      auto balanced = rewriter.create<AndInverterOp>(
          op.getLoc(), lhs.getValue(), rhs.getValue(), lhs.isComplement(),
          rhs.isComplement());

      size_t level = std::max(llv, rlv) + 1;
      sortByLevel.push({level, Signal(balanced, false)});
    }

    switch (sortByLevel.size()) {
    case 0:
      break;
    case 1: {
      auto signal = sortByLevel.top().second;
      sortByLevel.pop();
      rewriter.replaceOp(op, signal.getValue());
      break;
    }
    default:
      auto lhs = sortByLevel.top().second;
      sortByLevel.pop();
      auto rhs = sortByLevel.top().second;

      rewriter.replaceOp(op, rewriter.create<AndInverterOp>(
                                 op.getLoc(), lhs.getValue(), rhs.getValue(),
                                 lhs.isComplement(), rhs.isComplement()));
    }
  }

  void balanceRecursive(AndInverterOp op) {
    if (visited.count(op))
      return;

    visited.insert(op);
    assert(!op->use_empty());

    for (auto fanin : op.getOperands()) {
      auto faninOp = fanin.getDefiningOp<AndInverterOp>();
      if (faninOp) {
        balanceRecursive(faninOp);
      }
    }

    if (op.getOperands().size() <= 2)
      return;

    balanceVariadicAndInverterOp(op);
    // opDepthAnalysis->updateLevel(op, true);
  }

  void balancing() {
    // Balance each variadic AndInverterOp in reverse topological order
    // Will ignore dangling internal AIG nodes
    for (AndInverterOp po : opDepthAnalysis->getPOs()) {
      balanceRecursive(po);
    }
  }

private:
  DenseSet<Operation *> visited;
  mlir::IRRewriter &rewriter;
  aig::analysis::OpDepthAnalysis *opDepthAnalysis;
};

struct BalanceVariadicPass
    : public impl::BalanceVariadicBase<BalanceVariadicPass> {
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Balance Variadic pass
//===----------------------------------------------------------------------===//
void BalanceVariadicPass::runOnOperation() {
  auto *opDepthAnalysis = &getAnalysis<aig::analysis::OpDepthAnalysis>();

  auto module = getOperation();
  MLIRContext *ctx = module->getContext();
  mlir::IRRewriter rewriter(ctx);

  BalanceVariadicDriver driver(rewriter, opDepthAnalysis);
  driver.balancing();
}
