//===- LowerVariadic.cpp - Lowering Variadic to Binary Ops ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers variadic AndInverter operations to binary AndInverter
// operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGAnalysis.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/PriorityQueue.h"
#include <mlir/Analysis/TopologicalSortUtils.h>

#define DEBUG_TYPE "aig-lower-variadic"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_LOWERVARIADIC
#define GEN_PASS_DEF_LOWERVARIADICGLOBAL
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {
static Value lowerVariadicAndInverterOp(AndInverterOp op, OperandRange operands,
                                        ArrayRef<bool> inverts,
                                        PatternRewriter &rewriter) {
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    if (inverts[0])
      return rewriter.create<AndInverterOp>(op.getLoc(), operands[0], true);
    else
      return operands[0];
  case 2:
    return rewriter.create<AndInverterOp>(op.getLoc(), operands[0], operands[1],
                                          inverts[0], inverts[1]);
  default:
    auto firstHalf = operands.size() / 2;
    auto lhs =
        lowerVariadicAndInverterOp(op, operands.take_front(firstHalf),
                                   inverts.take_front(firstHalf), rewriter);
    auto rhs =
        lowerVariadicAndInverterOp(op, operands.drop_front(firstHalf),
                                   inverts.drop_front(firstHalf), rewriter);
    return rewriter.create<AndInverterOp>(op.getLoc(), lhs, rhs);
  }

  return Value();
}
static Value lowerVariadicAndInverterOp(AndInverterOp op, OperandRange operands,
                                        ArrayRef<bool> inverts,
                                        mlir::ImplicitLocOpBuilder &rewriter) {
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    if (inverts[0])
      return rewriter.create<AndInverterOp>(op.getLoc(), operands[0], true);
    else
      return operands[0];
  case 2:
    return rewriter.create<AndInverterOp>(op.getLoc(), operands[0], operands[1],
                                          inverts[0], inverts[1]);
  default:
    auto firstHalf = operands.size() / 2;
    auto lhs =
        lowerVariadicAndInverterOp(op, operands.take_front(firstHalf),
                                   inverts.take_front(firstHalf), rewriter);
    auto rhs =
        lowerVariadicAndInverterOp(op, operands.drop_front(firstHalf),
                                   inverts.drop_front(firstHalf), rewriter);
    return rewriter.create<AndInverterOp>(op.getLoc(), lhs, rhs);
  }
}

struct VariadicOpConversion : OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AndInverterOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() <= 2)
      return failure();

    // TODO: This is a naive implementation that creates a balanced binary tree.
    //       We can improve by analyzing the dataflow and creating a tree that
    //       improves the critical path or area.
    rewriter.replaceOp(op,
                       lowerVariadicAndInverterOp(op, op.getOperands(),
                                                  op.getInverted(), rewriter));
    return success();
  }
};

} // namespace

static void populateLowerVariadicPatterns(RewritePatternSet &patterns) {
  patterns.add<VariadicOpConversion>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Lower Variadic pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerVariadicPass : public impl::LowerVariadicBase<LowerVariadicPass> {
  void runOnOperation() override;
};
struct LowerVariadicGlobalPass
    : public impl::LowerVariadicGlobalBase<LowerVariadicGlobalPass> {
  void runOnOperation() override;
};
} // namespace

void LowerVariadicPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLowerVariadicPatterns(patterns);
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));

  if (failed(mlir::applyPatternsGreedily(getOperation(), frozen)))
    return signalPassFailure();
}

void LowerVariadicGlobalPass::runOnOperation() {
  auto &longestPath = getAnalysis<circt::aig::LongestPathAnalysis>();

  auto module = getOperation();
  auto *ctx = &getContext();
  struct LocalState {
    hw::HWModuleOp module;
    LocalState(hw::HWModuleOp module) : module(module) {}
    DenseMap<Value, int64_t> cost;
  };
  SmallVector<LocalState> hwMods;
  for (auto hwMod : module.getOps<hw::HWModuleOp>()) {
    if (!longestPath.isAnalysisAvaiable(hwMod))
      continue;
    hwMods.emplace_back(hwMod);
  }

  // 1. Compute cost for all values in the module.
  mlir::failableParallelForEach(ctx, hwMods, [&](auto &hwMod) {
    auto module = hwMod.module;
    auto &cost = hwMod.cost;
    for (auto arg : module.getBodyBlock()->getBlockArguments())
      cost[arg] = longestPath.getAverageMaxDelay(arg);
    hwMod.module->walk([&](Operation *op) {
      if (auto instance = dyn_cast<hw::InstanceOp>(op))
        for (auto result : instance.getResults())
          cost[result] = longestPath.getAverageMaxDelay(result);
      if (auto reg = dyn_cast<seq::FirRegOp>(op))
        cost[reg] = 0;
    });
  });

  mlir::failableParallelForEach(ctx, hwMods, [&](auto &hwMod) {
    auto module = hwMod.module;
    auto &cost = hwMod.cost;
    mlir::sortTopologically(module.getBodyBlock(), [](Value v, Operation *op) {
      return cost.count(v);
    });
    auto setMaximum = [&](Operation *op, int64_t additionalCost = 0) {
      int64_t result = 0;
      for (auto operand : op->getOperands())
        result = std::max(result, cost[operand]);
      for (auto r : op->getResults())
        cost[r] = result + additionalCost;
    };
    OpBuilder builder(module.getBodyBlock());
    for (auto &op : *module.getBodyBlock()) {
      if (isa<comb::ExtractOp, comb::ReplicateOp, hw::WireOp, comb::ConcatOp>(
              op)) {
        setMaximum(op);
        continue;
      }
      if (auto andInverter = dyn_cast<AndInverterOp>(op)) {
        if (andInverter.getInputs().size() <= 2) {
          setMaximum(op, 1);
          continue;
        }

        // Lower.
        llvm::PriorityQueue<std::tuple<int64_t, Value, bool>,
                            std::vector<std::tuple<int64_t, Value, bool>>>
            queue;
        for (auto operand : andInverter->getOperands())
          queue.push({operand, cost[operand]});
        while (queue.size() > 1) {
          auto lhs = queue.top();
          queue.pop();
          auto rhs = queue.top();
          queue.pop();
          auto newOp = builder.create<AndInverterOp>(andInverter->getLoc(),
                                                     lhs.first, rhs.first);
          cost[newOp] = std::max(lhs.second, rhs.second) + 1;
          queue.push({newOp, cost[newOp]});
        }

        return failure();
      }
    }
  });
}
