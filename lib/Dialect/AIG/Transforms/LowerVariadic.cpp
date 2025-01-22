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

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "aig-lower-variadic"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_LOWERVARIADIC
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
} // namespace

void LowerVariadicPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLowerVariadicPatterns(patterns);
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));

  if (failed(mlir::applyPatternsGreedily(getOperation(), frozen)))
    return signalPassFailure();
}
