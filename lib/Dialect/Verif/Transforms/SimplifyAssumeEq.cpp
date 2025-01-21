//===- SimplifyAssumeEq.cpp - Formal Preparations ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
// When a symbolic value is assumed equal to another value, the symbolic value
// is replaced with its equal value to simplify.
//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_SIMPLIFYASSUMEEQPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace verif;

namespace {
struct SimplifyAssumeEqPass
    : verif::impl::SimplifyAssumeEqPassBase<SimplifyAssumeEqPass> {
  void runOnOperation() override;
};

LogicalResult tryReplaceEqualValues(mlir::Value &from, mlir::Value &to,
                                    AssumeOp &op, PatternRewriter &rewriter) {
  auto *fromOp = from.getDefiningOp();
  if (fromOp && dyn_cast<verif::SymbolicValueOp>(fromOp)) {
    rewriter.replaceAllUsesWith(from, to);
    rewriter.eraseOp(fromOp);
    rewriter.eraseOp(op);
    return llvm::success();
  }
  return failure();
}

struct AssumeEqRewritePattern : public OpRewritePattern<verif::AssumeOp> {
  using OpRewritePattern<AssumeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AssumeOp op,
                                PatternRewriter &rewriter) const override {
    auto prop = op.getProperty();
    auto *definingOp = prop.getDefiningOp();
    if (!definingOp)
      return llvm::failure();

    auto icmp = dyn_cast<comb::ICmpOp>(definingOp);
    if (!icmp)
      return llvm::failure();

    if (icmp.getPredicate() != comb::ICmpPredicate::eq)
      return llvm::failure();

    auto lhs = icmp.getLhs();
    auto rhs = icmp.getRhs();
    // If lhs is symbolic value, replace with rhs, otherwise
    // if rhs is symbolic value, replace with lhs
    if (succeeded(tryReplaceEqualValues(lhs, rhs, op, rewriter)))
      return success();
    if (succeeded(tryReplaceEqualValues(rhs, lhs, op, rewriter)))
      return success();
    return llvm::failure();
  }
};
} // namespace

void SimplifyAssumeEqPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AssumeEqRewritePattern>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
