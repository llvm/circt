//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace circt::comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_ASSUMETWOVALUED
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {
struct ICmpOpConversion : OpRewritePattern<ICmpOp> {
  using OpRewritePattern<ICmpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ICmpOp op,
                                PatternRewriter &rewriter) const override {
    ICmpPredicate newPredicate;
    switch (op.getPredicate()) {
    case ICmpPredicate::ceq:
    case ICmpPredicate::weq:
      newPredicate = ICmpPredicate::eq;
      break;
    case ICmpPredicate::cne:
    case ICmpPredicate::wne:
      newPredicate = ICmpPredicate::ne;
      break;
    default:
      return failure();
    }
    rewriter.replaceOpWithNewOp<ICmpOp>(op, newPredicate, op.getLhs(),
                                        op.getRhs());
    return success();
  }
};
} // namespace

namespace {
class AssumeTwoValued : public impl::AssumeTwoValuedBase<AssumeTwoValued> {
public:
  using AssumeTwoValuedBase::AssumeTwoValuedBase;

  void runOnOperation() override;
};
} // namespace

void AssumeTwoValued::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ICmpOpConversion>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
