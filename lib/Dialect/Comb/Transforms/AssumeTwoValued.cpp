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
                                        op.getRhs(), /*twoState=*/true);
    return success();
  }
};

template <typename OpTy>
struct AddTwoStateFlag : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (op.getTwoState())
      return failure();
    rewriter.modifyOpInPlace(op, [&] { op.setTwoState(true); });
    return success();
  }
};

using AddOpTwoStateFlag = AddTwoStateFlag<AddOp>;
using AndOpTwoStateFlag = AddTwoStateFlag<AndOp>;
using DivSOpTwoStateFlag = AddTwoStateFlag<DivSOp>;
using DivUOpTwoStateFlag = AddTwoStateFlag<DivUOp>;
using ModSOpTwoStateFlag = AddTwoStateFlag<ModSOp>;
using ModUOpTwoStateFlag = AddTwoStateFlag<ModUOp>;
using MulOpTwoStateFlag = AddTwoStateFlag<MulOp>;
using MuxOpTwoStateFlag = AddTwoStateFlag<MuxOp>;
using OrOpTwoStateFlag = AddTwoStateFlag<OrOp>;
using ParityOpTwoStateFlag = AddTwoStateFlag<ParityOp>;
using ShlOpTwoStateFlag = AddTwoStateFlag<ShlOp>;
using ShrSOpTwoStateFlag = AddTwoStateFlag<ShrSOp>;
using ShrUOpTwoStateFlag = AddTwoStateFlag<ShrUOp>;
using SubOpTwoStateFlag = AddTwoStateFlag<SubOp>;
using XorOpTwoStateFlag = AddTwoStateFlag<XorOp>;

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
  patterns.add<ICmpOpConversion, AddOpTwoStateFlag, AndOpTwoStateFlag,
               DivSOpTwoStateFlag, DivUOpTwoStateFlag, ModSOpTwoStateFlag,
               ModUOpTwoStateFlag, MulOpTwoStateFlag, MuxOpTwoStateFlag,
               OrOpTwoStateFlag, ParityOpTwoStateFlag, ShlOpTwoStateFlag,
               ShrSOpTwoStateFlag, ShrUOpTwoStateFlag, SubOpTwoStateFlag,
               XorOpTwoStateFlag>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
