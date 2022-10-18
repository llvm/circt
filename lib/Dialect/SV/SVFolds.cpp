//===- SVFolds.cpp - Folds + Canonicalization for or to SV operations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace sv;
using namespace comb;
using namespace matchers;

/// A wrapper of `PatternRewriter::replaceOpWithNewOp` to propagate
/// "sv.namehint" attribute. If a replaced op has a "sv.namehint" attribute,
/// this function propagates the name to the new value.
template <typename OpTy, typename... Args>
static OpTy replaceOpWithNewOpAndCopyName(PatternRewriter &rewriter,
                                          Operation *op, Args &&...args) {
  auto name = op->getAttrOfType<StringAttr>("sv.namehint");
  auto newOp =
      rewriter.replaceOpWithNewOp<OpTy>(op, std::forward<Args>(args)...);
  if (name && !newOp->hasAttr("sv.namehint"))
    rewriter.updateRootInPlace(newOp,
                               [&] { newOp->setAttr("sv.namehint", name); });

  return newOp;
}

namespace {

 // change or ( _ == v1, _ == v2, ...) to inside(_,(v1,v1),(v2,v2))
 struct OrOfEq : public RewritePattern {
    OrOfEq(MLIRContext* context)
    : RewritePattern(comb::OrOp::getOperationName(), 0, context) {}

  LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter) const override {
    auto orOp = cast<comb::OrOp>(op);

  auto cmpMatch = [](Value v, Value& lhs, Value& c) {
    auto cmpOp = v.getDefiningOp<ICmpOp>();
    if (!cmpOp)
      return false;
    lhs = cmpOp.getLhs();
    c = cmpOp.getRhs().getDefiningOp<hw::ConstantOp>();
    return (bool)c;
  };

  if (orOp.getInputs().size() < 4)
    return failure();

  SmallVector<Value> cmpVals(2*orOp.getInputs().size());
  Value refVal;
    for (size_t i = 0; i < orOp.getInputs().size(); ++i) {
      Value lhs;
      if (!cmpMatch(orOp.getInputs()[i], lhs, cmpVals[2*i]))
      return failure();
      cmpVals[2*i+1] = cmpVals[2*i];
      if (i == 0)
        refVal = lhs;
      if (lhs != refVal)
        return failure();
    }
  replaceOpWithNewOpAndCopyName<sv::SetInclusionOp>(rewriter, op, refVal, cmpVals);

  return success();
}
 };

} // namespace

void SVDialect::getCanonicalizationPatterns(RewritePatternSet& results) const {
  results.insert<OrOfEq>(getContext());
}


