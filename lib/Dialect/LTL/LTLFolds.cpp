//===- LTLFolds.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace ltl;
using namespace mlir;

/// Concatenate two value ranges into a larger range. Useful for declarative
/// rewrites.
static SmallVector<Value> concatValues(ValueRange a, ValueRange b) {
  SmallVector<Value> v;
  v.append(a.begin(), a.end());
  v.append(b.begin(), b.end());
  return v;
}

/// Inline all `ConcatOp`s in a range of values.
static SmallVector<Value> flattenConcats(ValueRange values) {
  SmallVector<Value> flatInputs;
  for (auto value : values) {
    if (auto concatOp = value.getDefiningOp<ConcatOp>()) {
      auto inputs = concatOp.getInputs();
      flatInputs.append(inputs.begin(), inputs.end());
    } else {
      flatInputs.push_back(value);
    }
  }
  return flatInputs;
}

//===----------------------------------------------------------------------===//
// Declarative Rewrites
//===----------------------------------------------------------------------===//

namespace patterns {
#include "circt/Dialect/LTL/LTLFolds.cpp.inc"
} // namespace patterns

//===----------------------------------------------------------------------===//
// AndOp / OrOp / IntersectOp
//===----------------------------------------------------------------------===//

LogicalResult AndOp::canonicalize(AndOp op, PatternRewriter &rewriter) {
  if (op.getType() == rewriter.getI1Type()) {
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, op.getInputs(), true);
    return success();
  }
  return failure();
}

LogicalResult OrOp::canonicalize(OrOp op, PatternRewriter &rewriter) {
  if (op.getType() == rewriter.getI1Type()) {
    rewriter.replaceOpWithNewOp<comb::OrOp>(op, op.getInputs(), true);
    return success();
  }
  return failure();
}

LogicalResult IntersectOp::canonicalize(IntersectOp op,
                                        PatternRewriter &rewriter) {
  if (op.getType() == rewriter.getI1Type()) {
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, op.getInputs(), true);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//

OpFoldResult DelayOp::fold(FoldAdaptor adaptor) {
  // delay(posedge, clock, s, 0, 0) -> s
  if (adaptor.getDelay() == 0 && adaptor.getLength() == 0 &&
      isa<SequenceType>(getInput().getType()))
    return getInput();

  return {};
}

void DelayOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<patterns::NestedDelays>(results.getContext());
  results.add<patterns::MoveDelayIntoConcat>(results.getContext());
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  // concat(s) -> s
  if (getInputs().size() == 1)
    return getInputs()[0];

  return {};
}

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<patterns::FlattenConcats>(results.getContext());
}

//===----------------------------------------------------------------------===//
// RepeatLikeOps
//===----------------------------------------------------------------------===//

namespace {
struct RepeatLikeOp {
  static OpFoldResult fold(uint64_t base, uint64_t more, Value input) {
    // repeat(s, 1, 0) -> s
    if (base == 1 && more == 0 && isa<SequenceType>(input.getType()))
      return input;

    return {};
  }
};
} // namespace

OpFoldResult RepeatOp::fold(FoldAdaptor adaptor) {
  auto more = adaptor.getMore();
  if (more.has_value())
    return RepeatLikeOp::fold(adaptor.getBase(), *more, getInput());
  return {};
}

OpFoldResult GoToRepeatOp::fold(FoldAdaptor adaptor) {
  return RepeatLikeOp::fold(adaptor.getBase(), adaptor.getMore(), getInput());
}

OpFoldResult NonConsecutiveRepeatOp::fold(FoldAdaptor adaptor) {
  return RepeatLikeOp::fold(adaptor.getBase(), adaptor.getMore(), getInput());
}

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

OpFoldResult BooleanConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

OpFoldResult ImplicationOp::fold(FoldAdaptor adaptor) {
  // implication(false, x) -> true (false implies anything)
  if (auto antecedent = dyn_cast_or_null<IntegerAttr>(adaptor.getAntecedent()))
    if (antecedent.getValue().isZero())
      return BoolAttr::get(getContext(), true);

  // implication(x, true) -> true (anything implies true)
  if (auto consequent = dyn_cast_or_null<IntegerAttr>(adaptor.getConsequent()))
    if (consequent.getValue().isOne())
      return BoolAttr::get(getContext(), true);

  return {};
}
