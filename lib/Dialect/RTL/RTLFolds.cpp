//===- RTLOps.cpp - Implement the RTL operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the RTL ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/RTLOps.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace rtl;

static Attribute getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(context, value.getBitWidth()),
                          value);
}

namespace {
struct ConstantIntMatcher {
  APInt &value;
  ConstantIntMatcher(APInt &value) : value(value) {}
  bool match(Operation *op) {
    if (auto cst = dyn_cast<ConstantOp>(op)) {
      value = cst.value();
      return true;
    }
    return false;
  }
};
} // end anonymous namespace

static inline ConstantIntMatcher m_RConstant(APInt &value) {
  return ConstantIntMatcher(value);
}

/// Flattens a single input in `op` if `hasOneUse` is true and it can be defined
/// as an Op. Returns true if successful, and false otherwise.
///
/// Example: op(1, 2, op(3, 4), 5) -> op(1, 2, 3, 4, 5)  // returns true
///
template <typename Op>
static bool tryFlatteningOperands(Op op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();

  for (size_t i = 0, size = inputs.size(); i != size; ++i) {
    // Don't duplicate logic.
    if (!inputs[i].hasOneUse())
      continue;
    auto flattenOp = inputs[i].template getDefiningOp<Op>();
    if (!flattenOp)
      continue;
    auto flattenOpInputs = flattenOp.inputs();

    SmallVector<Value, 4> newOperands;
    newOperands.reserve(size + flattenOpInputs.size());

    auto flattenOpIndex = inputs.begin() + i;
    newOperands.append(inputs.begin(), flattenOpIndex);
    newOperands.append(flattenOpInputs.begin(), flattenOpInputs.end());
    newOperands.append(flattenOpIndex + 1, inputs.end());

    rewriter.replaceOpWithNewOp<Op>(op, op.getType(), newOperands);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "constant has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

OpFoldResult SExtOp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    auto destWidth = getType().getWidth();
    return getIntAttr(input.getValue().sext(destWidth), getContext());
  }

  return {};
}

OpFoldResult AndROp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue().isAllOnesValue()),
                      getContext());

  return {};
}

OpFoldResult OrROp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue() != 0), getContext());

  return {};
}

OpFoldResult XorROp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue().countPopulation() & 1),
                      getContext());

  return {};
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

OpFoldResult ShlOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.shl(b); });
}

OpFoldResult ShrUOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.lshr(b); });
}

OpFoldResult ShrSOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.ashr(b); });
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

OpFoldResult ExtractOp::fold(ArrayRef<Attribute> constants) {
  // If we are extracting the entire input, then return it.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    unsigned dstWidth = getType().getWidth();
    return getIntAttr(input.getValue().lshr(lowBit()).trunc(dstWidth),
                      getContext());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

// Reduce all operands to a single value by applying the `calculate` function.
// This will fail if any of the operands are not constant.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              function_ref<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldVariadicOp(ArrayRef<Attribute> operands,
                                     const CalculationT &calculate) {
  if (!operands.size())
    return {};

  if (!operands[0])
    return {};

  ElementValueT accum = operands[0].cast<AttrElementT>().getValue();
  for (auto i = operands.begin() + 1, end = operands.end(); i != end; ++i) {
    auto attr = *i;
    if (!attr)
      return {};

    auto typedAttr = attr.cast<AttrElementT>();
    calculate(accum, typedAttr.getValue());
  }
  return AttrElementT::get(operands[0].getType(), accum);
}

OpFoldResult AndOp::fold(ArrayRef<Attribute> constants) {
  auto width = getType().getWidth();
  APInt value(/*numBits=*/width, -1, /*isSigned=*/false);

  // and(x, 0, 1) -> 0 -- annulment
  for (auto operand : constants) {
    if (!operand)
      continue;
    value &= operand.cast<IntegerAttr>().getValue();
    if (value.isNullValue())
      return getIntAttr(value, getContext());
  }

  // and(x, x, x) -> x -- noop
  if (llvm::all_of(inputs(), [&](auto in) { return in == inputs()[0]; }))
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a &= b; });
}

void AndOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<AndOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(AndOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // and(..., '1) -> and(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isAllOnesValue()) {
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // TODO: remove all duplicate arguments
      // and(..., x, x) -> and(..., x) -- idempotent
      if (inputs[size - 1] == inputs[size - 2]) {
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // TODO: Combine all constants in one shot
      // and(..., c1, c2) -> and(..., c3) where c3 = c1 & c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value & value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), newOperands);
        return success();
      }
      // and(x, and(...)) -> and(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> constants) {
  auto width = getType().getWidth();
  APInt value(/*numBits=*/width, 0, /*isSigned=*/false);

  // or(x, 1, 0) -> 1
  for (auto operand : constants) {
    if (!operand)
      continue;
    value |= operand.cast<IntegerAttr>().getValue();
    if (value.isAllOnesValue())
      return getIntAttr(value, getContext());
  }

  // or(x, x, x) -> x
  if (llvm::all_of(inputs(), [&](auto in) { return in == inputs()[0]; }))
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a |= b; });
}

void OrOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *context) {
  struct Folder final : public OpRewritePattern<OrOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(OrOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // or(..., 0) -> or(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {

        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
        return success();
      }

      // or(..., x, x) -> or(..., x) -- idempotent
      if (inputs[size - 1] == inputs[size - 2]) {
        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
        return success();
      }

      // or(..., c1, c2) -> or(..., c3) where c3 = c1 | c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value | value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), newOperands);
        return success();
      }

      // or(x, or(...)) -> or(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // xor(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  // xor(x, x) -> 0 -- idempotent
  if (size == 2u && inputs()[0] == inputs()[1])
    return IntegerAttr::get(getType(), 0);

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a ^= b; });
}

void XorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<XorOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(XorOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // xor(..., 0) -> xor(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {

        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      if (inputs[size - 1] == inputs[size - 2]) {
        assert(size > 2 &&
               "expected idempotent case for 2 elements handled already.");
        // xor(..., x, x) -> xor (...) -- idempotent
        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                           inputs.drop_back(/*n=*/2));
        return success();
      }

      // xor(..., c1, c2) -> xor(..., c3) where c3 = c1 ^ c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value ^ value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(), newOperands);
        return success();
      }

      // xor(x, xor(...)) -> xor(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: xor(..., '1) -> not(xor(...))
      /// TODO: xor(..., x, not(x)) -> xor(..., '1)
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult MergeOp::fold(ArrayRef<Attribute> constants) {
  // rtl.merge(x, x, x) -> x.
  if (llvm::all_of(inputs(), [&](auto in) { return in == inputs()[0]; }))
    return inputs()[0];

  return {};
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // add(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a += b; });
}

void AddOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<AddOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(AddOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // add(..., 0) -> add(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // add(..., c1, c2) -> add(..., c3) where c3 = c1 + c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value + value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      // add(..., x, x) -> add(..., shl(x, 1))
      if (inputs[size - 1] == inputs[size - 2]) {
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));

        auto one = rewriter.create<ConstantOp>(op.getLoc(), op.getType(), 1);
        auto shiftLeftOp =
            rewriter.create<rtl::ShlOp>(op.getLoc(), inputs.back(), one);

        newOperands.push_back(shiftLeftOp);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      auto shlOp = inputs[size - 1].getDefiningOp<rtl::ShlOp>();
      // add(..., x, shl(x, c)) -> add(..., mul(x, (1 << c) + 1))
      if (shlOp && shlOp.lhs() == inputs[size - 2] &&
          matchPattern(shlOp.rhs(), m_RConstant(value))) {

        APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
        auto rhs =
            rewriter.create<ConstantOp>(op.getLoc(), (one << value) + one);

        std::array<Value, 2> factors = {shlOp.lhs(), rhs};
        auto mulOp = rewriter.create<rtl::MulOp>(op.getLoc(), factors);

        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(mulOp);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      auto mulOp = inputs[size - 1].getDefiningOp<rtl::MulOp>();
      // add(..., x, mul(x, c)) -> add(..., mul(x, c + 1))
      if (mulOp && mulOp.inputs().size() == 2 &&
          mulOp.inputs()[0] == inputs[size - 2] &&
          matchPattern(mulOp.inputs()[1], m_RConstant(value))) {

        APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
        auto rhs = rewriter.create<ConstantOp>(op.getLoc(), value + one);
        std::array<Value, 2> factors = {mulOp.inputs()[0], rhs};
        auto newMulOp = rewriter.create<rtl::MulOp>(op.getLoc(), factors);

        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(newMulOp);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      // add(x, add(...)) -> add(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // mul(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  auto width = getType().getWidth();
  APInt value(/*numBits=*/width, 1, /*isSigned=*/false);

  // mul(x, 0, 1) -> 0 -- annulment
  for (auto operand : constants) {
    if (!operand)
      continue;
    value *= operand.cast<IntegerAttr>().getValue();
    if (value.isNullValue())
      return getIntAttr(value, getContext());
  }

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a *= b; });
}

void MulOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(MulOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // mul(x, c) -> shl(x, log2(c)), where c is a power of two.
      if (size == 2 && matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isPowerOf2()) {
        auto shift = rewriter.create<ConstantOp>(op.getLoc(), op.getType(),
                                                 value.exactLogBase2());
        auto shlOp = rewriter.create<rtl::ShlOp>(op.getLoc(), inputs[0], shift);

        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                           ArrayRef<Value>(shlOp));
        return success();
      }

      // mul(..., 1) -> mul(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) && (value == 1u)) {
        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // mul(..., c1, c2) -> mul(..., c3) where c3 = c1 * c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value * value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newOperands);
        return success();
      }

      // mul(a, mul(...)) -> mul(a, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      return failure();
    }
  };
  results.insert<Folder>(context);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// Constant folding
OpFoldResult ConcatOp::fold(ArrayRef<Attribute> constants) {
  // If all the operands are constant, we can fold.
  for (auto attr : constants)
    if (!attr || !attr.isa<IntegerAttr>())
      return {};

  // If we got here, we can constant fold.
  unsigned resultWidth = getType().getIntOrFloatBitWidth();
  APInt result(resultWidth, 0);

  unsigned nextInsertion = resultWidth;
  // Insert each chunk into the result.
  for (auto attr : constants) {
    auto chunk = attr.cast<IntegerAttr>().getValue();
    nextInsertion -= chunk.getBitWidth();
    result |= chunk.zext(resultWidth) << nextInsertion;
  }

  return getIntAttr(result, getContext());
}

static LogicalResult tryCanonicalizeConcat(ConcatOp op,
                                           PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // This function is used when we flatten neighboring operands of a (variadic)
  // concat into a new vesion of the concat.  first/last indices are inclusive.
  auto flattenConcat = [&](size_t firstOpIndex, size_t lastOpIndex,
                           ValueRange replacements) -> LogicalResult {
    SmallVector<Value, 4> newOperands;
    newOperands.append(inputs.begin(), inputs.begin() + firstOpIndex);
    newOperands.append(replacements.begin(), replacements.end());
    newOperands.append(inputs.begin() + lastOpIndex + 1, inputs.end());
    if (newOperands.size() == 1)
      rewriter.replaceOp(op, newOperands[0]);
    else
      rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(), newOperands);
    return success();
  };

  for (size_t i = 0; i != size; ++i) {
    // If an operand to the concat is itself a concat, then we can fold them
    // together.
    if (auto subConcat = inputs[i].getDefiningOp<ConcatOp>())
      return flattenConcat(i, i, subConcat->getOperands());

    // We can flatten a sext into the concat, since a sext is an extraction of
    // a sign bit plus the input value itself.
    if (auto sext = inputs[i].getDefiningOp<SExtOp>()) {
      unsigned inputWidth = sext.input().getType().getIntOrFloatBitWidth();

      auto sign = rewriter.create<ExtractOp>(op.getLoc(), rewriter.getI1Type(),
                                             sext.input(), inputWidth - 1);
      SmallVector<Value> replacements;
      unsigned signWidth = sext.getType().getIntOrFloatBitWidth() - inputWidth;
      replacements.append(signWidth, sign);
      replacements.push_back(sext.input());
      return flattenConcat(i, i, replacements);
    }

    // Check for canonicalization due to neighboring operands.
    if (i != 0) {
      // Merge neighboring constants.
      if (auto cst = inputs[i].getDefiningOp<ConstantOp>()) {
        if (auto prevCst = inputs[i - 1].getDefiningOp<ConstantOp>()) {
          unsigned prevWidth = prevCst.getValue().getBitWidth();
          unsigned thisWidth = cst.getValue().getBitWidth();
          auto resultCst = cst.getValue().zext(prevWidth + thisWidth);
          resultCst |= prevCst.getValue().zext(prevWidth + thisWidth)
                       << thisWidth;
          Value replacement =
              rewriter.create<ConstantOp>(op.getLoc(), resultCst);
          return flattenConcat(i - 1, i, replacement);
        }
      }

      // Merge neighboring extracts of neighboring inputs, e.g.
      // {A[3], A[2]} -> A[3,2]
      if (auto extract = inputs[i].getDefiningOp<ExtractOp>()) {
        if (auto prevExtract = inputs[i - 1].getDefiningOp<ExtractOp>()) {
          if (extract.input() == prevExtract.input()) {
            auto thisWidth = extract.getType().getIntOrFloatBitWidth();
            if (prevExtract.lowBit() == extract.lowBit() + thisWidth) {
              auto prevWidth = prevExtract.getType().getIntOrFloatBitWidth();
              auto resType = rewriter.getIntegerType(thisWidth + prevWidth);
              Value replacement = rewriter.create<ExtractOp>(
                  op.getLoc(), resType, extract.input(), extract.lowBit());
              return flattenConcat(i - 1, i, replacement);
            }
          }
        }
      }
    }
  }

  // Return true if this extract is an extract of the sign bit of its input.
  auto isSignBitExtract = [](ExtractOp op) -> bool {
    if (op.getType().getIntOrFloatBitWidth() != 1)
      return false;
    return op.lowBit() == op.input().getType().getIntOrFloatBitWidth() - 1;
  };

  // Folds concat back into a sext.
  if (auto extract = inputs[0].getDefiningOp<ExtractOp>()) {
    if (extract.input() == inputs.back() && isSignBitExtract(extract)) {
      // Check intermediate bits.
      bool allMatch = true;
      // The intermediate bits are allowed to be difference instances of extract
      // (because canonicalize doesn't do CSE automatically) so long as they are
      // getting the sign bit.
      for (size_t i = 1, e = inputs.size() - 1; i != e; ++i) {
        auto extractInner = inputs[i].getDefiningOp<ExtractOp>();
        if (!extractInner || extractInner.input() != inputs.back() ||
            !isSignBitExtract(extractInner)) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        rewriter.replaceOpWithNewOp<SExtOp>(op, op.getType(), inputs.back());
        return success();
      }
    }
  }

  return failure();
}

void ConcatOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  struct Folder final : public OpRewritePattern<ConcatOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(ConcatOp op,
                                  PatternRewriter &rewriter) const override {
      return tryCanonicalizeConcat(op, rewriter);
    }
  };
  results.insert<Folder>(context);
}

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

OpFoldResult MuxOp::fold(ArrayRef<Attribute> constants) {
  // mux (c, b, b) -> b
  if (trueValue() == falseValue())
    return trueValue();

  // mux(0, a, b) -> b
  // mux(1, a, b) -> a
  if (auto pred = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    if (pred.getValue().isNullValue())
      return falseValue();
    return trueValue();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

// Calculate the result of a comparison when the LHS and RHS are both
// constants.
static bool applyCmpPredicate(ICmpPredicate predicate, const APInt &lhs,
                              const APInt &rhs) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return lhs.eq(rhs);
  case ICmpPredicate::ne:
    return lhs.ne(rhs);
  case ICmpPredicate::slt:
    return lhs.slt(rhs);
  case ICmpPredicate::sle:
    return lhs.sle(rhs);
  case ICmpPredicate::sgt:
    return lhs.sgt(rhs);
  case ICmpPredicate::sge:
    return lhs.sge(rhs);
  case ICmpPredicate::ult:
    return lhs.ult(rhs);
  case ICmpPredicate::ule:
    return lhs.ule(rhs);
  case ICmpPredicate::ugt:
    return lhs.ugt(rhs);
  case ICmpPredicate::uge:
    return lhs.uge(rhs);
  }
  llvm_unreachable("unknown comparison predicate");
}

// Returns the result of applying the predicate when the LHS and RHS are the
// exact same value.
static bool applyCmpPredicateToEqualOperands(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
  case ICmpPredicate::sle:
  case ICmpPredicate::sge:
  case ICmpPredicate::ule:
  case ICmpPredicate::uge:
    return true;
  case ICmpPredicate::ne:
  case ICmpPredicate::slt:
  case ICmpPredicate::sgt:
  case ICmpPredicate::ult:
  case ICmpPredicate::ugt:
    return false;
  }
  llvm_unreachable("unknown comparison predicate");
}

OpFoldResult ICmpOp::fold(ArrayRef<Attribute> constants) {

  // gt a, a -> false
  // gte a, a -> true
  if (lhs() == rhs()) {
    auto val = applyCmpPredicateToEqualOperands(predicate());
    return IntegerAttr::get(getType(), val);
  }

  // gt 1, 2 -> false
  if (auto lhs = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = constants[1].dyn_cast_or_null<IntegerAttr>()) {
      auto val = applyCmpPredicate(predicate(), lhs.getValue(), rhs.getValue());
      return IntegerAttr::get(getType(), val);
    }
  }
  return {};
}

namespace {
// Canonicalizes a ICmp with a single constant
struct ICmpCanonicalizeConstant final : public OpRewritePattern<ICmpOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ICmpOp op,
                                PatternRewriter &rewriter) const override {

    APInt lhs, rhs;

    // icmp 1, x -> icmp x, 1
    if (matchPattern(op.lhs(), m_RConstant(lhs))) {
      assert(!matchPattern(op.rhs(), m_RConstant(rhs)) && "Should be folded");
      rewriter.replaceOpWithNewOp<ICmpOp>(
          op, ICmpOp::getFlippedPredicate(op.predicate()), op.rhs(), op.lhs());
      return success();
    }

    // Canonicalize with RHS constant
    if (matchPattern(op.rhs(), m_RConstant(rhs))) {
      ConstantOp constant;

      auto getConstant = [&](APInt constant) -> Value {
        return rewriter.create<ConstantOp>(op.getLoc(), std::move(constant));
      };

      auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
                             Value rhs) -> LogicalResult {
        rewriter.replaceOpWithNewOp<ICmpOp>(op, predicate, lhs, rhs);
        return success();
      };

      auto replaceWithConstantI1 = [&](bool constant) -> LogicalResult {
        rewriter.replaceOpWithNewOp<ConstantOp>(op, APInt(1, constant));
        return success();
      };

      switch (op.predicate()) {
      case ICmpPredicate::slt:
        // x < max -> x != max
        if (rhs.isMaxSignedValue())
          return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
        // x < min -> false
        if (rhs.isMinSignedValue())
          return replaceWithConstantI1(0);
        // x < min+1 -> x == min
        if ((rhs - 1).isMinSignedValue())
          return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs - 1));
        break;
      case ICmpPredicate::sgt:
        // x > min -> x != min
        if (rhs.isMinSignedValue())
          return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
        // x > max -> false
        if (rhs.isMaxSignedValue())
          return replaceWithConstantI1(0);
        // x > max-1 -> x == max
        if ((rhs + 1).isMaxSignedValue())
          return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs + 1));
        break;
      case ICmpPredicate::ult:
        // x < max -> x != max
        if (rhs.isMaxValue())
          return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
        // x < min -> false
        if (rhs.isMinValue())
          return replaceWithConstantI1(0);
        // x < min+1 -> x == min
        if ((rhs - 1).isMinValue())
          return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs - 1));
        break;
      case ICmpPredicate::ugt:
        // x > min -> x != min
        if (rhs.isMinValue())
          return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
        // x > max -> false
        if (rhs.isMaxValue())
          return replaceWithConstantI1(0);
        // x > max-1 -> x == max
        if ((rhs + 1).isMaxValue())
          return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs + 1));
        break;
      case ICmpPredicate::sle:
        // x <= max -> true
        if (rhs.isMaxSignedValue())
          return replaceWithConstantI1(1);
        // x <= c -> x < (c+1)
        return replaceWith(ICmpPredicate::slt, op.lhs(), getConstant(rhs + 1));
      case ICmpPredicate::sge:
        // x >= min -> true
        if (rhs.isMinSignedValue())
          return replaceWithConstantI1(1);
        // x >= c -> x > (c-1)
        return replaceWith(ICmpPredicate::sgt, op.lhs(), getConstant(rhs - 1));
      case ICmpPredicate::ule:
        // x <= max -> true
        if (rhs.isMaxValue())
          return replaceWithConstantI1(1);
        // x <= c -> x < (c+1)
        return replaceWith(ICmpPredicate::ult, op.lhs(), getConstant(rhs + 1));
      case ICmpPredicate::uge:
        // x >= min -> true
        if (rhs.isMinValue())
          return replaceWithConstantI1(1);
        // x >= c -> x > (c-1)
        return replaceWith(ICmpPredicate::ugt, op.lhs(), getConstant(rhs - 1));
      default:
        break;
      }
    }

    return failure();
  }
};

} // namespace

void ICmpOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<ICmpCanonicalizeConstant>(context);
}
