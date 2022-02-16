//===- CombFolds.cpp - Folds + Canonicalization for Comb operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/KnownBits.h"

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace matchers;

/// Create a new instance of a generic operation that only has value operands,
/// and has a single result value whose type matches the first operand.
///
/// This should not be used to create instances of ops with attributes or with
/// more complicated type signatures.
static Value createGenericOp(Location loc, OperationName name,
                             ArrayRef<Value> operands, OpBuilder &builder) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(operands[0].getType());
  return builder.createOperation(state)->getResult(0);
}

static Attribute getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(context, value.getBitWidth()),
                          value);
}

/// Flatten concat and mux operands into a vector.
static void getConcatOperands(Value v, SmallVectorImpl<Value> &result) {
  if (auto concat = v.getDefiningOp<ConcatOp>()) {
    for (auto op : concat.getOperands())
      getConcatOperands(op, result);
  } else if (auto repl = v.getDefiningOp<ReplicateOp>()) {
    for (size_t i = 0, e = repl.getMultiple(); i != e; ++i)
      getConcatOperands(repl.getOperand(), result);
  } else {
    result.push_back(v);
  }
}

namespace {
struct ConstantIntMatcher {
  APInt &value;
  ConstantIntMatcher(APInt &value) : value(value) {}
  bool match(Operation *op) {
    if (auto cst = dyn_cast<hw::ConstantOp>(op)) {
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

namespace {
template <typename SubType>
struct ComplementMatcher {
  SubType lhs;
  ComplementMatcher(SubType lhs) : lhs(std::move(lhs)) {}
  bool match(Operation *op) {
    auto xorOp = dyn_cast<XorOp>(op);
    return xorOp && xorOp.isBinaryNot() && lhs.match(op->getOperand(0));
  }
};
} // end anonymous namespace

template <typename SubType>
static inline ComplementMatcher<SubType> m_Complement(const SubType &subExpr) {
  return ComplementMatcher(subExpr);
}

/// Flattens a single input in `op` if `hasOneUse` is true and it can be defined
/// as an Op. Returns true if successful, and false otherwise.
///
/// Example: op(1, 2, op(3, 4), 5) -> op(1, 2, 3, 4, 5)  // returns true
///
static bool tryFlatteningOperands(Operation *op, PatternRewriter &rewriter) {
  auto inputs = op->getOperands();

  for (size_t i = 0, size = inputs.size(); i != size; ++i) {
    Operation *flattenOp = inputs[i].getDefiningOp();
    if (!flattenOp || flattenOp->getName() != op->getName())
      continue;

    // Don't duplicate logic when it has multiple uses.
    if (!inputs[i].hasOneUse()) {
      // We can fold a multi-use binary operation into this one if this allows a
      // constant to fold though.  For example, fold
      //    (or a, b, c, (or d, cst1), cst2) --> (or a, b, c, d, cst1, cst2)
      // since the constants will both fold and we end up with the equiv cost.
      //
      // We don't do this for add/mul because the hardware won't be shared
      // between the two ops if duplicated.
      if (flattenOp->getNumOperands() != 2 || !isa<AndOp, OrOp, XorOp>(op) ||
          !flattenOp->getOperand(1).getDefiningOp<hw::ConstantOp>() ||
          !inputs.back().getDefiningOp<hw::ConstantOp>())
        continue;
    }

    // Otherwise, flatten away.
    auto flattenOpInputs = flattenOp->getOperands();

    SmallVector<Value, 4> newOperands;
    newOperands.reserve(size + flattenOpInputs.size());

    auto flattenOpIndex = inputs.begin() + i;
    newOperands.append(inputs.begin(), flattenOpIndex);
    newOperands.append(flattenOpInputs.begin(), flattenOpInputs.end());
    newOperands.append(flattenOpIndex + 1, inputs.end());

    Value result =
        createGenericOp(op->getLoc(), op->getName(), newOperands, rewriter);
    if (auto name = op->getAttr("sv.namehint"))
      result.getDefiningOp()->setAttr("sv.namehint", name);
    rewriter.replaceOp(op, result);
    return true;
  }
  return false;
}

// Given a range of uses of an operation, find the lowest and highest bits
// inclusive that are ever referenced. The range of uses must not be empty.
static std::pair<size_t, size_t>
getLowestBitAndHighestBitRequired(Operation *op, bool narrowTrailingBits,
                                  size_t originalOpWidth) {
  auto users = op->getUsers();
  assert(!users.empty() &&
         "getLowestBitAndHighestBitRequired cannot operate on "
         "a empty list of uses.");

  // when we don't want to narrowTrailingBits (namely in arithmetic
  // operations), forcing lowestBitRequired = 0
  size_t lowestBitRequired = narrowTrailingBits ? originalOpWidth - 1 : 0;
  size_t highestBitRequired = 0;

  for (auto *user : users) {
    if (auto extractOp = dyn_cast<ExtractOp>(user)) {
      size_t lowBit = extractOp.lowBit();
      size_t highBit = extractOp.getType().getWidth() + lowBit - 1;
      highestBitRequired = std::max(highestBitRequired, highBit);
      lowestBitRequired = std::min(lowestBitRequired, lowBit);
      continue;
    }

    highestBitRequired = originalOpWidth - 1;
    lowestBitRequired = 0;
    break;
  }

  return {lowestBitRequired, highestBitRequired};
}

// narrowOperationWidth(...) attempts to create a new instance of Op
// with createOp using narrowed operands. Ie: transforming f(x, y) into
// f(x[n:m], y[n:m]), It analyzes all usage sites of narrowingCandidate and
// mutate them to the newly created narrowed version if it's benefitial.
//
// narrowTrailingBits determines whether undemanded trailing bits should
// be aggressively stripped off.
//
// This function requires the narrowingCandidate to have at least 1 use.
//
// Returns true if IR is mutated. IR is mutated iff narrowing happens.
static bool
narrowOperationWidth(Operation *narrowingCandidate, ValueRange inputs,
                     bool narrowTrailingBits, PatternRewriter &rewriter,
                     function_ref<Value(ArrayRef<Value>)> createOp) {
  // If the result is never used, no point optimizing this. It will
  // also complicated error handling in getLowestBitAndHigestBitRequired.
  assert(!narrowingCandidate->getUsers().empty() &&
         "narrowingCandidate must have at least one use.");
  assert(narrowingCandidate->getNumResults() == 1 &&
         "narrowingCandidate must have exactly one result");
  IntegerType narrowingCandidateType =
      narrowingCandidate->getResultTypes().front().cast<IntegerType>();

  size_t highestBitRequired;
  size_t lowestBitRequired;
  size_t originalOpWidth = narrowingCandidateType.getIntOrFloatBitWidth();

  std::tie(lowestBitRequired, highestBitRequired) =
      getLowestBitAndHighestBitRequired(narrowingCandidate, narrowTrailingBits,
                                        originalOpWidth);

  // Give up, because we can't make it narrower than it already is.
  if (lowestBitRequired == 0 && highestBitRequired == originalOpWidth - 1)
    return false;

  auto loc = narrowingCandidate->getLoc();
  size_t narrowedWidth = highestBitRequired - lowestBitRequired + 1;
  auto narrowedType = rewriter.getIntegerType(narrowedWidth);

  // Insert the new narrowedOperation at a point where all narrowingCandidate's
  // operands are available, and resides before all users.
  rewriter.setInsertionPoint(narrowingCandidate);
  Value narrowedOperation =
      createOp(SmallVector<Value>(llvm::map_range(inputs, [&](auto input) {
        return rewriter.create<ExtractOp>(loc, narrowedType, input,
                                          lowestBitRequired);
      })));

  auto getNarrowedExtractReplacement = [&](ExtractOp extractUse) -> Value {
    auto oldLowBit = extractUse.lowBit();

    assert(oldLowBit >= lowestBitRequired &&
           "incorrectly deduced the lowest bit required in usage arguments.");

    auto loc = extractUse.getLoc();

    if (narrowedWidth == extractUse.getType().getWidth()) {
      return narrowedOperation;
    }

    uint32_t newLowBit = oldLowBit - lowestBitRequired;
    Value narrowedExtract = rewriter.create<ExtractOp>(
        loc, extractUse.getType(), narrowedOperation, newLowBit);
    return narrowedExtract;
  };

  // Replaces all existing use sites with the newly created narrowed operation.
  // This loop captures both the root and non-root use sites.  Note that
  // PatternRewriter allows calling rewriter.replaceOp on non-root users.
  for (Operation *user :
       llvm::make_early_inc_range(narrowingCandidate->getUsers())) {
    auto extractUser = cast<ExtractOp>(user);

    // This is necessary because the rewriter's insertion point can be a
    // replaced extractUser
    rewriter.setInsertionPoint(extractUser);
    rewriter.replaceOp(extractUser, getNarrowedExtractReplacement(extractUser));
  }

  // narrowingCandidate can be safely removed now, as all old users of
  // narrowingCandidate are replaced.
  rewriter.eraseOp(narrowingCandidate);

  return true;
}

template <class Op>
static bool narrowOperationWidth(Op op, ValueRange inputs,
                                 bool narrowTrailingBits,
                                 PatternRewriter &rewriter) {
  auto createOp = [&](ArrayRef<Value> args) -> Value {
    return rewriter.create<Op>(op.getLoc(), args);
  };
  return narrowOperationWidth(op, inputs, narrowTrailingBits, rewriter,
                              createOp);
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

OpFoldResult ReplicateOp::fold(ArrayRef<Attribute> constants) {
  // Replicate one time -> noop.
  if (getType().getWidth() == input().getType().getIntOrFloatBitWidth())
    return input();

  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    if (input.getValue().getBitWidth() == 1) {
      if (input.getValue().isZero())
        return getIntAttr(APInt::getZero(getType().getWidth()), getContext());
      return getIntAttr(APInt::getAllOnes(getType().getWidth()), getContext());
    }

    APInt result = APInt::getZeroWidth();
    for (auto i = getMultiple(); i != 0; --i)
      result = result.concat(input.getValue());
    return getIntAttr(result, getContext());
  }

  return {};
}

LogicalResult ReplicateOp::canonicalize(ReplicateOp op,
                                        PatternRewriter &rewriter) {
  // TODO.
  return failure();
}

OpFoldResult ParityOp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue().countPopulation() & 1),
                      getContext());

  return {};
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

/// Performs constant folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
static Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
                                   hw::PEO paramOpcode) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (!operands[0] || !operands[1])
    return {};

  // Fold constants with ParamExprAttr::get which handles simple constants as
  // well as parameter expressions.
  return hw::ParamExprAttr::get(paramOpcode, operands[0], operands[1]);
}

OpFoldResult ShlOp::fold(ArrayRef<Attribute> operands) {
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    unsigned shift = rhs.getValue().getZExtValue();
    unsigned width = getType().getIntOrFloatBitWidth();
    if (shift == 0)
      return getOperand(0);
    if (width <= shift)
      return getIntAttr(APInt::getZero(width), getContext());
  }

  return constFoldBinaryOp(operands, hw::PEO::Shl);
}

LogicalResult ShlOp::canonicalize(ShlOp op, PatternRewriter &rewriter) {
  // ShlOp(x, cst) -> Concat(Extract(x), zeros)
  APInt value;
  if (!matchPattern(op.rhs(), m_RConstant(value)))
    return failure();

  unsigned width = op.lhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  // This case is handled by fold.
  if (width <= shift || shift == 0)
    return failure();

  auto zeros =
      rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(shift));

  // Remove the high bits which would be removed by the Shl.
  auto extract =
      rewriter.create<ExtractOp>(op.getLoc(), op.lhs(), 0, width - shift);

  rewriter.replaceOpWithNewOp<ConcatOp>(op, extract, zeros);
  return success();
}

OpFoldResult ShrUOp::fold(ArrayRef<Attribute> operands) {
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    unsigned shift = rhs.getValue().getZExtValue();
    if (shift == 0)
      return getOperand(0);

    unsigned width = getType().getIntOrFloatBitWidth();
    if (width <= shift)
      return getIntAttr(APInt::getZero(width), getContext());
  }
  return constFoldBinaryOp(operands, hw::PEO::ShrU);
}

LogicalResult ShrUOp::canonicalize(ShrUOp op, PatternRewriter &rewriter) {
  // ShrUOp(x, cst) -> Concat(zeros, Extract(x))
  APInt value;
  if (!matchPattern(op.rhs(), m_RConstant(value)))
    return failure();

  unsigned width = op.lhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  // This case is handled by fold.
  if (width <= shift || shift == 0)
    return failure();

  auto zeros =
      rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(shift));

  // Remove the low bits which would be removed by the Shr.
  auto extract =
      rewriter.create<ExtractOp>(op.getLoc(), op.lhs(), shift, width - shift);

  rewriter.replaceOpWithNewOp<ConcatOp>(op, zeros, extract);
  return success();
}

OpFoldResult ShrSOp::fold(ArrayRef<Attribute> operands) {
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue().getZExtValue() == 0)
      return getOperand(0);
  }
  return constFoldBinaryOp(operands, hw::PEO::ShrS);
}

LogicalResult ShrSOp::canonicalize(ShrSOp op, PatternRewriter &rewriter) {
  // ShrSOp(x, cst) -> Concat(replicate(extract(x, topbit)),extract(x))
  APInt value;
  if (!matchPattern(op.rhs(), m_RConstant(value)))
    return failure();

  unsigned width = op.lhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  auto topbit =
      rewriter.createOrFold<ExtractOp>(op.getLoc(), op.lhs(), width - 1, 1);
  auto sext = rewriter.createOrFold<ReplicateOp>(op.getLoc(), topbit, shift);

  if (width <= shift) {
    rewriter.replaceOp(op, {sext});
    return success();
  }

  auto extract =
      rewriter.create<ExtractOp>(op.getLoc(), op.lhs(), shift, width - shift);

  rewriter.replaceOpWithNewOp<ConcatOp>(op, sext, extract);
  return success();
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

// Transforms extract(lo, cat(a, b, c, d, e)) into
// cat(extract(lo1, b), c, extract(lo2, d)).
// innerCat must be the argument of the provided ExtractOp.
static LogicalResult extractConcatToConcatExtract(ExtractOp op,
                                                  ConcatOp innerCat,
                                                  PatternRewriter &rewriter) {
  auto reversedConcatArgs = llvm::reverse(innerCat.inputs());
  size_t beginOfFirstRelevantElement = 0;
  auto it = reversedConcatArgs.begin();
  size_t lowBit = op.lowBit();

  // This loop finds the first concatArg that is covered by the ExtractOp
  for (; it != reversedConcatArgs.end(); it++) {
    assert(beginOfFirstRelevantElement <= lowBit &&
           "incorrectly moved past an element that lowBit has coverage over");
    auto operand = *it;

    size_t operandWidth = operand.getType().getIntOrFloatBitWidth();
    if (lowBit < beginOfFirstRelevantElement + operandWidth) {
      // A bit other than the first bit will be used in this element.
      // ...... ........ ...
      //           ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      // Edge-case close to the end of the range.
      // ...... ........ ...
      //                 ^---(position + operandWidth)
      //               ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      // Edge-case close to the beginning of the rang
      // ...... ........ ...
      //        ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      break;
    }

    // extraction discards this element.
    // ...... ........  ...
    // |      ^---lowBit
    // ^---beginOfFirstRelevantElement
    beginOfFirstRelevantElement += operandWidth;
  }
  assert(it != reversedConcatArgs.end() &&
         "incorrectly failed to find an element which contains coverage of "
         "lowBit");

  SmallVector<Value> reverseConcatArgs;
  size_t widthRemaining = op.getType().getWidth();
  size_t extractLo = lowBit - beginOfFirstRelevantElement;

  // Transform individual arguments of innerCat(..., a, b, c,) into
  // [ extract(a), b, extract(c) ], skipping an extract operation where
  // possible.
  for (; widthRemaining != 0 && it != reversedConcatArgs.end(); it++) {
    auto concatArg = *it;
    size_t operandWidth = concatArg.getType().getIntOrFloatBitWidth();
    size_t widthToConsume = std::min(widthRemaining, operandWidth - extractLo);

    if (widthToConsume == operandWidth && extractLo == 0) {
      reverseConcatArgs.push_back(concatArg);
    } else {
      auto resultType = IntegerType::get(rewriter.getContext(), widthToConsume);
      reverseConcatArgs.push_back(
          rewriter.create<ExtractOp>(op.getLoc(), resultType, *it, extractLo));
    }

    widthRemaining -= widthToConsume;

    // Beyond the first element, all elements are extracted from position 0.
    extractLo = 0;
  }

  if (reverseConcatArgs.size() == 1) {
    rewriter.replaceOp(op, reverseConcatArgs[0]);
  } else {
    rewriter.replaceOpWithNewOp<ConcatOp>(
        op, SmallVector<Value>(llvm::reverse(reverseConcatArgs)));
  }
  return success();
}

// Transforms extract(lo, replicate(a, N)) into replicate(a, N-c).
static bool extractFromReplicate(ExtractOp op, ReplicateOp replicate,
                                 PatternRewriter &rewriter) {
  auto extractResultWidth = op.getType().getWidth();
  auto replicateEltWidth =
      replicate.getOperand().getType().getIntOrFloatBitWidth();

  // If the extract starts at the base of an element and is an even multiple,
  // we can replace the extract with a smaller replicate.
  if (op.lowBit() % replicateEltWidth == 0 &&
      extractResultWidth % replicateEltWidth == 0) {
    rewriter.replaceOpWithNewOp<ReplicateOp>(op, op.getType(),
                                             replicate.getOperand());
    return true;
  }

  // If the extract is completely contained in one element, extract from the
  // element.
  if (op.lowBit() % replicateEltWidth + extractResultWidth <=
      replicateEltWidth) {
    rewriter.replaceOpWithNewOp<ExtractOp>(op, op.getType(),
                                           replicate.getOperand(),
                                           op.lowBit() % replicateEltWidth);
    return true;
  }

  // We don't currently handle the case of extracting from non-whole elements,
  // e.g. `extract (replicate 2-bit-thing, N), 1`.
  return false;
}

// Pattern matches on extract(f(a, b)), and transforms f(extract(a), extract(b))
// for some known f. This is performed by analyzing all usage sites of the
// result of f(a, b). When this transformation returns true, it mutates all the
// usage sites of outerExtractOp to reference the newly created narrowed
// operation.
static bool narrowExtractWidth(ExtractOp outerExtractOp,
                               PatternRewriter &rewriter) {
  auto *innerArg = outerExtractOp.input().getDefiningOp();
  if (!innerArg)
    return false;

  // In calls to narrowOperationWidth below, innerOp is guranteed to have at
  // least one use (ie: this extract operation). So we don't need to handle
  // innerOp with no uses.
  return llvm::TypeSwitch<Operation *, bool>(innerArg)
      // The unreferenced leading bits of Add, Sub and Mul can be stripped of,
      // but not the trailing bits. The trailing bits is used to compute the
      // results of arithmetic operations.
      .Case<AddOp, MulOp>([&](auto innerOp) {
        return narrowOperationWidth(innerOp, innerOp.inputs(),
                                    /* narrowTrailingBits= */ false, rewriter);
      })
      .Case<SubOp>([&](SubOp innerOp) {
        return narrowOperationWidth(innerOp, {innerOp.lhs(), innerOp.rhs()},
                                    /* narrowTrailingBits= */ false, rewriter);
      })
      // Bit-wise operations and muxes can be narrowed more aggressively.
      // Trailing bits that are not referenced in the use-sites can all be
      // removed.
      .Case<AndOp, OrOp, XorOp>([&](auto innerOp) {
        return narrowOperationWidth(innerOp, innerOp.inputs(),
                                    /* narrowTrailingBits= */ true, rewriter);
      })
      .Case<MuxOp>([&](MuxOp innerOp) {
        Type type = innerOp.getType();

        assert(type.isa<IntegerType>() &&
               "extract() requires input to be of type IntegerType!");

        auto cond = innerOp.cond();
        auto loc = innerOp.getLoc();
        auto createMuxOp = [&](ArrayRef<Value> values) -> MuxOp {
          assert(values.size() == 2 &&
                 "createMuxOp expects exactly two elements");
          return rewriter.create<MuxOp>(loc, cond, values[0], values[1]);
        };

        return narrowOperationWidth(
            innerOp, {innerOp.trueValue(), innerOp.falseValue()},
            /* narrowTrailingBits= */ true, rewriter, createMuxOp);
      })

      // TODO: Cautiously investigate whether this optimization can be performed
      // on other comb operators (shifts & divs), arrays, memory, or even
      // sequential operations.
      .Default([](Operation *op) { return false; });
}

LogicalResult ExtractOp::canonicalize(ExtractOp op, PatternRewriter &rewriter) {
  auto *inputOp = op.input().getDefiningOp();

  // If the extracted bits are all known, then return the result.
  auto knownBits = computeKnownBits(op.input())
                       .extractBits(op.getType().getWidth(), op.lowBit());
  if (knownBits.isConstant()) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, knownBits.getConstant());
    return success();
  }

  // extract(olo, extract(ilo, x)) = extract(olo + ilo, x)
  if (auto innerExtract = dyn_cast_or_null<ExtractOp>(inputOp)) {
    rewriter.replaceOpWithNewOp<ExtractOp>(op, op.getType(),
                                           innerExtract.input(),
                                           innerExtract.lowBit() + op.lowBit());
    return success();
  }

  // extract(lo, cat(a, b, c, d, e)) = cat(extract(lo1, b), c, extract(lo2, d))
  if (auto innerCat = dyn_cast_or_null<ConcatOp>(inputOp))
    return extractConcatToConcatExtract(op, innerCat, rewriter);

  // extract(lo, replicate(a))
  if (auto replicate = dyn_cast_or_null<ReplicateOp>(inputOp))
    if (extractFromReplicate(op, replicate, rewriter))
      return success();

  // extract(f(a, b)) = f(extract(a), extract(b)). This is performed only when
  // the number of bits to operation f can be reduced. See documentation of
  // narrowExtractWidth for more information.
  if (narrowExtractWidth(op, rewriter))
    return success();

  // `extract(and(a, cst))` -> `extract(a)` when the relevant bits of the
  // and/or/xor are not modifying the extracted bits.
  if (inputOp && inputOp->getNumOperands() == 2 &&
      isa<AndOp, OrOp, XorOp>(inputOp)) {
    if (auto cstRHS = inputOp->getOperand(1).getDefiningOp<hw::ConstantOp>()) {
      auto extractedCst =
          cstRHS.getValue().extractBits(op.getType().getWidth(), op.lowBit());
      if (isa<OrOp, XorOp>(inputOp) && extractedCst.isZero()) {
        rewriter.replaceOpWithNewOp<ExtractOp>(
            op, op.getType(), inputOp->getOperand(0), op.lowBit());
        return success();
      }

      // `extract(and(a, cst))` -> `concat(extract(a), 0)` if we only need one
      // extract to represent the result.  Turning it into a pile of extracts is
      // always fine by our cost model, but we don't want to explode things into
      // a ton of bits because it will bloat the IR and generated Verilog.
      if (isa<AndOp>(inputOp)) {
        // For our cost model, we only do this if the bit pattern is a
        // contiguous series of ones.
        unsigned lz = extractedCst.countLeadingZeros();
        unsigned tz = extractedCst.countTrailingZeros();
        unsigned pop = extractedCst.countPopulation();
        if (extractedCst.getBitWidth() - lz - tz == pop) {
          auto resultTy = rewriter.getIntegerType(pop);
          SmallVector<Value> resultElts;
          if (lz)
            resultElts.push_back(rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(lz)));
          resultElts.push_back(rewriter.createOrFold<ExtractOp>(
              op.getLoc(), resultTy, inputOp->getOperand(0), op.lowBit() + tz));
          if (tz)
            resultElts.push_back(rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(tz)));
          rewriter.replaceOpWithNewOp<ConcatOp>(op, resultElts);
          return success();
        }
      }
    }
  }

  // `extract(lowBit, shl(1, x))` -> `x == lowBit` when a single bit is
  // extracted.
  if (op.getType().getWidth() == 1 && inputOp)
    if (auto shlOp = dyn_cast<ShlOp>(inputOp))
      if (auto lhsCst = shlOp.getOperand(0).getDefiningOp<hw::ConstantOp>())
        if (lhsCst.getValue().isOne()) {
          auto newCst = rewriter.create<hw::ConstantOp>(
              shlOp.getLoc(),
              APInt(lhsCst.getValue().getBitWidth(), op.lowBit()));
          rewriter.replaceOpWithNewOp<ICmpOp>(op, ICmpPredicate::eq,
                                              shlOp->getOperand(1), newCst);
          return success();
        }

  return failure();
}

//===----------------------------------------------------------------------===//
// Associative Variadic operations
//===----------------------------------------------------------------------===//

// Reduce all operands to a single value (either integer constant or parameter
// expression) if all the operands are constants.
static Attribute constFoldAssociativeOp(ArrayRef<Attribute> operands,
                                        hw::PEO paramOpcode) {
  assert(operands.size() > 1 && "caller should handle one-operand case");
  // We can only fold anything in the case where all operands are known to be
  // constants.  Check the least common one first for an early out.
  if (!operands[1] || !operands[0])
    return {};

  // This will fold to a simple constant if all operands are constant.
  if (llvm::all_of(operands.drop_front(2), [&](Attribute in) { return !!in; }))
    return hw::ParamExprAttr::get(paramOpcode, operands);

  return {};
}

/// When we find a logical operation (and, or, xor) with a constant e.g.
/// `X & 42`, we want to push the constant into the computation of X if it leads
/// to simplification.
///
/// This function handles the case where the logical operation has a concat
/// operand.  We check to see if we can simplify the concat, e.g. when it has
/// constant operands.
///
/// This returns true when a simplification happens.
static bool canonicalizeLogicalCstWithConcat(Operation *logicalOp,
                                             size_t concatIdx, const APInt &cst,
                                             PatternRewriter &rewriter) {
  auto concatOp = logicalOp->getOperand(concatIdx).getDefiningOp<ConcatOp>();
  assert((isa<AndOp, OrOp, XorOp>(logicalOp) && concatOp));

  // Check to see if any operands can be simplified by pushing the logical op
  // into all parts of the concat.
  bool canSimplify =
      llvm::any_of(concatOp->getOperands(), [&](Value operand) -> bool {
        auto *operandOp = operand.getDefiningOp();
        if (!operandOp)
          return false;

        // If the concat has a constant operand then we can transform this.
        if (isa<hw::ConstantOp>(operandOp))
          return true;
        // If the concat has the same logical operation and that operation has
        // a constant operation than we can fold it into that suboperation.
        return operandOp->getName() == logicalOp->getName() &&
               operandOp->hasOneUse() && operandOp->getNumOperands() != 0 &&
               operandOp->getOperands().back().getDefiningOp<hw::ConstantOp>();
      });

  if (!canSimplify)
    return false;

  // Create a new instance of the logical operation.  We have to do this the
  // hard way since we're generic across a family of different ops.
  auto createLogicalOp = [&](ArrayRef<Value> operands) -> Value {
    return createGenericOp(logicalOp->getLoc(), logicalOp->getName(), operands,
                           rewriter);
  };

  // Ok, let's do the transformation.  We do this by slicing up the constant
  // for each unit of the concat and duplicate the operation into the
  // sub-operand.
  SmallVector<Value> newConcatOperands;
  newConcatOperands.reserve(concatOp->getNumOperands());

  // Work from MSB to LSB.
  size_t nextOperandBit = concatOp.getType().getIntOrFloatBitWidth();
  for (Value operand : concatOp->getOperands()) {
    size_t operandWidth = operand.getType().getIntOrFloatBitWidth();
    nextOperandBit -= operandWidth;
    // Take a slice of the constant.
    auto eltCst = rewriter.create<hw::ConstantOp>(
        logicalOp->getLoc(), cst.lshr(nextOperandBit).trunc(operandWidth));

    newConcatOperands.push_back(createLogicalOp({operand, eltCst}));
  }

  // Create the concat, and the rest of the logical op if we need it.
  Value newResult =
      rewriter.create<ConcatOp>(concatOp.getLoc(), newConcatOperands);

  // If we had a variadic logical op on the top level, then recreate it with the
  // new concat and without the constant operand.
  if (logicalOp->getNumOperands() > 2) {
    auto origOperands = logicalOp->getOperands();
    SmallVector<Value> operands;
    // Take any stuff before the concat.
    operands.append(origOperands.begin(), origOperands.begin() + concatIdx);
    // Take any stuff after the concat but before the constant.
    operands.append(origOperands.begin() + concatIdx + 1,
                    origOperands.begin() + (origOperands.size() - 1));
    // Include the new concat.
    operands.push_back(newResult);
    newResult = createLogicalOp(operands);
  }

  rewriter.replaceOp(logicalOp, newResult);
  return true;
}

OpFoldResult AndOp::fold(ArrayRef<Attribute> constants) {
  APInt value = APInt::getAllOnes(getType().getWidth());

  // and(x, 01, 10) -> 00 -- annulment.
  for (auto operand : constants) {
    if (!operand)
      continue;
    value &= operand.cast<IntegerAttr>().getValue();
    if (value.isZero())
      return getIntAttr(value, getContext());
  }

  // and(x, -1) -> x.
  if (constants.size() == 2 && constants[1] &&
      constants[1].cast<IntegerAttr>().getValue().isAllOnes())
    return inputs()[0];

  // and(x, x, x) -> x.  This also handles and(x) -> x.
  if (llvm::all_of(inputs(), [&](auto in) { return in == this->inputs()[0]; }))
    return inputs()[0];

  // and(..., x, ..., ~x, ...) -> 0
  for (Value arg : inputs()) {
    Value subExpr;
    if (matchPattern(arg, m_Complement(m_Any(&subExpr)))) {
      for (Value arg2 : inputs())
        if (arg2 == subExpr)
          return getIntAttr(APInt::getZero(getType().getWidth()), getContext());
    }
  }

  // Constant fold
  return constFoldAssociativeOp(constants, hw::PEO::And);
}

LogicalResult AndOp::canonicalize(AndOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands, `fold` should handle this");

  // and(..., x, ..., x) -> and(..., x, ...) -- idempotent
  // Trivial and(x), and(x, x) cases are handled by [AndOp::fold] above.
  if (inputs.size() > 2) {
    llvm::DenseSet<mlir::Value> dedupedArguments;
    SmallVector<Value, 4> newOperands;

    for (const auto input : inputs) {
      auto insertionResult = dedupedArguments.insert(input);
      if (insertionResult.second) {
        newOperands.push_back(input);
      }
    }

    if (newOperands.size() < inputs.size()) {
      rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), newOperands);
      return success();
    }
  }

  // Patterns for and with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_RConstant(value))) {
    // and(..., '1) -> and(...) -- identity
    if (value.isAllOnes()) {
      rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), inputs.drop_back());
      return success();
    }

    // TODO: Combine multiple constants together even if they aren't at the
    // end. and(..., c1, c2) -> and(..., c3) where c3 = c1 & c2 -- constant
    // folding
    APInt value2;
    if (matchPattern(inputs[size - 2], m_RConstant(value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value & value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), newOperands);
      return success();
    }

    // Handle 'and' with a single bit constant on the RHS.
    if (size == 2 && value.isPowerOf2()) {
      // If the LHS is a replicate from a single bit, we can 'concat' it
      // into place.  e.g.:
      //   `replicate(x) & 4` -> `concat(zeros, x, zeros)`
      // TODO: Generalize this for non-single-bit operands.
      if (auto replicate = inputs[0].getDefiningOp<ReplicateOp>()) {
        auto replicateOperand = replicate.getOperand();
        if (replicateOperand.getType().isInteger(1)) {
          unsigned resultWidth = op.getType().getIntOrFloatBitWidth();
          auto trailingZeros = value.countTrailingZeros();

          // Don't add zero bit constants unnecessarily.
          SmallVector<Value, 3> concatOperands;
          if (trailingZeros != resultWidth - 1) {
            auto highZeros = rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(resultWidth - trailingZeros - 1));
            concatOperands.push_back(highZeros);
          }
          concatOperands.push_back(replicateOperand);
          if (trailingZeros != 0) {
            auto lowZeros = rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(trailingZeros));
            concatOperands.push_back(lowZeros);
          }
          rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(),
                                                concatOperands);
          return success();
        }
      }
    }

    // If this is an and from an extract op, try shrinking the extract.
    if (auto extractOp = inputs[0].getDefiningOp<ExtractOp>()) {
      if (size == 2 &&
          // We can shrink it if the mask has leading or trailing zeros.
          (value.countLeadingZeros() || value.countTrailingZeros())) {
        unsigned lz = value.countLeadingZeros();
        unsigned tz = value.countTrailingZeros();

        // Start by extracting the smaller number of bits.
        auto smallTy = rewriter.getIntegerType(value.getBitWidth() - lz - tz);
        Value smallElt = rewriter.createOrFold<ExtractOp>(
            extractOp.getLoc(), smallTy, extractOp->getOperand(0),
            extractOp.lowBit() + tz);
        // Apply the 'and' mask if needed.
        APInt smallMask = value.extractBits(smallTy.getWidth(), tz);
        if (!smallMask.isAllOnes()) {
          auto loc = inputs.back().getLoc();
          smallElt = rewriter.createOrFold<AndOp>(
              loc, smallElt, rewriter.create<hw::ConstantOp>(loc, smallMask));
        }

        // The final replacement will be a concat of the leading/trailing zeros
        // along with the smaller extracted value.
        SmallVector<Value> resultElts;
        if (lz)
          resultElts.push_back(
              rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(lz)));
        resultElts.push_back(smallElt);
        if (tz)
          resultElts.push_back(
              rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(tz)));
        rewriter.replaceOpWithNewOp<ConcatOp>(op, resultElts);
        return success();
      }
    }

    // and(concat(x, cst1), a, b, c, cst2)
    //    ==> and(a, b, c, concat(and(x,cst2'), and(cst1,cst2'')).
    // We do this for even more multi-use concats since they are "just wiring".
    for (size_t i = 0; i < size - 1; ++i) {
      if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();
    }
  }

  // and(x, and(...)) -> and(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement
  return failure();
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> constants) {
  auto value = APInt::getZero(getType().getWidth());

  // or(x, 10, 01) -> 11
  for (auto operand : constants) {
    if (!operand)
      continue;
    value |= operand.cast<IntegerAttr>().getValue();
    if (value.isAllOnes())
      return getIntAttr(value, getContext());
  }

  // or(x, 0) -> x
  if (constants.size() == 2 && constants[1] &&
      constants[1].cast<IntegerAttr>().getValue().isZero())
    return inputs()[0];

  // or(x, x, x) -> x.  This also handles or(x) -> x
  if (llvm::all_of(inputs(), [&](auto in) { return in == this->inputs()[0]; }))
    return inputs()[0];

  // or(..., x, ..., ~x, ...) -> -1
  for (Value arg : inputs()) {
    Value subExpr;
    if (matchPattern(arg, m_Complement(m_Any(&subExpr)))) {
      for (Value arg2 : inputs())
        if (arg2 == subExpr)
          return getIntAttr(APInt::getAllOnes(getType().getWidth()),
                            getContext());
    }
  }

  // Constant fold
  return constFoldAssociativeOp(constants, hw::PEO::Or);
}

/// Simplify concat ops in an or op when a constant operand is present in either
/// concat.
///
/// This will invert an or(concat, concat) into concat(or, or, ...), which can
/// often be further simplified due to the smaller or ops being easier to fold.
///
/// For example:
///
/// or(..., concat(x, 0), concat(0, y))
///    ==> or(..., concat(x, 0, y)), when x and y don't overlap.
///
/// or(..., concat(x: i2, cst1: i4), concat(cst2: i5, y: i1))
///    ==> or(..., concat(or(x: i2,               extract(cst2, 4..3)),
///                       or(extract(cst1, 3..1), extract(cst2, 2..0)),
///                       or(extract(cst1, 0..0), y: i1))
static bool canonicalizeOrOfConcatsWithCstOperands(OrOp op, size_t concatIdx1,
                                                   size_t concatIdx2,
                                                   PatternRewriter &rewriter) {
  assert(concatIdx1 < concatIdx2 && "concatIdx1 must be < concatIdx2");

  auto inputs = op.inputs();
  auto concat1 = inputs[concatIdx1].getDefiningOp<ConcatOp>();
  auto concat2 = inputs[concatIdx2].getDefiningOp<ConcatOp>();

  assert(concat1 && concat2 && "expected indexes to point to ConcatOps");

  // We can simplify as long as a constant is present in either concat.
  bool hasConstantOp1 =
      llvm::any_of(concat1->getOperands(), [&](Value operand) -> bool {
        return operand.getDefiningOp<hw::ConstantOp>();
      });
  if (!hasConstantOp1) {
    bool hasConstantOp2 =
        llvm::any_of(concat2->getOperands(), [&](Value operand) -> bool {
          return operand.getDefiningOp<hw::ConstantOp>();
        });
    if (!hasConstantOp2)
      return false;
  }

  SmallVector<Value> newConcatOperands;

  // Simultaneously iterate over the operands of both concat ops, from MSB to
  // LSB, pushing out or's of overlapping ranges of the operands. When operands
  // span different bit ranges, we extract only the maximum overlap.
  auto operands1 = concat1->getOperands();
  auto operands2 = concat2->getOperands();
  // Number of bits already consumed from operands 1 and 2, respectively.
  unsigned consumedWidth1 = 0;
  unsigned consumedWidth2 = 0;
  for (auto it1 = operands1.begin(), end1 = operands1.end(),
            it2 = operands2.begin(), end2 = operands2.end();
       it1 != end1 && it2 != end2;) {
    auto operand1 = *it1;
    auto operand2 = *it2;

    unsigned remainingWidth1 =
        hw::getBitWidth(operand1.getType()) - consumedWidth1;
    unsigned remainingWidth2 =
        hw::getBitWidth(operand2.getType()) - consumedWidth2;
    unsigned widthToConsume = std::min(remainingWidth1, remainingWidth2);
    auto narrowedType = rewriter.getIntegerType(widthToConsume);

    auto extract1 = rewriter.createOrFold<ExtractOp>(
        op.getLoc(), narrowedType, operand1, remainingWidth1 - widthToConsume);
    auto extract2 = rewriter.createOrFold<ExtractOp>(
        op.getLoc(), narrowedType, operand2, remainingWidth2 - widthToConsume);

    newConcatOperands.push_back(
        rewriter.createOrFold<OrOp>(op.getLoc(), extract1, extract2));

    consumedWidth1 += widthToConsume;
    consumedWidth2 += widthToConsume;

    if (widthToConsume == remainingWidth1) {
      ++it1;
      consumedWidth1 = 0;
    }
    if (widthToConsume == remainingWidth2) {
      ++it2;
      consumedWidth2 = 0;
    }
  }

  ConcatOp newOp = rewriter.create<ConcatOp>(op.getLoc(), newConcatOperands);

  // Copy the old operands except for concatIdx1 and concatIdx2, and append the
  // new ConcatOp to the end.
  SmallVector<Value> newOrOperands;
  newOrOperands.append(inputs.begin(), inputs.begin() + concatIdx1);
  newOrOperands.append(inputs.begin() + concatIdx1 + 1,
                       inputs.begin() + concatIdx2);
  newOrOperands.append(inputs.begin() + concatIdx2 + 1,
                       inputs.begin() + inputs.size());
  newOrOperands.push_back(newOp);

  rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), newOrOperands);
  return true;
}

LogicalResult OrOp::canonicalize(OrOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // or(..., x, x) -> or(..., x) -- idempotent
  if (inputs[size - 1] == inputs[size - 2]) {
    rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
    return success();
  }

  // Patterns for and with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_RConstant(value))) {
    // or(..., '0) -> or(...) -- identity
    if (value.isZero()) {
      rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
      return success();
    }

    // or(..., c1, c2) -> or(..., c3) where c3 = c1 | c2 -- constant folding
    APInt value2;
    if (matchPattern(inputs[size - 2], m_RConstant(value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value | value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), newOperands);
      return success();
    }

    // or(concat(x, cst1), a, b, c, cst2)
    //    ==> or(a, b, c, concat(or(x,cst2'), or(cst1,cst2'')).
    // We do this for even more multi-use concats since they are "just wiring".
    for (size_t i = 0; i < size - 1; ++i) {
      if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();
    }
  }

  // or(x, or(...)) -> or(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  // or(..., concat(x, cst1), concat(cst2, y)
  //    ==> or(..., concat(x, cst3, y)), when x and y don't overlap.
  for (size_t i = 0; i < size - 1; ++i) {
    if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
      for (size_t j = i + 1; j < size; ++j)
        if (auto concat = inputs[j].getDefiningOp<ConcatOp>())
          if (canonicalizeOrOfConcatsWithCstOperands(op, i, j, rewriter))
            return success();
  }

  /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement
  return failure();
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // xor(x) -> x -- noop
  if (size == 1)
    return inputs()[0];

  // xor(x, x) -> 0 -- idempotent
  if (size == 2 && inputs()[0] == inputs()[1])
    return IntegerAttr::get(getType(), 0);

  // xor(x, 0) -> x
  if (constants.size() == 2 && constants[1] &&
      constants[1].cast<IntegerAttr>().getValue().isZero())
    return inputs()[0];

  // xor(xor(x,1),1) -> x
  if (isBinaryNot()) {
    Value subExpr;
    if (matchPattern(getOperand(0), m_Complement(m_Any(&subExpr))))
      return subExpr;
  }

  // Constant fold
  return constFoldAssociativeOp(constants, hw::PEO::Xor);
}

// xor(icmp, a, b, 1) -> xor(icmp, a, b) if icmp has one user.
static void canonicalizeXorIcmpTrue(XorOp op, unsigned icmpOperand,
                                    PatternRewriter &rewriter) {
  auto icmp = op.getOperand(icmpOperand).getDefiningOp<ICmpOp>();
  auto negatedPred = ICmpOp::getNegatedPredicate(icmp.predicate());

  Value result = rewriter.create<ICmpOp>(
      icmp.getLoc(), negatedPred, icmp.getOperand(0), icmp.getOperand(1));

  // If the xor had other operands, rebuild it.
  if (op.getNumOperands() > 2) {
    SmallVector<Value, 4> newOperands(op.getOperands());
    newOperands.pop_back();
    newOperands.erase(newOperands.begin() + icmpOperand);
    newOperands.push_back(result);
    result = rewriter.create<XorOp>(op.getLoc(), newOperands);
  }

  rewriter.replaceOp(op, result);
}

LogicalResult XorOp::canonicalize(XorOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // xor(..., x, x) -> xor (...) -- idempotent
  if (inputs[size - 1] == inputs[size - 2]) {
    assert(size > 2 &&
           "expected idempotent case for 2 elements handled already.");
    rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                       inputs.drop_back(/*n=*/2));
    return success();
  }

  // Patterns for xor with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_RConstant(value))) {
    // xor(..., 0) -> xor(...) -- identity
    if (value.isZero()) {
      rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(), inputs.drop_back());
      return success();
    }

    // xor(..., c1, c2) -> xor(..., c3) where c3 = c1 ^ c2.
    APInt value2;
    if (matchPattern(inputs[size - 2], m_RConstant(value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value ^ value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(), newOperands);
      return success();
    }

    bool isSingleBit = value.getBitWidth() == 1;

    // Check for subexpressions that we can simplify.
    for (size_t i = 0; i < size - 1; ++i) {
      Value operand = inputs[i];

      // xor(concat(x, cst1), a, b, c, cst2)
      //    ==> xor(a, b, c, concat(xor(x,cst2'), xor(cst1,cst2'')).
      // We do this for even more multi-use concats since they are "just
      // wiring".
      if (auto concat = operand.getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();

      // xor(icmp, a, b, 1) -> xor(icmp, a, b) if icmp has one user.
      if (isSingleBit && operand.hasOneUse()) {
        assert(value == 1 && "single bit constant has to be one if not zero");
        if (auto icmp = operand.getDefiningOp<ICmpOp>())
          return canonicalizeXorIcmpTrue(op, i, rewriter), success();
      }
    }
  }

  // xor(x, xor(...)) -> xor(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  return failure();
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> constants) {
  // sub(x - x) -> 0
  if (rhs() == lhs())
    return getIntAttr(APInt::getZero(lhs().getType().getIntOrFloatBitWidth()),
                      getContext());

  if (constants[1]) {
    // If both are constants, we can unconditionally fold.
    if (constants[0]) {
      // Constant fold (c1 - c2) => (c1 + -1*c2).
      auto negOne =
          getIntAttr(APInt::getAllOnes(lhs().getType().getIntOrFloatBitWidth()),
                     getContext());
      auto rhsNeg = hw::ParamExprAttr::get(hw::PEO::Mul, constants[1], negOne);
      return hw::ParamExprAttr::get(hw::PEO::Add, constants[0], rhsNeg);
    }

    // sub(x - 0) -> x
    if (auto rhsC = constants[1].dyn_cast<IntegerAttr>()) {
      if (rhsC.getValue().isZero())
        return lhs();
    }
  }

  return {};
}

LogicalResult SubOp::canonicalize(SubOp op, PatternRewriter &rewriter) {
  // sub(x, cst) -> add(x, -cst)
  APInt value;
  if (matchPattern(op.rhs(), m_RConstant(value))) {
    auto negCst = rewriter.create<hw::ConstantOp>(op.getLoc(), -value);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.lhs(), negCst);
    return success();
  }

  return failure();
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // add(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  // Constant fold constant operands.
  return constFoldAssociativeOp(constants, hw::PEO::Add);
}

LogicalResult AddOp::canonicalize(AddOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  APInt value, value2;

  // add(..., 0) -> add(...) -- identity
  if (matchPattern(inputs.back(), m_RConstant(value)) && value.isZero()) {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), inputs.drop_back());
    return success();
  }

  // add(..., c1, c2) -> add(..., c3) where c3 = c1 + c2 -- constant folding
  if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
      matchPattern(inputs[size - 2], m_RConstant(value2))) {
    auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value + value2);
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(cst);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
    return success();
  }

  // add(..., x, x) -> add(..., shl(x, 1))
  if (inputs[size - 1] == inputs[size - 2]) {
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));

    auto one = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(), 1);
    auto shiftLeftOp =
        rewriter.create<comb::ShlOp>(op.getLoc(), inputs.back(), one);

    newOperands.push_back(shiftLeftOp);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
    return success();
  }

  auto shlOp = inputs[size - 1].getDefiningOp<comb::ShlOp>();
  // add(..., x, shl(x, c)) -> add(..., mul(x, (1 << c) + 1))
  if (shlOp && shlOp.lhs() == inputs[size - 2] &&
      matchPattern(shlOp.rhs(), m_RConstant(value))) {

    APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
    auto rhs =
        rewriter.create<hw::ConstantOp>(op.getLoc(), (one << value) + one);

    std::array<Value, 2> factors = {shlOp.lhs(), rhs};
    auto mulOp = rewriter.create<comb::MulOp>(op.getLoc(), factors);

    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(mulOp);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
    return success();
  }

  auto mulOp = inputs[size - 1].getDefiningOp<comb::MulOp>();
  // add(..., x, mul(x, c)) -> add(..., mul(x, c + 1))
  if (mulOp && mulOp.inputs().size() == 2 &&
      mulOp.inputs()[0] == inputs[size - 2] &&
      matchPattern(mulOp.inputs()[1], m_RConstant(value))) {

    APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
    auto rhs = rewriter.create<hw::ConstantOp>(op.getLoc(), value + one);
    std::array<Value, 2> factors = {mulOp.inputs()[0], rhs};
    auto newMulOp = rewriter.create<comb::MulOp>(op.getLoc(), factors);

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
    if (value.isZero())
      return getIntAttr(value, getContext());
  }

  // Constant fold
  return constFoldAssociativeOp(constants, hw::PEO::Mul);
}

LogicalResult MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  APInt value, value2;

  // mul(x, c) -> shl(x, log2(c)), where c is a power of two.
  if (size == 2 && matchPattern(inputs.back(), m_RConstant(value)) &&
      value.isPowerOf2()) {
    auto shift = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(),
                                                 value.exactLogBase2());
    auto shlOp = rewriter.create<comb::ShlOp>(op.getLoc(), inputs[0], shift);

    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                       ArrayRef<Value>(shlOp));
    return success();
  }

  // mul(..., 1) -> mul(...) -- identity
  if (matchPattern(inputs.back(), m_RConstant(value)) && value.isOne()) {
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), inputs.drop_back());
    return success();
  }

  // mul(..., c1, c2) -> mul(..., c3) where c3 = c1 * c2 -- constant folding
  if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
      matchPattern(inputs[size - 2], m_RConstant(value2))) {
    auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value * value2);
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

template <class Op, bool isSigned>
static OpFoldResult foldDiv(Op op, ArrayRef<Attribute> constants) {
  if (auto rhs_value = constants[1].dyn_cast_or_null<IntegerAttr>()) {
    // divu(x, 1) -> x, divs(x, 1) -> x
    if (rhs_value.getValue() == 1)
      return op.lhs();

    // If the divisor is zero, do not fold for now.
    if (rhs_value.getValue().isZero())
      return {};
  }

  return constFoldBinaryOp(constants, isSigned ? hw::PEO::DivS : hw::PEO::DivU);
}

OpFoldResult DivUOp::fold(ArrayRef<Attribute> constants) {
  return foldDiv<DivUOp, /*isSigned=*/false>(*this, constants);
}

OpFoldResult DivSOp::fold(ArrayRef<Attribute> constants) {
  return foldDiv<DivSOp, /*isSigned=*/true>(*this, constants);
}

template <class Op, bool isSigned>
static OpFoldResult foldMod(Op op, ArrayRef<Attribute> constants) {
  if (auto rhs_value = constants[1].dyn_cast_or_null<IntegerAttr>()) {
    // modu(x, 1) -> 0, mods(x, 1) -> 0
    if (rhs_value.getValue() == 1)
      return getIntAttr(APInt::getZero(op.getType().getIntOrFloatBitWidth()),
                        op.getContext());

    // If the divisor is zero, do not fold for now.
    if (rhs_value.getValue().isZero())
      return {};
  }

  if (auto lhs_value = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    // modu(0, x) -> 0, mods(0, x) -> 0
    if (lhs_value.getValue().isZero())
      return getIntAttr(APInt::getZero(op.getType().getIntOrFloatBitWidth()),
                        op.getContext());
  }

  return constFoldBinaryOp(constants, isSigned ? hw::PEO::ModS : hw::PEO::ModU);
}

OpFoldResult ModUOp::fold(ArrayRef<Attribute> constants) {
  return foldMod<ModUOp, /*isSigned=*/false>(*this, constants);
}

OpFoldResult ModSOp::fold(ArrayRef<Attribute> constants) {
  return foldMod<ModSOp, /*isSigned=*/true>(*this, constants);
}
//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// Constant folding
OpFoldResult ConcatOp::fold(ArrayRef<Attribute> constants) {
  if (getNumOperands() == 1)
    return getOperand(0);

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
    result.insertBits(chunk, nextInsertion);
  }

  return getIntAttr(result, getContext());
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // This function is used when we flatten neighboring operands of a
  // (variadic) concat into a new vesion of the concat.  first/last indices
  // are inclusive.
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

  Value commonOperand = inputs[0];
  for (size_t i = 0; i != size; ++i) {
    // Check to see if all operands are the same.
    if (inputs[i] != commonOperand)
      commonOperand = Value();

    // If an operand to the concat is itself a concat, then we can fold them
    // together.
    if (auto subConcat = inputs[i].getDefiningOp<ConcatOp>())
      return flattenConcat(i, i, subConcat->getOperands());

    // Check for canonicalization due to neighboring operands.
    if (i != 0) {
      // Merge neighboring constants.
      if (auto cst = inputs[i].getDefiningOp<hw::ConstantOp>()) {
        if (auto prevCst = inputs[i - 1].getDefiningOp<hw::ConstantOp>()) {
          unsigned prevWidth = prevCst.getValue().getBitWidth();
          unsigned thisWidth = cst.getValue().getBitWidth();
          auto resultCst = cst.getValue().zext(prevWidth + thisWidth);
          resultCst |= prevCst.getValue().zext(prevWidth + thisWidth)
                       << thisWidth;
          Value replacement =
              rewriter.create<hw::ConstantOp>(op.getLoc(), resultCst);
          return flattenConcat(i - 1, i, replacement);
        }
      }

      // If the two operands are the same, turn them into a replicate.
      if (inputs[i] == inputs[i - 1]) {
        Value replacement =
            rewriter.createOrFold<ReplicateOp>(op.getLoc(), inputs[i], 2);
        return flattenConcat(i - 1, i, replacement);
      }

      // If this input is a replicate, see if we can fold it with the previous
      // one.
      if (auto repl = inputs[i].getDefiningOp<ReplicateOp>()) {
        // ... x, repl(x, n), ...  ==> ..., repl(x, n+1), ...
        if (repl.getOperand() == inputs[i - 1]) {
          Value replacement = rewriter.createOrFold<ReplicateOp>(
              op.getLoc(), repl.getOperand(), repl.getMultiple() + 1);
          return flattenConcat(i - 1, i, replacement);
        }
        // ... repl(x, n), repl(x, m), ...  ==> ..., repl(x, n+m), ...
        if (auto prevRepl = inputs[i - 1].getDefiningOp<ReplicateOp>()) {
          if (prevRepl.getOperand() == repl.getOperand()) {
            Value replacement = rewriter.createOrFold<ReplicateOp>(
                op.getLoc(), repl.getOperand(),
                repl.getMultiple() + prevRepl.getMultiple());
            return flattenConcat(i - 1, i, replacement);
          }
        }
      }

      // ... repl(x, n), x, ...  ==> ..., repl(x, n+1), ...
      if (auto repl = inputs[i - 1].getDefiningOp<ReplicateOp>()) {
        if (repl.getOperand() == inputs[i]) {
          Value replacement = rewriter.createOrFold<ReplicateOp>(
              op.getLoc(), inputs[i], repl.getMultiple() + 1);
          return flattenConcat(i - 1, i, replacement);
        }
      }

      // Merge neighboring extracts of neighboring inputs, e.g.
      // {A[3], A[2]} -> A[3:2]
      if (auto extract = inputs[i].getDefiningOp<ExtractOp>()) {
        if (auto prevExtract = inputs[i - 1].getDefiningOp<ExtractOp>()) {
          if (extract.input() == prevExtract.input()) {
            auto thisWidth = extract.getType().getWidth();
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

  // If all operands were the same, then this is a replicate.
  if (commonOperand) {
    rewriter.replaceOpWithNewOp<ReplicateOp>(op, op.getType(), commonOperand);
    return success();
  }

  return failure();
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
    if (pred.getValue().isZero())
      return falseValue();
    return trueValue();
  }

  return {};
}

/// Check to see if the condition to the specified mux is an equality
/// comparison `indexValue` and one or more constants.  If so, put the
/// constants in the constants vector and return true, otherwise return false.
///
/// This is part of foldMuxChain.
///
static bool
getMuxChainCondConstant(Value cond, Value indexValue, bool isInverted,
                        std::function<void(hw::ConstantOp)> constantFn) {
  // Handle `idx == 42` and `idx != 42`.
  if (auto cmp = cond.getDefiningOp<ICmpOp>()) {
    // TODO: We could handle things like "x < 2" as two entries.
    auto requiredPredicate =
        (isInverted ? ICmpPredicate::eq : ICmpPredicate::ne);
    if (cmp.lhs() == indexValue && cmp.predicate() == requiredPredicate) {
      if (auto cst = cmp.rhs().getDefiningOp<hw::ConstantOp>()) {
        constantFn(cst);
        return true;
      }
    }
    return false;
  }

  // Handle mux(`idx == 1 || idx == 3`, value, muxchain).
  if (auto orOp = cond.getDefiningOp<OrOp>()) {
    if (!isInverted)
      return false;
    for (auto operand : orOp.getOperands())
      if (!getMuxChainCondConstant(operand, indexValue, isInverted, constantFn))
        return false;
    return true;
  }

  // Handle mux(`idx != 1 && idx != 3`, muxchain, value).
  if (auto andOp = cond.getDefiningOp<AndOp>()) {
    if (isInverted)
      return false;
    for (auto operand : andOp.getOperands())
      if (!getMuxChainCondConstant(operand, indexValue, isInverted, constantFn))
        return false;
    return true;
  }

  return false;
}

/// Given a mux, check to see if the "on true" value (or "on false" value if
/// isFalseSide=true) is a mux tree with the same condition.  This allows us
/// to turn things like `mux(VAL == 0, A, (mux (VAL == 1), B, C))` into
/// `array_get (array_create(A, B, C), VAL)` which is far more compact and
/// allows synthesis tools to do more interesting optimizations.
///
/// This returns false if we cannot form the mux tree (or do not want to) and
/// returns true if the mux was replaced.
static bool foldMuxChain(MuxOp rootMux, bool isFalseSide,
                         PatternRewriter &rewriter) {
  // Get the index value being compared.  Later we check to see if it is
  // compared to a constant with the right predicate.
  auto rootCmp = rootMux.cond().getDefiningOp<ICmpOp>();
  if (!rootCmp)
    return false;
  Value indexValue = rootCmp.lhs();

  // Return the value to use if the equality match succeeds.
  auto getCaseValue = [&](MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(!isFalseSide));
  };

  // Return the value to use if the equality match fails.  This is the next
  // mux in the sequence or the "otherwise" value.
  auto getTreeValue = [&](MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(isFalseSide));
  };

  // Start scanning the mux tree to see what we've got.  Keep track of the
  // constant comparison value and the SSA value to use when equal to it.
  SmallVector<Location> locationsFound;
  SmallVector<std::pair<hw::ConstantOp, Value>, 4> valuesFound;

  /// Extract constants and values into `valuesFound` and return true if this is
  /// part of the mux tree, otherwise return false.
  auto collectConstantValues = [&](MuxOp mux) -> bool {
    return getMuxChainCondConstant(
        mux.cond(), indexValue, isFalseSide, [&](hw::ConstantOp cst) {
          valuesFound.push_back({cst, getCaseValue(mux)});
          locationsFound.push_back(mux.cond().getLoc());
          locationsFound.push_back(mux->getLoc());
        });
  };

  // Make sure the root is a correct comparison with a constant.
  if (!collectConstantValues(rootMux))
    return false;

  // Make sure that we're not looking at the intermediate node in a mux tree.
  if (rootMux->hasOneUse()) {
    if (auto userMux = dyn_cast<MuxOp>(*rootMux->user_begin())) {
      if (getTreeValue(userMux) == rootMux.getResult() &&
          getMuxChainCondConstant(userMux.cond(), indexValue, isFalseSide,
                                  [&](hw::ConstantOp cst) {}))
        return false;
    }
  }

  // Scan up the tree linearly.
  auto nextTreeValue = getTreeValue(rootMux);
  while (1) {
    auto nextMux = nextTreeValue.getDefiningOp<MuxOp>();
    if (!nextMux || !nextMux->hasOneUse())
      break;
    if (!collectConstantValues(nextMux))
      break;
    nextTreeValue = getTreeValue(nextMux);
  }

  // We need to have more than three values to create an array.  This is an
  // arbitrary threshold which is saying that one or two muxes together is ok,
  // but three should be folded.
  if (valuesFound.size() < 3)
    return false;

  // If the array is greater that 9 bits, it will take over 512 elements and
  // it will be too large for a single expression.
  auto indexWidth = indexValue.getType().cast<IntegerType>().getWidth();
  if (indexWidth >= 9)
    return false;

  // Next we need to see if the values are dense-ish.  We don't want to have
  // a tremendous number of replicated entries in the array.  Some sparsity is
  // ok though, so we require the table to be at least 5/8 utilized.
  uint64_t tableSize = 1ULL << indexWidth;
  if (valuesFound.size() < (tableSize * 5) / 8)
    return false; // Not dense enough.

  // Ok, we're going to do the transformation, start by building the table
  // filled with the "otherwise" value.
  SmallVector<Value, 8> table(tableSize, nextTreeValue);

  // Fill in entries in the table from the leaf to the root of the expression.
  // This ensures that any duplicate matches end up with the ultimate value,
  // which is the one closer to the root.
  for (auto &elt : llvm::reverse(valuesFound)) {
    uint64_t idx = elt.first.getValue().getZExtValue();
    assert(idx < table.size() && "constant should be same bitwidth as index");
    table[idx] = elt.second;
  }

  // The hw.array_create operation has the operand list in unintuitive order
  // with a[0] stored as the last element, not the first.
  std::reverse(table.begin(), table.end());

  // Build the array_create and the array_get.
  auto fusedLoc = rewriter.getFusedLoc(locationsFound);
  auto array = rewriter.create<hw::ArrayCreateOp>(fusedLoc, table);
  rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(rootMux, array, indexValue);
  return true;
}

/// Given a fully associative variadic operation like (a+b+c+d), break the
/// expression into two parts, one without the specified operand (e.g.
/// `tmp = a+b+d`) and one that combines that into the full expression (e.g.
/// `tmp+c`), and return the inner expression.
///
/// NOTE: This mutates the operation in place if it only has a single user,
/// which assumes that user will be removed.
///
static Value extractOperandFromFullyAssociative(Operation *fullyAssoc,
                                                size_t operandNo,
                                                PatternRewriter &rewriter) {
  assert(fullyAssoc->getNumOperands() >= 2 && "cannot split up unary ops");
  assert(operandNo < fullyAssoc->getNumOperands() && "Invalid operand #");

  // If this expression already has two operands (the common case) no splitting
  // is necessary.
  if (fullyAssoc->getNumOperands() == 2)
    return fullyAssoc->getOperand(operandNo ^ 1);

  // If the operation has a single use, mutate it in place.
  if (fullyAssoc->hasOneUse()) {
    fullyAssoc->eraseOperand(operandNo);
    return fullyAssoc->getResult(0);
  }

  // Form the new operation with the operands that remain.
  SmallVector<Value> operands;
  operands.append(fullyAssoc->getOperands().begin(),
                  fullyAssoc->getOperands().begin() + operandNo);
  operands.append(fullyAssoc->getOperands().begin() + operandNo + 1,
                  fullyAssoc->getOperands().end());
  Value opWithoutExcluded = createGenericOp(
      fullyAssoc->getLoc(), fullyAssoc->getName(), operands, rewriter);
  Value excluded = fullyAssoc->getOperand(operandNo);

  Value fullResult =
      createGenericOp(fullyAssoc->getLoc(), fullyAssoc->getName(),
                      ArrayRef<Value>{opWithoutExcluded, excluded}, rewriter);
  rewriter.replaceOp(fullyAssoc, fullResult);
  return opWithoutExcluded;
}

/// Fold things like `mux(cond, x|y|z|a, a)` -> `(x|y|z)&replicate(cond)|a` and
/// `mux(cond, a, x|y|z|a) -> `(x|y|z)&replicate(~cond) | a` (when isTrueOperand
/// is true.  Return true on successful transformation, false if not.
///
/// These are various forms of "predicated ops" that can be handled with a
/// replicate/and combination.
static bool foldCommonMuxValue(MuxOp op, bool isTrueOperand,
                               PatternRewriter &rewriter) {
  // Check to see the operand in question is an operation.  If it is a port,
  // we can't simplify it.
  Operation *subExpr =
      (isTrueOperand ? op.falseValue() : op.trueValue()).getDefiningOp();
  if (!subExpr || subExpr->getNumOperands() < 2)
    return false;

  // If this isn't an operation we can handle, don't spend energy on it.
  if (!isa<AndOp, XorOp, OrOp, MuxOp>(subExpr))
    return false;

  // Check to see if the common value occurs in the operand list for the
  // subexpression op.  If so, then we can simplify it.
  Value commonValue = isTrueOperand ? op.trueValue() : op.falseValue();
  size_t opNo = 0, e = subExpr->getNumOperands();
  while (opNo != e && subExpr->getOperand(opNo) != commonValue)
    ++opNo;
  if (opNo == e)
    return false;

  // If we got a hit, then go ahead and simplify it!
  Value cond = op.cond();

  auto invertBoolValue = [&](Value value) -> Value {
    auto one = rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(1, 1));
    return rewriter.createOrFold<XorOp>(op.getLoc(), value, one);
  };

  // `mux(cond, a, mux(cond2, a, b))` -> `mux(cond|cond2, a, b)`
  // `mux(cond, a, mux(cond2, b, a))` -> `mux(cond|~cond2, a, b)`
  // `mux(cond, mux(cond2, a, b), a)` -> `mux(~cond|cond2, a, b)`
  // `mux(cond, mux(cond2, b, a), a)` -> `mux(~cond|~cond2, a, b)`
  if (auto subMux = dyn_cast<MuxOp>(subExpr)) {
    Value otherValue;
    Value subCond = subMux.cond();

    // Invert th subCond if needed and dig out the 'b' value.
    if (subMux.trueValue() == commonValue)
      otherValue = subMux.falseValue();
    else if (subMux.falseValue() == commonValue) {
      otherValue = subMux.trueValue();
      subCond = invertBoolValue(subCond);
    } else {
      // We can't fold `mux(cond, a, mux(a, x, y))`.
      return false;
    }

    // Invert the outer cond if needed, and combine the mux conditions.
    if (!isTrueOperand)
      cond = invertBoolValue(cond);
    cond = rewriter.createOrFold<OrOp>(op.getLoc(), cond, subCond);
    rewriter.replaceOpWithNewOp<MuxOp>(op, cond, commonValue, otherValue);
    return true;
  }

  // Invert the condition if needed.  Or/Xor invert when dealing with
  // TrueOperand, And inverts for False operand.
  bool isaAndOp = isa<AndOp>(subExpr);
  if (isTrueOperand ^ isaAndOp)
    cond = invertBoolValue(cond);

  auto extendedCond =
      rewriter.createOrFold<ReplicateOp>(op.getLoc(), op.getType(), cond);

  // Cache this information before subExpr is erased by extraction below.
  bool isaXorOp = isa<XorOp>(subExpr);
  bool isaOrOp = isa<OrOp>(subExpr);

  // Handle the fully associative ops, start by pulling out the subexpression
  // from a many operand version of the op.
  auto restOfAssoc =
      extractOperandFromFullyAssociative(subExpr, opNo, rewriter);

  // `mux(cond, x|y|z|a, a)` -> `(x|y|z)&replicate(cond) | a`
  // `mux(cond, x^y^z^a, a)` -> `(x^y^z)&replicate(cond) ^ a`
  if (isaOrOp || isaXorOp) {
    auto masked =
        rewriter.createOrFold<AndOp>(op.getLoc(), extendedCond, restOfAssoc);
    if (isaXorOp)
      rewriter.replaceOpWithNewOp<XorOp>(op, masked, commonValue);
    else
      rewriter.replaceOpWithNewOp<OrOp>(op, masked, commonValue);
    return true;
  }

  // `mux(cond, a, x&y&z&a)` -> `((x&y&z)|replicate(cond)) & a`
  assert(isaAndOp && "unexpected operation here");
  auto masked =
      rewriter.createOrFold<OrOp>(op.getLoc(), extendedCond, restOfAssoc);
  rewriter.replaceOpWithNewOp<AndOp>(op, masked, commonValue);
  return true;
}

/// This function is invoke when we find a mux with true/false operations that
/// have the same opcode.  Check to see if we can strength reduce the mux by
/// applying it to less data by applying this transformation:
///   `mux(cond, op(a, b), op(a, c))` -> `op(a, mux(cond, b, c))`
static bool foldCommonMuxOperation(MuxOp mux, Operation *trueOp,
                                   Operation *falseOp,
                                   PatternRewriter &rewriter) {
  // Right now we only apply to concat.
  // TODO: Generalize this to and, or, xor, icmp(!), which all occur in practice
  if (!isa<ConcatOp>(trueOp))
    return false;

  // Decode the operands, looking through recursive concats and replicates.
  SmallVector<Value> trueOperands, falseOperands;
  getConcatOperands(trueOp->getResult(0), trueOperands);
  getConcatOperands(falseOp->getResult(0), falseOperands);

  size_t numTrueOperands = trueOperands.size();
  size_t numFalseOperands = falseOperands.size();

  if (!numTrueOperands || !numFalseOperands ||
      (trueOperands.front() != falseOperands.front() &&
       trueOperands.back() != falseOperands.back()))
    return false;

  // Pull all leading shared operands out into their own op if any are common.
  if (trueOperands.front() == falseOperands.front()) {
    SmallVector<Value> operands;
    size_t i;
    for (i = 0; i < numTrueOperands; ++i) {
      Value trueOperand = trueOperands[i];
      if (trueOperand == falseOperands[i])
        operands.push_back(trueOperand);
      else
        break;
    }
    if (i == numTrueOperands) {
      // Selecting between distinct, but lexically identical, concats.
      rewriter.replaceOp(mux, trueOp->getResult(0));
      return true;
    }

    Value sharedMSB;
    if (llvm::all_of(operands, [&](Value v) { return v == operands.front(); }))
      sharedMSB = rewriter.createOrFold<ReplicateOp>(
          mux->getLoc(), operands.front(), operands.size());
    else
      sharedMSB = rewriter.createOrFold<ConcatOp>(mux->getLoc(), operands);
    operands.clear();

    // Get a concat of the LSB's on each side.
    operands.append(trueOperands.begin() + i, trueOperands.end());
    Value trueLSB = rewriter.createOrFold<ConcatOp>(trueOp->getLoc(), operands);
    operands.clear();
    operands.append(falseOperands.begin() + i, falseOperands.end());
    Value falseLSB =
        rewriter.createOrFold<ConcatOp>(falseOp->getLoc(), operands);
    // Merge the LSBs with a new mux and concat the MSB with the LSB to be
    // done.
    Value lsb = rewriter.createOrFold<MuxOp>(mux->getLoc(), mux.cond(), trueLSB,
                                             falseLSB);
    rewriter.replaceOpWithNewOp<ConcatOp>(mux, sharedMSB, lsb);
    return true;
  }

  // If trailing operands match, try to commonize them.
  if (trueOperands.back() == falseOperands.back()) {
    SmallVector<Value> operands;
    size_t i;
    for (i = 0;; ++i) {
      Value trueOperand = trueOperands[numTrueOperands - i - 1];
      if (trueOperand == falseOperands[numFalseOperands - i - 1])
        operands.push_back(trueOperand);
      else
        break;
    }
    std::reverse(operands.begin(), operands.end());
    Value sharedLSB = rewriter.createOrFold<ConcatOp>(mux->getLoc(), operands);
    operands.clear();

    // Get a concat of the MSB's on each side.
    operands.append(trueOperands.begin(), trueOperands.end() - i);
    Value trueMSB = rewriter.createOrFold<ConcatOp>(trueOp->getLoc(), operands);
    operands.clear();
    operands.append(falseOperands.begin(), falseOperands.end() - i);
    Value falseMSB =
        rewriter.createOrFold<ConcatOp>(falseOp->getLoc(), operands);
    // Merge the MSBs with a new mux and concat the MSB with the LSB to be done.
    Value msb = rewriter.createOrFold<MuxOp>(mux->getLoc(), mux.cond(), trueMSB,
                                             falseMSB);
    rewriter.replaceOpWithNewOp<ConcatOp>(mux, msb, sharedLSB);
    return true;
  }

  return false;
}

LogicalResult MuxOp::canonicalize(MuxOp op, PatternRewriter &rewriter) {
  APInt value;

  if (matchPattern(op.trueValue(), m_RConstant(value))) {
    if (value.getBitWidth() == 1) {
      // mux(a, 0, b) -> and(~a, b) for single-bit values.
      if (value.isZero()) {
        auto one = rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(1, 1));
        auto notCond =
            rewriter.createOrFold<XorOp>(op.getLoc(), op.cond(), one);
        rewriter.replaceOpWithNewOp<AndOp>(op, notCond, op.falseValue());
        return success();
      }

      // mux(a, 1, b) -> or(a, b) for single-bit values.
      rewriter.replaceOpWithNewOp<OrOp>(op, op.cond(), op.falseValue());
      return success();
    }

    // Check for mux of two constants.  There are many ways to simplify them.
    APInt value2;
    if (matchPattern(op.falseValue(), m_RConstant(value2))) {
      // When both inputs are constants and differ by only one bit, we can
      // simplify by splitting the mux into up to three contiguous chunks: one
      // for the differing bit and up to two for the bits that are the same.
      // E.g. mux(a, 3'h2, 0) -> concat(0, mux(a, 1, 0), 0) -> concat(0, a, 0)
      APInt xorValue = value ^ value2;
      if (xorValue.isPowerOf2()) {
        unsigned leadingZeros = xorValue.countLeadingZeros();
        unsigned trailingZeros = value.getBitWidth() - leadingZeros - 1;
        SmallVector<Value, 3> operands;

        // Concat operands go from MSB to LSB, so we handle chunks in reverse
        // order of bit indexes.
        // For the chunks that are identical (i.e. correspond to 0s in
        // xorValue), we can extract directly from either input value, and we
        // arbitrarily pick the trueValue().

        if (leadingZeros > 0)
          operands.push_back(rewriter.createOrFold<ExtractOp>(
              op.getLoc(), op.trueValue(), trailingZeros + 1, leadingZeros));

        // Handle the differing bit, which should simplify into either cond or
        // ~cond.
        operands.push_back(rewriter.createOrFold<MuxOp>(
            op.getLoc(), op.cond(),
            rewriter.createOrFold<ExtractOp>(op.getLoc(), op.trueValue(),
                                             trailingZeros, 1),
            rewriter.createOrFold<ExtractOp>(op.getLoc(), op.falseValue(),
                                             trailingZeros, 1)));

        if (trailingZeros > 0)
          operands.push_back(rewriter.createOrFold<ExtractOp>(
              op.getLoc(), op.trueValue(), 0, trailingZeros));

        rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(), operands);
        return success();
      }

      // If the true value is all ones and the false is all zeros then we have a
      // replicate pattern.
      if (value.isAllOnes() && value2.isZero()) {
        rewriter.replaceOpWithNewOp<ReplicateOp>(op, op.getType(), op.cond());
        return success();
      }
    }
  }

  if (matchPattern(op.falseValue(), m_RConstant(value)) &&
      value.getBitWidth() == 1) {
    // mux(a, b, 0) -> and(a, b) for single-bit values.
    if (value.isZero()) {
      rewriter.replaceOpWithNewOp<AndOp>(op, op.cond(), op.trueValue());
      return success();
    }

    // mux(a, b, 1) -> or(~a, b) for single-bit values.
    // falseValue() is known to be a single-bit 1, which we can use for
    // the 1 in the representation of ~ using xor.
    auto notCond =
        rewriter.createOrFold<XorOp>(op.getLoc(), op.cond(), op.falseValue());
    rewriter.replaceOpWithNewOp<OrOp>(op, notCond, op.trueValue());
    return success();
  }

  // mux(!a, b, c) -> mux(a, c, b)
  Value subExpr;
  Operation *condOp = op.cond().getDefiningOp();
  if (condOp && matchPattern(condOp, m_Complement(m_Any(&subExpr)))) {
    rewriter.replaceOpWithNewOp<MuxOp>(op, op.getType(), subExpr,
                                       op.falseValue(), op.trueValue());
    return success();
  }

  // Same but with Demorgan's law.
  // mux(and(~a, ~b, ~c), x, y) -> mux(or(a, b, c), y, x)
  // mux(or(~a, ~b, ~c), x, y) -> mux(and(a, b, c), y, x)
  if (condOp && condOp->hasOneUse()) {
    SmallVector<Value> invertedOperands;

    /// Scan all the operands to see if they are complemented.  If so, build a
    /// vector of them and return true, otherwise return false.
    auto getInvertedOperands = [&]() -> bool {
      for (Value operand : condOp->getOperands()) {
        if (matchPattern(operand, m_Complement(m_Any(&subExpr))))
          invertedOperands.push_back(subExpr);
        else
          return false;
      }
      return true;
    };

    if (isa<AndOp>(condOp) && getInvertedOperands()) {
      auto newOr = rewriter.createOrFold<OrOp>(op.getLoc(), invertedOperands);
      rewriter.replaceOpWithNewOp<MuxOp>(op, newOr, op.falseValue(),
                                         op.trueValue());
      return success();
    }
    if (isa<OrOp>(condOp) && getInvertedOperands()) {
      auto newAnd = rewriter.createOrFold<AndOp>(op.getLoc(), invertedOperands);
      rewriter.replaceOpWithNewOp<MuxOp>(op, newAnd, op.falseValue(),
                                         op.trueValue());
      return success();
    }
  }

  if (auto falseMux =
          dyn_cast_or_null<MuxOp>(op.falseValue().getDefiningOp())) {
    // mux(selector, x, mux(selector, y, z) = mux(selector, x, z)
    if (op.cond() == falseMux.cond()) {
      rewriter.replaceOpWithNewOp<MuxOp>(op, op.cond(), op.trueValue(),
                                         falseMux.falseValue());
      return success();
    }

    // Check to see if we can fold a mux tree into an array_create/get pair.
    if (foldMuxChain(op, /*isFalse*/ true, rewriter))
      return success();
  }

  if (auto trueMux = dyn_cast_or_null<MuxOp>(op.trueValue().getDefiningOp())) {
    // mux(selector, mux(selector, a, b), c) = mux(selector, a, c)
    if (op.cond() == trueMux.cond()) {
      rewriter.replaceOpWithNewOp<MuxOp>(op, op.cond(), trueMux.trueValue(),
                                         op.falseValue());
      return success();
    }

    // Check to see if we can fold a mux tree into an array_create/get pair.
    if (foldMuxChain(op, /*isFalseSide*/ false, rewriter))
      return success();
  }

  // mux(cond, x|y|z|a, a) -> (x|y|z)&replicate(cond) | a
  if (foldCommonMuxValue(op, false, rewriter))
    return success();
  // mux(cond, a, x|y|z|a) -> (x|y|z)&replicate(~cond) | a
  if (foldCommonMuxValue(op, true, rewriter))
    return success();

  // `mux(cond, op(a, b), op(a, c))` -> `op(a, mux(cond, b, c))`
  if (Operation *trueOp = op.trueValue().getDefiningOp())
    if (Operation *falseOp = op.falseValue().getDefiningOp())
      if (trueOp->getName() == falseOp->getName())
        if (foldCommonMuxOperation(op, trueOp, falseOp, rewriter))
          return success();

  return failure();
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

// Given a range of operands, computes the number of matching prefix and
// suffix elements. This does not perform cross-element matching.
template <typename Range>
static size_t computeCommonPrefixLength(const Range &a, const Range &b) {
  size_t commonPrefixLength = 0;
  auto ia = a.begin();
  auto ib = b.begin();

  for (; ia != a.end() && ib != b.end(); ia++, ib++, commonPrefixLength++) {
    if (*ia != *ib) {
      break;
    }
  }

  return commonPrefixLength;
}

static size_t getTotalWidth(ArrayRef<Value> operands) {
  size_t totalWidth = 0;
  for (auto operand : operands) {
    // getIntOrFloatBitWidth should never raise, since all arguments to
    // ConcatOp are integers.
    ssize_t width = operand.getType().getIntOrFloatBitWidth();
    assert(width >= 0);
    totalWidth += width;
  }
  return totalWidth;
}

/// Reduce the strength icmp(concat(...), concat(...)) by doing a element-wise
/// comparison on common prefix and suffixes. Returns success() if a rewriting
/// happens.  This handles both concat and replicate.
static LogicalResult matchAndRewriteCompareConcat(ICmpOp op, Operation *lhs,
                                                  Operation *rhs,
                                                  PatternRewriter &rewriter) {
  // It is safe to assume that [{lhsOperands, rhsOperands}.size() > 0] and
  // all elements have non-zero length. Both these invariants are verified
  // by the ConcatOp verifier.
  SmallVector<Value> lhsOperands, rhsOperands;
  getConcatOperands(lhs->getResult(0), lhsOperands);
  getConcatOperands(rhs->getResult(0), rhsOperands);
  ArrayRef<Value> lhsOperandsRef = lhsOperands, rhsOperandsRef = rhsOperands;

  auto formCatOrReplicate = [&](Location loc,
                                ArrayRef<Value> operands) -> Value {
    assert(!operands.empty());
    Value sameElement = operands[0];
    for (size_t i = 1, e = operands.size(); i != e && sameElement; ++i)
      if (sameElement != operands[i])
        sameElement = Value();
    if (sameElement)
      return rewriter.createOrFold<ReplicateOp>(loc, sameElement,
                                                operands.size());
    return rewriter.createOrFold<ConcatOp>(loc, operands);
  };

  auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
                         Value rhs) -> LogicalResult {
    rewriter.replaceOpWithNewOp<ICmpOp>(op, predicate, lhs, rhs);
    return success();
  };

  size_t commonPrefixLength =
      computeCommonPrefixLength(lhsOperands, rhsOperands);
  if (commonPrefixLength == lhsOperands.size()) {
    // cat(a, b, c) == cat(a, b, c) -> 1
    bool result = applyCmpPredicateToEqualOperands(op.predicate());
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, APInt(1, result));
    return success();
  }

  size_t commonSuffixLength = computeCommonPrefixLength(
      llvm::reverse(lhsOperandsRef), llvm::reverse(rhsOperandsRef));

  size_t commonPrefixTotalWidth =
      getTotalWidth(lhsOperandsRef.take_front(commonPrefixLength));
  size_t commonSuffixTotalWidth =
      getTotalWidth(lhsOperandsRef.take_back(commonSuffixLength));
  auto lhsOnly = lhsOperandsRef.drop_front(commonPrefixLength)
                     .drop_back(commonSuffixLength);
  auto rhsOnly = rhsOperandsRef.drop_front(commonPrefixLength)
                     .drop_back(commonSuffixLength);

  auto replaceWithoutReplicatingSignBit = [&]() {
    auto newLhs = formCatOrReplicate(lhs->getLoc(), lhsOnly);
    auto newRhs = formCatOrReplicate(rhs->getLoc(), rhsOnly);
    return replaceWith(op.predicate(), newLhs, newRhs);
  };

  auto replaceWithReplicatingSignBit = [&]() {
    auto firstNonEmptyValue = lhsOperands[0];
    auto firstNonEmptyElemWidth =
        firstNonEmptyValue.getType().getIntOrFloatBitWidth();
    Value signBit = rewriter.createOrFold<ExtractOp>(
        op.getLoc(), firstNonEmptyValue, firstNonEmptyElemWidth - 1, 1);

    auto newLhs = rewriter.create<ConcatOp>(lhs->getLoc(), signBit, lhsOnly);
    auto newRhs = rewriter.create<ConcatOp>(rhs->getLoc(), signBit, rhsOnly);
    return replaceWith(op.predicate(), newLhs, newRhs);
  };

  if (ICmpOp::isPredicateSigned(op.predicate())) {
    // scmp(cat(..x, b), cat(..y, b)) == scmp(cat(..x), cat(..y))
    if (commonPrefixTotalWidth == 0 && commonSuffixTotalWidth > 0)
      return replaceWithoutReplicatingSignBit();

    // scmp(cat(a, ..x, b), cat(a, ..y, b)) == scmp(cat(sgn(a), ..x),
    // cat(sgn(b), ..y)) Note that we cannot perform this optimization if
    // [width(b) = 0 && width(a) <= 1]. since that common prefix is the sign
    // bit. Doing the rewrite can result in an infinite loop.
    if (commonPrefixTotalWidth > 1 || commonSuffixTotalWidth > 0)
      return replaceWithReplicatingSignBit();

  } else if (commonPrefixTotalWidth > 0 || commonSuffixTotalWidth > 0) {
    // ucmp(cat(a, ..x, b), cat(a, ..y, b)) = ucmp(cat(..x), cat(..y))
    return replaceWithoutReplicatingSignBit();
  }

  return failure();
}

/// Given an equality comparison with a constant value and some operand that has
/// known bits, simplify the comparison to check only the unknown bits of the
/// input.
///
/// One simple example of this is that `concat(0, stuff) == 0` can be simplified
/// to `stuff == 0`, or `and(x, 3) == 0` can be simplified to
/// `extract x[1:0] == 0`
static void combineEqualityICmpWithKnownBitsAndConstant(
    ICmpOp cmpOp, const KnownBits &bitAnalysis, const APInt &rhsCst,
    PatternRewriter &rewriter) {

  // If any of the known bits disagree with any of the comparison bits, then
  // we can constant fold this comparison right away.
  APInt bitsKnown = bitAnalysis.Zero | bitAnalysis.One;
  if ((bitsKnown & rhsCst) != bitAnalysis.One) {
    // If we discover a mismatch then we know an "eq" comparison is false
    // and a "ne" comparison is true!
    bool result = cmpOp.predicate() == ICmpPredicate::ne;
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(cmpOp, APInt(1, result));
    return;
  }

  // Check to see if we can prove the result entirely of the comparison (in
  // which we bail out early), otherwise build a list of values to concat and a
  // smaller constant to compare against.
  SmallVector<Value> newConcatOperands;
  auto newConstant = APInt::getZeroWidth();

  // Ok, some (maybe all) bits are known and some others may be unknown.
  // Extract out segments of the operand and compare against the
  // corresponding bits.
  unsigned knownMSB = bitsKnown.countLeadingOnes();

  Value operand = cmpOp.lhs();

  // Ok, some bits are known but others are not.  Extract out sequences of
  // bits that are unknown and compare just those bits.  We work from MSB to
  // LSB.
  while (knownMSB != bitsKnown.getBitWidth()) {
    // Drop any high bits that are known.
    if (knownMSB)
      bitsKnown = bitsKnown.trunc(bitsKnown.getBitWidth() - knownMSB);

    // Find the span of unknown bits, and extract it.
    unsigned unknownBits = bitsKnown.countLeadingZeros();
    unsigned lowBit = bitsKnown.getBitWidth() - unknownBits;
    auto spanOperand = rewriter.createOrFold<ExtractOp>(
        operand.getLoc(), operand, /*lowBit=*/lowBit,
        /*bitWidth=*/unknownBits);
    auto spanConstant = rhsCst.lshr(lowBit).trunc(unknownBits);

    // Add this info to the concat we're generating.
    newConcatOperands.push_back(spanOperand);
    // FIXME(llvm merge, cc697fc292b0): concat doesn't work with zero bit values
    // newConstant = newConstant.concat(spanConstant);
    if (newConstant.getBitWidth() != 0)
      newConstant = newConstant.concat(spanConstant);
    else
      newConstant = spanConstant;

    // Drop the unknown bits in prep for the next chunk.
    unsigned newWidth = bitsKnown.getBitWidth() - unknownBits;
    bitsKnown = bitsKnown.trunc(newWidth);
    knownMSB = bitsKnown.countLeadingOnes();
  }

  // If all the operands to the concat are foldable then we have an identity
  // situation where all the sub-elements equal each other.  This implies that
  // the overall result is foldable.
  if (newConcatOperands.empty()) {
    bool result = cmpOp.predicate() == ICmpPredicate::eq;
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(cmpOp, APInt(1, result));
    return;
  }

  // If we have a single operand remaining, use it, otherwise form a concat.
  Value concatResult =
      rewriter.createOrFold<ConcatOp>(operand.getLoc(), newConcatOperands);

  // Form the comparison against the smaller constant.
  auto newConstantOp = rewriter.create<hw::ConstantOp>(
      cmpOp.getOperand(1).getLoc(), newConstant);

  rewriter.replaceOpWithNewOp<ICmpOp>(cmpOp, cmpOp.predicate(), concatResult,
                                      newConstantOp);
}

// Simplify icmp eq(xor(a,b,cst1), cst2) -> icmp eq(xor(a,b), cst1^cst2).
static void combineEqualityICmpWithXorOfConstant(ICmpOp cmpOp, XorOp xorOp,
                                                 const APInt &rhs,
                                                 PatternRewriter &rewriter) {
  auto xorRHS = xorOp.getOperands().back().getDefiningOp<hw::ConstantOp>();
  auto newRHS = rewriter.create<hw::ConstantOp>(xorRHS->getLoc(),
                                                xorRHS.getValue() ^ rhs);
  Value newLHS;
  switch (xorOp.getNumOperands()) {
  case 1:
    // This isn't common but is defined so we need to handle it.
    newLHS = rewriter.create<hw::ConstantOp>(xorOp.getLoc(),
                                             APInt::getZero(rhs.getBitWidth()));
    break;
  case 2:
    // The binary case is the most common.
    newLHS = xorOp.getOperand(0);
    break;
  default:
    // The general case forces us to form a new xor with the remaining operands.
    SmallVector<Value> newOperands(xorOp.getOperands());
    newOperands.pop_back();
    newLHS = rewriter.create<XorOp>(xorOp.getLoc(), newOperands);
    break;
  }

  bool xorMultipleUses = !xorOp->hasOneUse();

  // If the xor has multiple uses (not just the compare, then we need/want to
  // replace them as well.
  if (xorMultipleUses) {
    auto newXor = rewriter.create<XorOp>(xorOp.getLoc(), newLHS, xorRHS);
    rewriter.replaceOp(xorOp, newXor->getResult(0));
  }

  // Replace the comparison.
  rewriter.replaceOpWithNewOp<ICmpOp>(cmpOp, cmpOp.predicate(), newLHS, newRHS);
}

LogicalResult ICmpOp::canonicalize(ICmpOp op, PatternRewriter &rewriter) {
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
    auto getConstant = [&](APInt constant) -> Value {
      return rewriter.create<hw::ConstantOp>(op.getLoc(), std::move(constant));
    };

    auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
                           Value rhs) -> LogicalResult {
      rewriter.replaceOpWithNewOp<ICmpOp>(op, predicate, lhs, rhs);
      return success();
    };

    auto replaceWithConstantI1 = [&](bool constant) -> LogicalResult {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, APInt(1, constant));
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
      if (rhs.isAllOnes())
        return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
      // x < min -> false
      if (rhs.isZero())
        return replaceWithConstantI1(0);
      // x < min+1 -> x == min
      if ((rhs - 1).isZero())
        return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs - 1));

      // x < 0xE0 -> extract(x, 5..7) != 0b111
      if (rhs.countLeadingOnes() + rhs.countTrailingZeros() ==
          rhs.getBitWidth()) {
        auto numOnes = rhs.countLeadingOnes();
        auto smaller = rewriter.create<ExtractOp>(
            op.getLoc(), op.lhs(), rhs.getBitWidth() - numOnes, numOnes);
        return replaceWith(ICmpPredicate::ne, smaller,
                           getConstant(APInt::getAllOnes(numOnes)));
      }

      break;
    case ICmpPredicate::ugt:
      // x > min -> x != min
      if (rhs.isZero())
        return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
      // x > max -> false
      if (rhs.isAllOnes())
        return replaceWithConstantI1(0);
      // x > max-1 -> x == max
      if ((rhs + 1).isAllOnes())
        return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs + 1));

      // x > 0x07 -> extract(x, 3..7) != 0b00000
      if ((rhs + 1).isPowerOf2()) {
        auto numOnes = rhs.countTrailingOnes();
        auto newWidth = rhs.getBitWidth() - numOnes;
        auto smaller = rewriter.create<ExtractOp>(op.getLoc(), op.lhs(),
                                                  numOnes, newWidth);
        return replaceWith(ICmpPredicate::ne, smaller,
                           getConstant(APInt::getZero(newWidth)));
      }

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
      if (rhs.isAllOnes())
        return replaceWithConstantI1(1);
      // x <= c -> x < (c+1)
      return replaceWith(ICmpPredicate::ult, op.lhs(), getConstant(rhs + 1));
    case ICmpPredicate::uge:
      // x >= min -> true
      if (rhs.isZero())
        return replaceWithConstantI1(1);
      // x >= c -> x > (c-1)
      return replaceWith(ICmpPredicate::ugt, op.lhs(), getConstant(rhs - 1));
    case ICmpPredicate::eq:
      if (rhs.getBitWidth() == 1) {
        if (rhs.isZero()) {
          // x == 0 -> x ^ 1
          rewriter.replaceOpWithNewOp<XorOp>(op, op.lhs(),
                                             getConstant(APInt(1, 1)));
          return success();
        }
        if (rhs.isAllOnes()) {
          // x == 1 -> x
          rewriter.replaceOp(op, op.lhs());
          return success();
        }
      }
      break;
    case ICmpPredicate::ne:
      if (rhs.getBitWidth() == 1) {
        if (rhs.isZero()) {
          // x != 0 -> x
          rewriter.replaceOp(op, op.lhs());
          return success();
        }
        if (rhs.isAllOnes()) {
          // x != 1 -> x ^ 1
          rewriter.replaceOpWithNewOp<XorOp>(op, op.lhs(),
                                             getConstant(APInt(1, 1)));
          return success();
        }
      }
      break;
    }

    // We have some specific optimizations for comparison with a constant that
    // are only supported for equality comparisons.
    if (op.predicate() == ICmpPredicate::eq ||
        op.predicate() == ICmpPredicate::ne) {
      // Simplify `icmp(value_with_known_bits, rhscst)` into some extracts
      // with a smaller constant.  We only support equality comparisons for
      // this.
      auto knownBits = computeKnownBits(op.lhs());
      if (!knownBits.isUnknown())
        return combineEqualityICmpWithKnownBitsAndConstant(op, knownBits, rhs,
                                                           rewriter),
               success();

      // Simplify icmp eq(xor(a,b,cst1), cst2) -> icmp eq(xor(a,b),
      // cst1^cst2).
      if (auto xorOp = op.lhs().getDefiningOp<XorOp>())
        if (xorOp.getOperands().back().getDefiningOp<hw::ConstantOp>())
          return combineEqualityICmpWithXorOfConstant(op, xorOp, rhs, rewriter),
                 success();
    }
  }

  // icmp(cat(prefix, a, b, suffix), cat(prefix, c, d, suffix)) => icmp(cat(a,
  // b), cat(c, d)). contains special handling for sign bit in signed
  // compressions.
  if (Operation *opLHS = op.lhs().getDefiningOp())
    if (Operation *opRHS = op.rhs().getDefiningOp())
      if (isa<ConcatOp, ReplicateOp>(opLHS) &&
          isa<ConcatOp, ReplicateOp>(opRHS)) {
        if (succeeded(matchAndRewriteCompareConcat(op, opLHS, opRHS, rewriter)))
          return success();
      }

  return failure();
}
