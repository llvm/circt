//===- CombOps.cpp - Implement the Comb operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements combinational ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace comb;

Value comb::createZExt(OpBuilder &builder, Location loc, Value value,
                       unsigned targetWidth) {
  assert(value.getType().isSignlessInteger());
  auto inputWidth = value.getType().getIntOrFloatBitWidth();
  assert(inputWidth <= targetWidth);

  // Nothing to do if the width already matches.
  if (inputWidth == targetWidth)
    return value;

  // Create a zero constant for the upper bits.
  auto zeros = hw::ConstantOp::create(
      builder, loc, builder.getIntegerType(targetWidth - inputWidth), 0);
  return builder.createOrFold<ConcatOp>(loc, zeros, value);
}

/// Create a sign extension operation from a value of integer type to an equal
/// or larger integer type.
Value comb::createOrFoldSExt(Location loc, Value value, Type destTy,
                             OpBuilder &builder) {
  IntegerType valueType = dyn_cast<IntegerType>(value.getType());
  assert(valueType && isa<IntegerType>(destTy) &&
         valueType.getWidth() <= destTy.getIntOrFloatBitWidth() &&
         valueType.getWidth() != 0 && "invalid sext operands");
  // If already the right size, we are done.
  if (valueType == destTy)
    return value;

  // sext is concat with a replicate of the sign bits and the bottom part.
  auto signBit =
      builder.createOrFold<ExtractOp>(loc, value, valueType.getWidth() - 1, 1);
  auto signBits = builder.createOrFold<ReplicateOp>(
      loc, signBit, destTy.getIntOrFloatBitWidth() - valueType.getWidth());
  return builder.createOrFold<ConcatOp>(loc, signBits, value);
}

Value comb::createOrFoldSExt(Value value, Type destTy,
                             ImplicitLocOpBuilder &builder) {
  return createOrFoldSExt(builder.getLoc(), value, destTy, builder);
}

Value comb::createOrFoldNot(Location loc, Value value, OpBuilder &builder,
                            bool twoState) {
  auto allOnes = hw::ConstantOp::create(builder, loc, value.getType(), -1);
  return builder.createOrFold<XorOp>(loc, value, allOnes, twoState);
}

Value comb::createOrFoldNot(Value value, ImplicitLocOpBuilder &builder,
                            bool twoState) {
  return createOrFoldNot(builder.getLoc(), value, builder, twoState);
}

// Extract individual bits from a value
void comb::extractBits(OpBuilder &builder, Value val,
                       SmallVectorImpl<Value> &bits) {
  assert(val.getType().isInteger() && "expected integer");
  auto width = val.getType().getIntOrFloatBitWidth();
  bits.reserve(width);

  // Check if we can reuse concat operands
  if (auto concat = val.getDefiningOp<comb::ConcatOp>()) {
    if (concat.getNumOperands() == width &&
        llvm::all_of(concat.getOperandTypes(), [](Type type) {
          return type.getIntOrFloatBitWidth() == 1;
        })) {
      // Reverse the operands to match the bit order
      bits.append(std::make_reverse_iterator(concat.getOperands().end()),
                  std::make_reverse_iterator(concat.getOperands().begin()));
      return;
    }
  }

  // Extract individual bits
  for (int64_t i = 0; i < width; ++i)
    bits.push_back(
        builder.createOrFold<comb::ExtractOp>(val.getLoc(), val, i, 1));
}

// Construct a mux tree for given leaf nodes. `selectors` is the selector for
// each level of the tree. Currently the selector is tested from MSB to LSB.
Value comb::constructMuxTree(OpBuilder &builder, Location loc,
                             ArrayRef<Value> selectors,
                             ArrayRef<Value> leafNodes,
                             Value outOfBoundsValue) {
  // Recursive helper function to construct the mux tree
  std::function<Value(size_t, size_t)> constructTreeHelper =
      [&](size_t id, size_t level) -> Value {
    // Base case: at the lowest level, return the result
    if (level == 0) {
      // Return the result for the given index. If the index is out of bounds,
      // return the out-of-bound value.
      return id < leafNodes.size() ? leafNodes[id] : outOfBoundsValue;
    }

    auto selector = selectors[level - 1];

    // Recursive case: create muxes for true and false branches
    auto trueVal = constructTreeHelper(2 * id + 1, level - 1);
    auto falseVal = constructTreeHelper(2 * id, level - 1);

    // Combine the results with a mux
    return builder.createOrFold<comb::MuxOp>(loc, selector, trueVal, falseVal);
  };

  return constructTreeHelper(0, llvm::Log2_64_Ceil(leafNodes.size()));
}

Value comb::createDynamicExtract(OpBuilder &builder, Location loc, Value value,
                                 Value offset, unsigned width) {
  assert(value.getType().isSignlessInteger());
  auto valueWidth = value.getType().getIntOrFloatBitWidth();
  assert(width <= valueWidth);

  // Handle the special case where the offset is constant.
  APInt constOffset;
  if (matchPattern(offset, mlir::m_ConstantInt(&constOffset)))
    if (constOffset.getActiveBits() < 32)
      return builder.createOrFold<comb::ExtractOp>(
          loc, value, constOffset.getZExtValue(), width);

  // Zero-extend the offset, shift the value down, and extract the requested
  // number of bits.
  offset = createZExt(builder, loc, offset, valueWidth);
  value = builder.createOrFold<comb::ShrUOp>(loc, value, offset);
  return builder.createOrFold<comb::ExtractOp>(loc, value, 0, width);
}

Value comb::createDynamicInject(OpBuilder &builder, Location loc, Value value,
                                Value offset, Value replacement,
                                bool twoState) {
  assert(value.getType().isSignlessInteger());
  assert(replacement.getType().isSignlessInteger());
  auto largeWidth = value.getType().getIntOrFloatBitWidth();
  auto smallWidth = replacement.getType().getIntOrFloatBitWidth();
  assert(smallWidth <= largeWidth);

  // If we're inserting a zero-width value there's nothing to do.
  if (smallWidth == 0)
    return value;

  // Handle the special case where the offset is constant.
  APInt constOffset;
  if (matchPattern(offset, mlir::m_ConstantInt(&constOffset)))
    if (constOffset.getActiveBits() < 32)
      return createInject(builder, loc, value, constOffset.getZExtValue(),
                          replacement);

  // Zero-extend the offset and clear the value bits we are replacing.
  offset = createZExt(builder, loc, offset, largeWidth);
  Value mask = hw::ConstantOp::create(
      builder, loc, APInt::getLowBitsSet(largeWidth, smallWidth));
  mask = builder.createOrFold<comb::ShlOp>(loc, mask, offset);
  mask = createOrFoldNot(loc, mask, builder, true);
  value = builder.createOrFold<comb::AndOp>(loc, value, mask, twoState);

  // Zero-extend the replacement value, shift it up to the offset, and merge it
  // with the value that has the corresponding bits cleared.
  replacement = createZExt(builder, loc, replacement, largeWidth);
  replacement = builder.createOrFold<comb::ShlOp>(loc, replacement, offset);
  return builder.createOrFold<comb::OrOp>(loc, value, replacement, twoState);
}

Value comb::createInject(OpBuilder &builder, Location loc, Value value,
                         unsigned offset, Value replacement) {
  assert(value.getType().isSignlessInteger());
  assert(replacement.getType().isSignlessInteger());
  auto largeWidth = value.getType().getIntOrFloatBitWidth();
  auto smallWidth = replacement.getType().getIntOrFloatBitWidth();
  assert(smallWidth <= largeWidth);

  // If the offset is outside the value there's nothing to do.
  if (offset >= largeWidth)
    return value;

  // If we're inserting a zero-width value there's nothing to do.
  if (smallWidth == 0)
    return value;

  // Assemble the pieces of the injection as everything below the offset, the
  // replacement value, and everything above the replacement value.
  SmallVector<Value, 3> fragments;
  auto end = offset + smallWidth;
  if (end < largeWidth)
    fragments.push_back(
        comb::ExtractOp::create(builder, loc, value, end, largeWidth - end));
  if (end <= largeWidth)
    fragments.push_back(replacement);
  else
    fragments.push_back(comb::ExtractOp::create(builder, loc, replacement, 0,
                                                largeWidth - offset));
  if (offset > 0)
    fragments.push_back(
        comb::ExtractOp::create(builder, loc, value, 0, offset));
  return builder.createOrFold<comb::ConcatOp>(loc, fragments);
}

llvm::LogicalResult comb::convertSubToAdd(comb::SubOp subOp,
                                          mlir::PatternRewriter &rewriter) {
  auto lhs = subOp.getLhs();
  auto rhs = subOp.getRhs();
  // Since `-rhs = ~rhs + 1` holds, rewrite `sub(lhs, rhs)` to:
  // sub(lhs, rhs) => add(lhs, -rhs) => add(lhs, add(~rhs, 1))
  // => add(lhs, ~rhs, 1)
  auto notRhs =
      comb::createOrFoldNot(subOp.getLoc(), rhs, rewriter, subOp.getTwoState());
  auto one =
      hw::ConstantOp::create(rewriter, subOp.getLoc(), subOp.getType(), 1);
  replaceOpWithNewOpAndCopyNamehint<comb::AddOp>(
      rewriter, subOp, ValueRange{lhs, notRhs, one}, subOp.getTwoState());
  return success();
}

static llvm::LogicalResult convertDivModUByPowerOfTwo(PatternRewriter &rewriter,
                                                      Operation *op, Value lhs,
                                                      Value rhs, bool isDiv) {
  // Check if the divisor is a power of two constant.
  auto rhsConstantOp = rhs.getDefiningOp<hw::ConstantOp>();
  if (!rhsConstantOp)
    return failure();

  APInt rhsValue = rhsConstantOp.getValue();
  if (!rhsValue.isPowerOf2())
    return failure();

  Location loc = op->getLoc();

  unsigned width = lhs.getType().getIntOrFloatBitWidth();
  unsigned bitPosition = rhsValue.ceilLogBase2();

  if (isDiv) {
    // divu(x, 2^n) -> concat(0...0, extract(x, n, width-n))
    // This is equivalent to a right shift by n bits.

    // Extract the upper bits (equivalent to right shift).
    Value upperBits = rewriter.createOrFold<comb::ExtractOp>(
        loc, lhs, bitPosition, width - bitPosition);

    // Concatenate with zeros on the left.
    Value zeros =
        hw::ConstantOp::create(rewriter, loc, APInt::getZero(bitPosition));

    // use replaceOpWithNewOpAndCopyNamehint?
    replaceOpAndCopyNamehint(
        rewriter, op,
        comb::ConcatOp::create(rewriter, loc,
                               ArrayRef<Value>{zeros, upperBits}));
    return success();
  }

  // modu(x, 2^n) -> concat(0...0, extract(x, 0, n))
  // This extracts the lower n bits (equivalent to bitwise AND with 2^n - 1).

  // Extract the lower bits.
  Value lowerBits =
      rewriter.createOrFold<comb::ExtractOp>(loc, lhs, 0, bitPosition);

  // Concatenate with zeros on the left.
  Value zeros = hw::ConstantOp::create(rewriter, loc,
                                       APInt::getZero(width - bitPosition));

  replaceOpAndCopyNamehint(
      rewriter, op,
      comb::ConcatOp::create(rewriter, loc, ArrayRef<Value>{zeros, lowerBits}));
  return success();
}

LogicalResult comb::convertDivUByPowerOfTwo(DivUOp divOp,
                                            mlir::PatternRewriter &rewriter) {
  return convertDivModUByPowerOfTwo(rewriter, divOp, divOp.getLhs(),
                                    divOp.getRhs(), /*isDiv=*/true);
}

LogicalResult comb::convertModUByPowerOfTwo(ModUOp modOp,
                                            mlir::PatternRewriter &rewriter) {
  return convertDivModUByPowerOfTwo(rewriter, modOp, modOp.getLhs(),
                                    modOp.getRhs(), /*isDiv=*/false);
}

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

ICmpPredicate ICmpOp::getFlippedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return ICmpPredicate::eq;
  case ICmpPredicate::ne:
    return ICmpPredicate::ne;
  case ICmpPredicate::slt:
    return ICmpPredicate::sgt;
  case ICmpPredicate::sle:
    return ICmpPredicate::sge;
  case ICmpPredicate::sgt:
    return ICmpPredicate::slt;
  case ICmpPredicate::sge:
    return ICmpPredicate::sle;
  case ICmpPredicate::ult:
    return ICmpPredicate::ugt;
  case ICmpPredicate::ule:
    return ICmpPredicate::uge;
  case ICmpPredicate::ugt:
    return ICmpPredicate::ult;
  case ICmpPredicate::uge:
    return ICmpPredicate::ule;
  case ICmpPredicate::ceq:
    return ICmpPredicate::ceq;
  case ICmpPredicate::cne:
    return ICmpPredicate::cne;
  case ICmpPredicate::weq:
    return ICmpPredicate::weq;
  case ICmpPredicate::wne:
    return ICmpPredicate::wne;
  }
  llvm_unreachable("unknown comparison predicate");
}

bool ICmpOp::isPredicateSigned(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::ult:
  case ICmpPredicate::ugt:
  case ICmpPredicate::ule:
  case ICmpPredicate::uge:
  case ICmpPredicate::ne:
  case ICmpPredicate::eq:
  case ICmpPredicate::cne:
  case ICmpPredicate::ceq:
  case ICmpPredicate::wne:
  case ICmpPredicate::weq:
    return false;
  case ICmpPredicate::slt:
  case ICmpPredicate::sgt:
  case ICmpPredicate::sle:
  case ICmpPredicate::sge:
    return true;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Returns the predicate for a logically negated comparison, e.g. mapping
/// EQ => NE and SLE => SGT.
ICmpPredicate ICmpOp::getNegatedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return ICmpPredicate::ne;
  case ICmpPredicate::ne:
    return ICmpPredicate::eq;
  case ICmpPredicate::slt:
    return ICmpPredicate::sge;
  case ICmpPredicate::sle:
    return ICmpPredicate::sgt;
  case ICmpPredicate::sgt:
    return ICmpPredicate::sle;
  case ICmpPredicate::sge:
    return ICmpPredicate::slt;
  case ICmpPredicate::ult:
    return ICmpPredicate::uge;
  case ICmpPredicate::ule:
    return ICmpPredicate::ugt;
  case ICmpPredicate::ugt:
    return ICmpPredicate::ule;
  case ICmpPredicate::uge:
    return ICmpPredicate::ult;
  case ICmpPredicate::ceq:
    return ICmpPredicate::cne;
  case ICmpPredicate::cne:
    return ICmpPredicate::ceq;
  case ICmpPredicate::weq:
    return ICmpPredicate::wne;
  case ICmpPredicate::wne:
    return ICmpPredicate::weq;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Return true if this is an equality test with -1, which is a "reduction
/// and" operation in Verilog.
bool ICmpOp::isEqualAllOnes() {
  if (getPredicate() != ICmpPredicate::eq)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isAllOnes();
  return false;
}

/// Return true if this is a not equal test with 0, which is a "reduction
/// or" operation in Verilog.
bool ICmpOp::isNotEqualZero() {
  if (getPredicate() != ICmpPredicate::ne)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isZero();
  return false;
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

LogicalResult ReplicateOp::verify() {
  // The source must be equal or smaller than the dest type, and an even
  // multiple of it.  Both are already known to be signless integers.
  auto srcWidth = cast<IntegerType>(getOperand().getType()).getWidth();
  auto dstWidth = cast<IntegerType>(getType()).getWidth();
  if (srcWidth == 0)
    return emitOpError("replicate does not take zero bit integer");

  if (srcWidth > dstWidth)
    return emitOpError("replicate cannot shrink bitwidth of operand"),
           failure();

  if (dstWidth % srcWidth)
    return emitOpError("replicate must produce integer multiple of operand"),
           failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyUTBinOp(Operation *op) {
  if (op->getOperands().empty())
    return op->emitOpError("requires 1 or more args");
  return success();
}

LogicalResult AddOp::verify() { return verifyUTBinOp(*this); }

LogicalResult MulOp::verify() { return verifyUTBinOp(*this); }

LogicalResult AndOp::verify() { return verifyUTBinOp(*this); }

LogicalResult OrOp::verify() { return verifyUTBinOp(*this); }

LogicalResult XorOp::verify() { return verifyUTBinOp(*this); }

/// Return true if this is a two operand xor with an all ones constant as
/// its RHS operand.
bool XorOp::isBinaryNot() {
  if (getNumOperands() != 2)
    return false;
  if (auto cst = getOperand(1).getDefiningOp<hw::ConstantOp>())
    if (cst.getValue().isAllOnes())
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static unsigned getTotalWidth(ValueRange inputs) {
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += hw::type_cast<IntegerType>(input.getType()).getWidth();
  }
  return resultWidth;
}

void ConcatOp::build(OpBuilder &builder, OperationState &result, Value hd,
                     ValueRange tl) {
  result.addOperands(ValueRange{hd});
  result.addOperands(tl);
  unsigned hdWidth = cast<IntegerType>(hd.getType()).getWidth();
  result.addTypes(builder.getIntegerType(getTotalWidth(tl) + hdWidth));
}

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  unsigned resultWidth = getTotalWidth(operands);
  results.push_back(IntegerType::get(context, resultWidth));
  return success();
}

ParseResult ConcatOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> types;

  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();

  // Parse the operand list, attributes and colon
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  // If we have operands, we need to parse the types too
  if (!operands.empty() && parser.parseTypeList(types))
    return failure();

  if (parser.resolveOperands(operands, types, allOperandLoc, result.operands))
    return failure();

  SmallVector<Type, 1> inferredTypes;
  if (failed(ConcatOp::inferReturnTypes(
          parser.getContext(), result.location, result.operands,
          result.attributes.getDictionary(parser.getContext()),
          result.getRawProperties(), {}, inferredTypes)))
    return failure();

  result.addTypes(inferredTypes);
  return success();
}

void ConcatOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getOperands());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  llvm::interleaveComma(getOperandTypes(), p);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

// Folding of ReverseOp: if the input is constant, compute the reverse at
// compile time.
OpFoldResult comb::ReverseOp::fold(FoldAdaptor adaptor) {
  // Try to cast the input attribute to an IntegerAttr.
  auto cstInput = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getInput());
  if (!cstInput)
    return {};

  APInt val = cstInput.getValue();
  APInt reversedVal = val.reverseBits();

  return mlir::IntegerAttr::get(getType(), reversedVal);
}

namespace {
struct ReverseOfReverse : public OpRewritePattern<comb::ReverseOp> {
  using OpRewritePattern<comb::ReverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ReverseOp op,
                                PatternRewriter &rewriter) const override {
    auto inputOp = op.getInput().getDefiningOp<comb::ReverseOp>();
    if (!inputOp)
      return failure();

    rewriter.replaceOp(op, inputOp.getInput());
    return success();
  }
};
} // namespace

void comb::ReverseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<ReverseOfReverse>(context);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  unsigned srcWidth = cast<IntegerType>(getInput().getType()).getWidth();
  unsigned dstWidth = cast<IntegerType>(getType()).getWidth();
  if (getLowBit() >= srcWidth || srcWidth - getLowBit() < dstWidth)
    return emitOpError("from bit too large for input"), failure();

  return success();
}

LogicalResult TruthTableOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs >= sizeof(size_t) * 8)
    return emitOpError("Truth tables support a maximum of ")
           << sizeof(size_t) * 8 - 1 << " inputs on your platform";

  ArrayAttr table = getLookupTable();
  if (table.size() != (1ull << numInputs))
    return emitOpError("Expected lookup table of 2^n length");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Comb/Comb.cpp.inc"
