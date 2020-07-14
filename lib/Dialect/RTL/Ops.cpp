//===- Ops.cpp - Implement the RTL operations -----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Visitors.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"

using namespace circt;
using namespace rtl;

/// Return true if the specified operation is a combinatorial logic op.
bool rtl::isCombinatorial(Operation *op) {
  struct IsCombClassifier
      : public CombinatorialVisitor<IsCombClassifier, bool> {
    bool visitInvalidComb(Operation *op) { return false; }
    bool visitUnhandledComb(Operation *op) { return true; }
  };

  return IsCombClassifier().dispatchCombinatorialVisitor(op);
}

static Attribute getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(value.getBitWidth(), context),
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

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyConstantOp(ConstantOp constant) {
  // If the result type has a bitwidth, then the attribute must match its width.
  auto intType = constant.getType().cast<IntegerType>();
  if (constant.value().getBitWidth() != intType.getWidth()) {
    constant.emitError(
        "firrtl.constant attribute bitwidth doesn't match return type");
    return failure();
  }

  return success();
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

/// Build a ConstantOp from an APInt and a FIRRTL type, handling the attribute
/// formation for the 'value' attribute.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APInt &value) {

  auto type = IntegerType::get(value.getBitWidth(), IntegerType::Signless,
                               builder.getContext());
  auto attr = builder.getIntegerAttr(type, value);
  return build(builder, result, type, attr);
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto intTy = getType().cast<IntegerType>();
  auto intCst = getValue();

  // Sugar i1 constants with 'true' and 'false'.
  if (intTy.getWidth() == 1)
    return setNameFn(getResult(), intCst.isNullValue() ? "false" : "true");

  // Otherwise, build a complex name with the value and type.
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << intCst << '_' << intTy;
  setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Verify SExtOp and ZExtOp.
static LogicalResult verifyExtOp(Operation *op) {
  // The source must be smaller than the dest type.  Both are already known to
  // be signless integers.
  auto srcType = op->getOperand(0).getType().cast<IntegerType>();
  auto dstType = op->getResult(0).getType().cast<IntegerType>();
  if (srcType.getWidth() >= dstType.getWidth()) {
    op->emitOpError("extension must increase bitwidth of operand");
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

void ConcatOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange inputs) {
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += input.getType().cast<IntegerType>().getWidth();
  }
  build(builder, result, builder.getIntegerType(resultWidth), inputs);
}

void ExtractOp::build(OpBuilder &builder, OperationState &result,
                      Type resultType, Value input, unsigned lowBit) {
  build(builder, result, resultType, input, llvm::APInt(32, lowBit));
}

static LogicalResult verifyExtractOp(ExtractOp op) {
  unsigned srcWidth = op.input().getType().cast<IntegerType>().getWidth();
  unsigned dstWidth = op.getType().cast<IntegerType>().getWidth();
  if (op.getLowBit() >= srcWidth || srcWidth - op.getLowBit() < dstWidth)
    return op.emitOpError("from bit too large for input"), failure();

  return success();
}

OpFoldResult ExtractOp::fold(ArrayRef<Attribute> operands) {
  // If we are extracting the entire input, then return it.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  APInt value;
  if (mlir::matchPattern(input(), m_RConstant(value))) {
    unsigned dstWidth = getType().cast<IntegerType>().getWidth();
    return getIntAttr(value.lshr(getLowBit()).trunc(dstWidth), getContext());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();
  // and(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  APInt value;

  // and(..., 0) -> 0 -- annulment
  if (matchPattern(inputs()[size - 1], m_RConstant(value)) &&
      value.isNullValue())
    return inputs()[size - 1];

  /// TODO: and(..., '1) -> and(...) -- identity
  /// TODO: and(..., x, x) -> and(..., x) -- idempotent
  /// TODO: and(..., c1, c2) -> and(..., c3) where c3 = c1 & c2 -- constant
  /// folding
  /// TODO: and(x, and(...)) -> and(x, ...) -- flatten
  /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement

  return {};
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();
  // or(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  APInt value;

  // or(..., '1) -> '1 -- annulment
  if (matchPattern(inputs()[size - 1], m_RConstant(value)) &&
      value.isAllOnesValue())
    return inputs()[size - 1];

  /// TODO: or(..., 0) -> or(...) -- identity
  /// TODO: or(..., x, x) -> or(..., x) -- idempotent
  /// TODO: or(..., c1, c2) -> or(..., c3) where c3 = c1 | c2 -- constant
  /// folding
  /// TODO: or(x, or(...)) -> or(x, ...) -- flatten
  /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement

  return {};
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();
  // xor(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  /// TODO: xor(..., 0) -> xor(...) -- identity
  /// TODO: xor(..., '1) -> not(xor(...))
  /// TODO: xor(..., x, x) -> xor(..., 0) -- idempotent?
  /// TODO: xor(..., c1, c2) -> xor(..., c3) where c3 = c1 ^ c2 -- constant
  /// folding
  /// TODO: xor(x, xor(...)) -> xor(x, ...) -- flatten
  /// TODO: xor(..., x, not(x)) -> xor(..., '1)

  return {};
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();
  // add(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  /// TODO: add(..., 0) -> add(...) -- identity
  /// TODO: add(..., x, x) -> add(..., shl(x, 1))
  /// TODO: add(..., c1, c2) -> add(..., c3) where c3 = c1 + c2 -- constant
  /// folding
  /// TODO: add(x, add(...)) -> add(x, ...) -- flatten

  return {};
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();
  // mul(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  APInt value;

  // mul(..., 0) -> 0 -- annulment
  if (matchPattern(inputs()[size-1], m_RConstant(value)) &&
  value.isNullValue())
    return inputs()[size-1];

  /// TODO: mul(..., 1) -> mul(...) -- identity
  /// TODO: mul(..., c1, c2) -> mul(..., c3) where c3 = c1 * c2 -- constant
  /// folding
  /// TODO: mul(a, mul(...)) -> mul(a, ...) -- flatten

  return {};
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.cpp.inc"
