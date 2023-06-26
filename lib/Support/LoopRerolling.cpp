#include "circt/Support/LoopRerolling.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
using namespace circt;
using namespace mlir;

static bool isSafeToInspect(Operation *op) {
  if (isa<hw::InstanceOp>(op))
    return false;
  return isa<hw::HWDialect, comb::CombDialect>(op->getDialect());
}

static bool isSimpleExpression(Value v) {
  auto *op = v.getDefiningOp();
  // Ports or iteration values are ok.
  if (!op)
    return true;
  if (isa<comb::ExtractOp, hw::BitcastOp>(op) && isSimpleExpression(op->getOperand(0)))
    return true;
  // if (isa<hw::ArrayGetOp>(op) && isSimpleExpression(op->getOperand(0)) &&
  //     isSimpleExpression(op->getOperand(1)))
  //   return true;
  // Constants, output ports are ok.
  return isa<hw::ConstantOp, seq::FirRegOp, hw::InstanceOp>(op);
}

LogicalResult circt::LoopReroller::unifyTwoValues(Value value, Value next) {
  sandbox = builder.create<sv::IfDefProceduralOp>(
      "HACK", [&]() { templateValue = unifyTwoValuesImpl(value, next); });
  if (!templateValue)
    return failure();
  return success();
}

Value circt::LoopReroller::unifyTwoValuesImpl(Value value, Value next) {

  if (upperLimitTermSize == termSize || value.getType() != next.getType())
    return {};
  termSize++;
  auto valueOp = value.getDefiningOp();
  auto nextOp = next.getDefiningOp();

  // Base case. Check constantOp.
  if (isSimpleExpression(value) && isSimpleExpression(next)) {
    auto v = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     value.getLoc(), ArrayRef<Type>{value.getType()},
                     ArrayRef<Value>{})
                 .getResult(0);
    dummyValues[v] = SmallVector<Value>{value, next};
    return v;
  }

  if (value == next)
    return value;

  // Make sure that we have the same structure. Attributes are checked
  // afterwards. Also allow only comb and hw operations for correctness.
  if (!valueOp || !nextOp || valueOp->getName() != nextOp->getName() ||
      valueOp->getNumResults() != nextOp->getNumResults() ||
      valueOp->getNumResults() != 1 ||
      valueOp->getNumOperands() != nextOp->getNumOperands() ||
      !isSafeToInspect(valueOp))
    return {};

  APInt valueInt, nextInt;
  SmallVector<Value> newOperands;
  bool changed = false;
  // Check operands.
  for (auto [lhs, rhs] :
       llvm::zip(valueOp->getOperands(), nextOp->getOperands())) {
    auto operand = unifyTwoValuesImpl(lhs, rhs);
    if (!operand)
      return {};
    newOperands.push_back(operand);
    changed |= lhs != operand;
  }

  valueOp->removeAttr("sv.namehint");
  nextOp->removeAttr("sv.namehint");

  // Inspect attributes.
  if (valueOp->getAttrs() == nextOp->getAttrs()) {
    // Reuse the value.
    if (!changed)
      return value;
    Operation *op = builder.clone(*valueOp);
    for (unsigned i = 0; i < valueOp->getNumOperands(); i++)
      op->setOperand(i, newOperands[i]);
    return op->getResult(0);
  }

  // Otherwise, handle operations separately.
  return {};
}

// FIXME: Make this recursive.
LogicalResult circt::LoopReroller::unifyIntoTemplate(Value value) {
  return unifyIntoTemplateImpl(templateValue, value);
}

LogicalResult circt::LoopReroller::unifyIntoTemplateImpl(Value templateValue,
                                                         Value value) {
  if (templateValue == value)
    return success();
  if (templateValue.getType() != value.getType())
    return failure();

  // Base Case.
  if (dummyValues.count(templateValue)) {
    if (isSimpleExpression(value)) {
      // Ok!
      dummyValues[templateValue].push_back(value);
      return success();
    }
    return failure();
  }

  auto valueOp = templateValue.getDefiningOp();
  auto nextOp = value.getDefiningOp();
  // Make sure that we have the same structure. Attributes are checked
  // afterwards. Also allow only comb and hw operations for correctness.
  if (!valueOp || !nextOp || valueOp->getName() != nextOp->getName() ||
      valueOp->getNumOperands() != nextOp->getNumOperands() ||
      !isSafeToInspect(valueOp)) {
    return failure();
  }

  for (auto [lhs, rhs] :
       llvm::zip(valueOp->getOperands(), nextOp->getOperands()))
    if (failed(unifyIntoTemplateImpl(lhs, rhs)))
      return failure();

  // Inspect attributes.
  // TODO: Drop namehints.
  // TODO: Accept ExtractOp.
  nextOp->removeAttr("sv.namehint");
  if (valueOp->getAttrs() == nextOp->getAttrs())
    return success();
  return failure();
}