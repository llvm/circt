#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
using namespace circt;
using namespace hir;

void BusMapOp::build(
    OpBuilder &builder, OperationState &result, ArrayRef<Value> operands,
    std::function<hir::YieldOp(OpBuilder &, ArrayRef<Value>)> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  auto *bb = builder.createBlock(result.addRegion());
  SmallVector<Value> blockArgs;
  for (auto operand : operands) {
    assert(operand.getType().isa<hir::BusType>());
    blockArgs.push_back(bb->addArgument(
        operand.getType().dyn_cast<hir::BusType>().getElementType()));
  }

  hir::YieldOp terminator = bodyCtor(builder, blockArgs);
  auto results = terminator.operands();
  SmallVector<Type> resultTypes;
  for (auto res : results) {
    resultTypes.push_back(
        hir::BusType::get(builder.getContext(), res.getType()));
  }
  result.addOperands(operands);
  result.addTypes(resultTypes);
}

void BusTensorMapOp::build(
    OpBuilder &builder, OperationState &result, ArrayRef<Value> operands,
    std::function<hir::YieldOp(OpBuilder &, ArrayRef<Value>)> bodyCtor) {

  OpBuilder::InsertionGuard guard(builder);
  auto *bb = builder.createBlock(result.addRegion());
  SmallVector<Value> blockArgs;
  for (auto operand : operands) {
    assert(operand.getType().isa<hir::BusTensorType>());
    blockArgs.push_back(bb->addArgument(
        operand.getType().dyn_cast<hir::BusTensorType>().getElementType()));
  }

  hir::YieldOp terminator = bodyCtor(builder, blockArgs);
  auto results = terminator.operands();
  SmallVector<Type> resultTypes;
  auto shape = operands[0].getType().dyn_cast<hir::BusTensorType>().getShape();
  for (auto res : results) {
    resultTypes.push_back(
        hir::BusTensorType::get(builder.getContext(), shape, res.getType()));
  }
  result.addOperands(operands);
  result.addTypes(resultTypes);
}

void ForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                  Value ub, Value step, Value tstart, IntegerAttr offset,
                  std::function<hir::NextIterOp(OpBuilder &, Value, Value,
                                                Value, Value, IntegerAttr)>
                      bodyCtor) {
  result.addOperands(lb);
  result.addOperands(ub);
  result.addOperands(step);
  if (tstart)
    result.addOperands(tstart);
  if (offset)
    result.addAttribute(offsetAttrName(result.name), offset);

  OpBuilder::InsertionGuard guard(builder);
  auto *bb = builder.createBlock(result.addRegion());
  bb->addArgument(lb.getType());
  auto hirTimeTy = hir::TimeType::get(lb.getContext());
  bb->addArgument(hirTimeTy);

  bodyCtor(builder, lb, ub, step, tstart, offset);
  result.addTypes(hirTimeTy);
}

void BusOp::build(OpBuilder &builder, OperationState &result, Type resTy) {
  assert(resTy.isa<hir::BusType>());
  result.addTypes(resTy);
}
