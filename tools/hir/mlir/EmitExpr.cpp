#include "../include/HIRGen.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "circt/Dialect/HIR/helper.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace hir;

mlir::Value emitUnaryOp(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                        StringRef opName, Value input, Value t, Value offset,
                        int delay, Type resultTy) {
  mlir::Value out;
  SmallVector<Type, 1> resultTypes = {resultTy};
  FunctionType functionTy =
      FunctionType::get(&context, input.getType(), resultTypes);

  ArrayAttr inputDelays =
      ArrayAttr::get(&context, {helper::getIntegerAttr(&context, 64, 0),
                                helper::getIntegerAttr(&context, 64, 0)});
  ArrayAttr outputDelays =
      ArrayAttr::get(&context, {helper::getIntegerAttr(&context, 64, delay)});
  hir::FuncType funcTy =
      hir::FuncType::get(&context, functionTy, inputDelays, outputDelays);
  auto op = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), resultTypes,
      /*callee*/ FlatSymbolRefAttr::get(&context, opName),
      /*funcTy*/ funcTy,
      /*callee_var*/ Value(),
      /*operands*/ SmallVector<mlir::Value, 4>({input}), /*tstart*/ t,
      /*offset*/ offset);
  return op.getResult(0);
}

mlir::Value emitBinOp(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                      StringRef opName, Value left, Value right, Value t,
                      Value offset, int delay, Type resultTy) {
  mlir::Value out;
  SmallVector<Type, 1> resultTypes = {resultTy};
  FunctionType functionTy = FunctionType::get(
      &context, {left.getType(), right.getType()}, resultTypes);
  ArrayAttr inputDelays =
      ArrayAttr::get(&context, {helper::getIntegerAttr(&context, 64, 0),
                                helper::getIntegerAttr(&context, 64, 0)});
  ArrayAttr outputDelays =
      ArrayAttr::get(&context, {helper::getIntegerAttr(&context, 64, delay)});
  hir::FuncType funcTy =
      hir::FuncType::get(&context, functionTy, inputDelays, outputDelays);
  auto op = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), resultTypes,
      /*callee*/ FlatSymbolRefAttr::get(&context, opName),
      /*funcTy*/ funcTy,
      /*callee_var*/ Value(),
      /*operands*/ SmallVector<mlir::Value, 4>({left, right}), /*tstart*/ t,
      /*offset*/ offset);
  return op.getResult(0);
}

mlir::Value emitSelectOp(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                         Value input, SmallVectorImpl<int> &indices,
                         Type resultTy) {

  GroupType interfaceTy = GroupType::get(&context, {resultTy}, {Attribute()});
  SmallVector<Value, 4> addr;

  for (auto index : indices) {
    assert(index >= 0);
    auto op = builder.create<hir::ConstantOp>(
        builder.getUnknownLoc(), helper::getConstIntType(&context),
        helper::getIntegerAttr(&context, 64, index));
    addr.push_back(op.getResult());
  }

  auto op =
      builder.create<hir::SelectOp>(builder.getUnknownLoc(), interfaceTy, addr);

  return op.getResult();
}

mlir::Value emitExprBody(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                         SmallVectorImpl<Value> &args, Value t,
                         ExprTree *expr) {
  if (auto *unaryOp = expr->getUnaryOp()) {
    Value input =
        emitExprBody(builder, context, args, t, unaryOp->getInputOperand());
    Value offset =
        builder
            .create<hir::ConstantOp>(
                builder.getUnknownLoc(), helper::getConstIntType(&context),
                helper::getIntegerAttr(&context, 64, unaryOp->getOffset()))
            .getResult();
    return emitUnaryOp(builder, context, unaryOp->getOpName(), input, t, offset,
                       unaryOp->getOpDelay(), unaryOp->getResultType());
  }

  if (auto *binOp = expr->getBinOp()) {
    Value left =
        emitExprBody(builder, context, args, t, binOp->getLeftOperand());
    Value right =
        emitExprBody(builder, context, args, t, binOp->getRightOperand());
    Value offset =
        builder
            .create<hir::ConstantOp>(
                builder.getUnknownLoc(), helper::getConstIntType(&context),
                helper::getIntegerAttr(&context, 64, binOp->getOffset()))
            .getResult();

    return emitBinOp(builder, context, binOp->getOpName(), left, right, t,
                     offset, binOp->getOpDelay(), binOp->getResultType());
  }

  if (auto *selectOp = expr->getSelectOp()) {
    Value input =
        emitExprBody(builder, context, args, t, selectOp->getInputOperand());
    return emitSelectOp(builder, context, input, selectOp->getIndices(),
                        selectOp->getResultType());
  }

  return expr->getValue(args);
}

void emitExpr(mlir::OpBuilder &builder, mlir::MLIRContext &context,
              mlir::OwningModuleRef &module, StringRef funcName,
              SmallVectorImpl<Type> &argTypes, ExprTree *expr) {

  auto functionTy =
      builder.getFunctionType(argTypes, expr->getResultType(argTypes));
  IntegerAttr zeroAttr = helper::getIntegerAttr(&context, 64, 0);
  SmallVector<Attribute> inputDelays;
  for (size_t i = 0; i < argTypes.size(); i++)
    inputDelays.push_back(zeroAttr);
  SmallVector<Attribute> outputDelays = {
      helper::getIntegerAttr(&context, 64, expr->getResultOffset())};

  auto funcTy =
      FuncType::get(&context, functionTy, ArrayAttr::get(&context, inputDelays),
                    ArrayAttr::get(&context, outputDelays));
  auto symName = mlir::StringAttr::get(&context, funcName);
  auto funcOp = builder.create<hir::FuncOp>(builder.getUnknownLoc(),
                                            TypeAttr::get(functionTy), symName,
                                            TypeAttr::get(funcTy));
  Block *entryBlock = funcOp.addEntryBlock();
  entryBlock->addArgument(TimeType::get(&context));
  SmallVector<Value, 4> args;
  for (size_t i = 0; i < entryBlock->getNumArguments() - 1; i++) {
    auto arg = entryBlock->getArgument(i);
    args.push_back(arg);
  }
  Value t = entryBlock->getArgument(entryBlock->getNumArguments() - 1);

  builder.setInsertionPointToStart(entryBlock);
  Value result = emitExprBody(builder, context, args, t, expr);
  builder.create<hir::ReturnOp>(builder.getUnknownLoc(),
                                SmallVector<Value>({result}));
  module->push_back(funcOp);
}
