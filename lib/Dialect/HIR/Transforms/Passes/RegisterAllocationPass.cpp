//=========- RegisterAllocationPass.cpp - Lower all ops---===//
//
// This file implements register allocation pass for HIR.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace hir;
namespace {
class RegisterAllocationPass
    : public hir::RegisterAllocationBase<RegisterAllocationPass> {
public:
  void runOnOperation() override {
    hir::FuncOp funcOp = getOperation();
    if (failed(updateRegion(funcOp.getFuncBody())))
      signalPassFailure();
    return;
  }

private:
  LogicalResult updateRegion(Region &region) {
    for (auto &operation : region.front()) {
      if (auto op = dyn_cast<hir::ForOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::LoadOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::StoreOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::DelayOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::AddIOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::SubIOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::MulIOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::AddFOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::SubFOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::MulFOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::ConstantOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<mlir::IndexCastOp>(operation)) {
        return operation.emitError(
            "Unsupported op for RegisterAllocationPass. All IndexCast should "
            "have been replaced by ConstantOp by now.");
      } else if (auto op = dyn_cast<mlir::TruncateIOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::SignExtendIOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::YieldOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::ReturnOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::CallOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else
        return operation.emitError(
            "Unsupported op for RegisterAllocationPass.");
    }
    return success();
  }

private:
  LogicalResult getPipelineBalancedValue(ImplicitLocOpBuilder &, Value &, Value,
                                         Value, Value);
  LogicalResult getPipelineBalancedValue(ImplicitLocOpBuilder &, Value &, Value,
                                         Value, int64_t);

private:
  LogicalResult updateOp(hir::ForOp);
  LogicalResult updateOp(hir::LoadOp);
  LogicalResult updateOp(hir::StoreOp);
  LogicalResult updateOp(hir::DelayOp);
  LogicalResult updateOp(hir::YieldOp);
  LogicalResult updateOp(hir::ReturnOp);
  LogicalResult updateOp(hir::CallOp);
  LogicalResult updateOp(mlir::TruncateIOp);
  LogicalResult updateOp(mlir::SignExtendIOp);

private:
  template <typename T>
  LogicalResult updateBinOp(T);

private:
  llvm::DenseMap<Value, Value> mapValueToTimeVar;
  llvm::DenseMap<Value, int64_t> mapValueToOffset;
};

} // end anonymous namespace

//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------
//
LogicalResult RegisterAllocationPass::getPipelineBalancedValue(
    ImplicitLocOpBuilder &builder, Value &newValue, Value oldValue,
    Value tstart, Value offset) {

  // If the value is constant then no need to add pipeline stages.
  if (oldValue.getDefiningOp() &&
      dyn_cast<mlir::ConstantOp>(oldValue.getDefiningOp())) {
    newValue = oldValue;
    return success();
  }

  if (mapValueToTimeVar.find(oldValue) == mapValueToTimeVar.end())
    return builder.emitError("Could not find Value in mapValueToTimeVar : ")
           << oldValue.getType();
  Value srcTimeVar = mapValueToTimeVar[oldValue];
  if (mapValueToOffset.find(oldValue) == mapValueToOffset.end())
    return builder.emitError("Could not find Value in mapValueToOffset.");
  int64_t destOffset;
  if (offset)
    destOffset = dyn_cast<mlir::ConstantOp>(offset.getDefiningOp())
                     .value()
                     .dyn_cast<IntegerAttr>()
                     .getInt();
  else
    destOffset = 0;

  int64_t srcOffset = mapValueToOffset[oldValue];
  Value cSrcOffset = builder.create<mlir::ConstantOp>(
      builder.getIndexType(), builder.getIndexAttr(destOffset - srcOffset));

  if (srcTimeVar == tstart) {
    if (destOffset < srcOffset)
      return builder.emitError("Dest offset < source offset.");
    Value cDelay = builder.create<mlir::ConstantOp>(
        builder.getIndexType(), builder.getIndexAttr(destOffset - srcOffset));
    newValue = builder
                   .create<hir::DelayOp>(oldValue.getType(), oldValue, cDelay,
                                         tstart, cSrcOffset)
                   .getResult();
    return success();
  }
  newValue = builder
                 .create<hir::LatchOp>(oldValue.getType(), oldValue, srcTimeVar,
                                       cSrcOffset, tstart, offset)
                 .getResult();
  return success();
}

LogicalResult RegisterAllocationPass::getPipelineBalancedValue(
    ImplicitLocOpBuilder &builder, Value &newValue, Value oldValue,
    Value tstart, int64_t destOffset) {

  // If the value is constant then no need to add pipeline stages.
  if (oldValue.getDefiningOp() &&
      dyn_cast<mlir::ConstantOp>(oldValue.getDefiningOp())) {
    newValue = oldValue;
    return success();
  }
  if (mapValueToTimeVar.find(oldValue) == mapValueToTimeVar.end())
    return builder.emitError("Could not find Value in mapValueToTimeVar.");
  Value srcTimeVar = mapValueToTimeVar[oldValue];
  if (mapValueToOffset.find(oldValue) == mapValueToOffset.end())
    return builder.emitError("Could not find Value in mapValueToOffset.");
  int64_t srcOffset = mapValueToOffset[oldValue];
  if (destOffset < srcOffset)
    return builder.emitError("Dest offset < source offset.");

  Value cSrcOffset = builder.create<mlir::ConstantOp>(
      builder.getIndexType(), builder.getIndexAttr(destOffset - srcOffset));
  if (srcTimeVar != tstart)
    return builder.emitError("Expected src and dest time vars to be same.");
  Value cDelay = builder.create<mlir::ConstantOp>(
      builder.getIndexType(), builder.getIndexAttr(destOffset - srcOffset));
  newValue = builder
                 .create<hir::DelayOp>(oldValue.getType(), oldValue, cDelay,
                                       tstart, cSrcOffset)
                 .getResult();
  return success();
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Methods for register allocation.
//-----------------------------------------------------------------------------
LogicalResult RegisterAllocationPass::updateOp(hir::ForOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  if (op->getNumResults() > 0) {
    Value tfinish = op.getResult(0);
    assert(tfinish.getType().dyn_cast<hir::TimeType>());
    for (auto res : op.getResults()) {
      mapValueToTimeVar[res] = tfinish;
    }
  }
  Value lbNew;
  Value ubNew;
  Value stepNew;
  if (failed(getPipelineBalancedValue(builder, lbNew, op.lb(), op.tstart(),
                                      op.offset())))
    return failure();
  op.lbMutable().assign(lbNew);
  if (failed(getPipelineBalancedValue(builder, ubNew, op.ub(), op.tstart(),
                                      op.offset())))
    return failure();
  op.ubMutable().assign(ubNew);
  if (failed(getPipelineBalancedValue(builder, stepNew, op.step(), op.tstart(),
                                      op.offset())))
    return failure();
  op.stepMutable().assign(stepNew);
  mapValueToTimeVar[op.getInductionVar()] = op.getIterTimeVar();
  mapValueToOffset[op.getInductionVar()] = 0;
  if (failed(updateRegion(op.getLoopBody())))
    return failure();
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(hir::LoadOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto indices = op.indices();
  SmallVector<Value> indicesNew;
  mapValueToTimeVar[op.getResult()] = op.tstart();
  mapValueToOffset[op.getResult()] =
      op.offset() ? dyn_cast<mlir::ConstantOp>(op.offset().getDefiningOp())
                        .value()
                        .dyn_cast<IntegerAttr>()
                        .getInt()
                  : 0;
  for (size_t i = 0; i < indices.size(); i++) {
    Value idxNew;
    if (failed(getPipelineBalancedValue(builder, idxNew, indices[i],
                                        op.tstart(), op.offset())))
      return failure();
    indicesNew.push_back(idxNew);
  }
  op.indicesMutable().assign(indicesNew);
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(hir::StoreOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto indices = op.indices();
  SmallVector<Value> indicesNew;
  for (size_t i = 0; i < indices.size(); i++) {
    Value idxNew;
    if (failed(getPipelineBalancedValue(builder, idxNew, indices[i],
                                        op.tstart(), op.offset())))
      return failure();
    indicesNew.push_back(idxNew);
  }

  op.indicesMutable().assign(indicesNew);
  Value valueNew;
  if (failed(getPipelineBalancedValue(builder, valueNew, op.value(),
                                      op.tstart(), op.offset())))
    return failure();
  op.valueMutable().assign(valueNew);
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(hir::DelayOp op) {
  // TimeType delays are assumed to be correct-by-construction from the
  // scheduleing pass.
  if (!op.getResult().getType().isa<hir::TimeType>())
    return op.emitError("Only hir.delay of !hir.time type is supported in the "
                        "register allocation pass.");
  return success();
}

template <typename T>
LogicalResult RegisterAllocationPass::updateBinOp(T op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  mapValueToTimeVar[op.getResult()] = op.tstart();
  if (op.offset()) {
    Attribute valueAttr =
        dyn_cast<mlir::ConstantOp>(op.offset().getDefiningOp()).value();
    mapValueToOffset[op.getResult()] =
        valueAttr.dyn_cast<IntegerAttr>().getInt();
  } else
    mapValueToOffset[op.getResult()] = 0;
  Value op1New, op2New;
  if (failed(getPipelineBalancedValue(builder, op1New, op.op1(), op.tstart(),
                                      op.offset())))
    return failure();
  if (failed(getPipelineBalancedValue(builder, op2New, op.op1(), op.tstart(),
                                      op.offset())))
    return failure();
  op.op1Mutable().assign(op1New);
  op.op2Mutable().assign(op2New);
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(hir::YieldOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto operands = op.operands();
  SmallVector<Value> operandsNew;
  for (size_t i = 0; i < operands.size(); i++) {
    Value operandNew;
    if (failed(getPipelineBalancedValue(builder, operandNew, operands[i],
                                        op.tstart(), op.offset())))
      return failure();
    operandsNew.push_back(operandNew);
  }
  op.operandsMutable().assign(operandsNew);
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(hir::ReturnOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto operands = op.operands();
  SmallVector<Value> operandsNew;
  auto parentFuncOp = dyn_cast<hir::FuncOp>(op->getParentOp());
  auto parentFuncTy = parentFuncOp.funcTy().dyn_cast<hir::FuncType>();
  assert(parentFuncOp);
  Value tstartFuncOp = parentFuncOp.getTimeVar();
  for (size_t i = 0; i < operands.size(); i++) {
    int64_t argDelay = parentFuncTy.getResultAttrs()[i]
                           .getNamed("hir.delay")
                           ->second.dyn_cast<IntegerAttr>()
                           .getInt();
    Value operandNew;
    if (failed(getPipelineBalancedValue(builder, operandNew, operands[i],
                                        tstartFuncOp, argDelay)))
      return failure();
    operandsNew.push_back(operandNew);
  }
  op.operandsMutable().assign(operandsNew);
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(hir::CallOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto operands = op.operands();
  SmallVector<Value> operandsNew;
  for (size_t i = 0; i < operands.size(); i++) {
    Value operandNew;
    if (failed(getPipelineBalancedValue(builder, operandNew, operands[i],
                                        op.tstart(), op.offset())))
      ;
    return failure();
    operandsNew.push_back(operandNew);
  }
  op.operandsMutable().assign(operandsNew);
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(mlir::TruncateIOp op) {
  mapValueToTimeVar[op.value()] = mapValueToTimeVar[op.getResult()];
  mapValueToOffset[op.value()] = mapValueToOffset[op.getResult()];
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(mlir::SignExtendIOp op) {
  mapValueToTimeVar[op.value()] = mapValueToTimeVar[op.getResult()];
  mapValueToOffset[op.value()] = mapValueToOffset[op.getResult()];
  return success();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createRegisterAllocationPass() {
  return std::make_unique<RegisterAllocationPass>();
}
} // namespace hir
} // namespace circt
