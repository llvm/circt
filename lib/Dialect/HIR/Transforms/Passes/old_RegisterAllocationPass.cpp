//=========- RegisterAllocationPass.cpp - Lower all ops---===//
//
// This file implements register allocation pass for HIR.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/Analysis/ScheduleAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetVector.h"

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
    printPassStatistics(funcOp);
    return;
  }

private:
  void printPassStatistics(hir::FuncOp op) {
    op.emitRemark() << "Number of LatchOp added = " << numLatcheOpsAdded
                    << ", Number of DelayOp added = " << numDelayOpsAdded
                    << ", Peak delay inserted = " << peakDelay
                    << ", Total delay registers inserted = "
                    << totalDelayRegistersInserted;
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
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::IndexCastOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
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
      } else if (auto op = dyn_cast<hir::TimeOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<hir::LatchOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else {
        return operation.emitError(
            "Unsupported op for RegisterAllocationPass.");
      }
    }
    return success();
  }

private:
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
  LogicalResult updateOp(mlir::ConstantOp);
  LogicalResult updateOp(mlir::IndexCastOp);
  LogicalResult updateOp(hir::LatchOp);

private:
  template <typename T>
  LogicalResult updateBinOp(T);

private:
  llvm::DenseMap<Value, Value> mapValueToTimeVar;
  llvm::DenseMap<Value, int64_t> mapValueToOffset;
  // Set of those values which do not have a time instance associated with them
  // and are forever valid in their scope. For example, constants and ForOp
  // latched values.
  llvm::SetVector<Value> setOfForeverValidValues;
  size_t numLatcheOpsAdded = 0;
  size_t numDelayOpsAdded = 0;
  size_t peakDelay = 0;
  size_t totalDelayRegistersInserted = 0;
};

} // end anonymous namespace

//-----------------------------------------------------------------------------
// Helper method.
//-----------------------------------------------------------------------------
LogicalResult RegisterAllocationPass::getPipelineBalancedValue(
    ImplicitLocOpBuilder &builder, Value &newValue, Value oldValue,
    Value tstart, int64_t offset) {

  // If the value is IndexType (constant) or forever valid then no need to
  // pipeline it.
  if (oldValue.getType().isa<IndexType>() ||
      setOfForeverValidValues.contains(oldValue)) {
    newValue = oldValue;
    return success();
  }

  if (mapValueToTimeVar.find(oldValue) == mapValueToTimeVar.end())
    return builder.emitError("Could not find Value in mapValueToTimeVar : ")
           << oldValue.getType();
  if (mapValueToOffset.find(oldValue) == mapValueToOffset.end())
    return builder.emitError("Could not find Value in mapValueToOffset.");

  Value srcTimeVar = mapValueToTimeVar[oldValue];
  int64_t srcOffset = mapValueToOffset[oldValue];
  assert(srcTimeVar);
  if (auto timeOp =
          llvm::dyn_cast_or_null<hir::TimeOp>(srcTimeVar.getDefiningOp())) {
    srcTimeVar = timeOp.timevar();
    srcOffset += timeOp.delay();
  }
  if (auto timeOp =
          llvm::dyn_cast_or_null<hir::TimeOp>(tstart.getDefiningOp())) {
    tstart = timeOp.timevar();
    offset += timeOp.delay();
  }
  if (srcTimeVar == tstart) {
    if (offset < srcOffset)
      return builder.emitError("Op offset < operand offset.");
    newValue =
        builder
            .create<hir::DelayOp>(oldValue.getType(), oldValue,
                                  builder.getI64IntegerAttr(offset - srcOffset),
                                  tstart, builder.getI64IntegerAttr(srcOffset))
            .getResult();
    this->numDelayOpsAdded++;
    this->peakDelay = (long)this->peakDelay > (offset - srcOffset)
                          ? (long)this->peakDelay
                          : (offset - srcOffset);
    this->totalDelayRegistersInserted += (offset - srcOffset);
    return success();
  }
  newValue = builder
                 .create<hir::LatchOp>(oldValue.getType(), oldValue, srcTimeVar,
                                       builder.getI64IntegerAttr(srcOffset))
                 .getResult();
  this->numLatcheOpsAdded++;
  return success();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Methods for register allocation.
//-----------------------------------------------------------------------------
LogicalResult RegisterAllocationPass::updateOp(hir::ForOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  // Value lbNew;
  // Value ubNew;
  // Value stepNew;
  // if (failed(getPipelineBalancedValue(builder, lbNew, op.lb(), op.tstart(),
  //                                    op.offset().getValue())))
  //  return failure();
  // op.lbMutable().assign(lbNew);
  // if (failed(getPipelineBalancedValue(builder, ubNew, op.ub(), op.tstart(),
  //                                    op.offset().getValue())))
  //  return failure();
  // op.ubMutable().assign(ubNew);
  // if (failed(getPipelineBalancedValue(builder, stepNew, op.step(),
  // op.tstart(),
  //                                    op.offset().getValue())))
  //  return failure();
  // op.stepMutable().assign(stepNew);

  // Process all operands.
  for (auto operand : op.getOperands()) {
    Value operandNew;
    if (operand.getType().isa<TimeType>())
      continue;
    if (failed(getPipelineBalancedValue(builder, operandNew, operand,
                                        op.tstart(), op.offset().getValue())))
      return failure();
  }

  // Latched values are valid for the whole duration of the loop.
  for (auto latchedInput : op.getLatchedInputs()) {
    setOfForeverValidValues.insert(latchedInput);
  }

  // Update the ForOp body.
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
  mapValueToOffset[op.getResult()] = op.offset().getValue();
  for (size_t i = 0; i < indices.size(); i++) {
    Value idxNew;
    if (failed(getPipelineBalancedValue(builder, idxNew, indices[i],
                                        op.tstart(), op.offset().getValue())))
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
                                        op.tstart(), op.offset().getValue())))
      return failure();
    indicesNew.push_back(idxNew);
  }

  op.indicesMutable().assign(indicesNew);
  Value valueNew;
  if (failed(getPipelineBalancedValue(builder, valueNew, op.value(),
                                      op.tstart(), op.offset().getValue())))
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
  mapValueToOffset[op.getResult()] = op.offset().getValue();
  Value op1New, op2New;
  if (failed(getPipelineBalancedValue(builder, op1New, op.op1(), op.tstart(),
                                      op.offset().getValue())))
    return failure();
  if (failed(getPipelineBalancedValue(builder, op2New, op.op1(), op.tstart(),
                                      op.offset().getValue())))
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
                                        op.tstart(), op.offset().getValue())))
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
                                        op.tstart(), op.offset().getValue())))
      ;
    return failure();
    operandsNew.push_back(operandNew);
  }
  op.operandsMutable().assign(operandsNew);
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(mlir::TruncateIOp op) {
  if (setOfForeverValidValues.contains(op.value())) {
    setOfForeverValidValues.insert(op.getResult());
    return success();
  }
  if (mapValueToTimeVar.find(op.value()) == mapValueToTimeVar.end())
    return op.emitError("Could not find input in mapValueToTimeVar : ");
  mapValueToTimeVar[op.getResult()] = mapValueToTimeVar[op.value()];
  mapValueToOffset[op.getResult()] = mapValueToOffset[op.value()];
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(mlir::SignExtendIOp op) {
  if (setOfForeverValidValues.contains(op.value())) {
    setOfForeverValidValues.insert(op.getResult());
    return success();
  }
  if (mapValueToTimeVar.find(op.value()) == mapValueToTimeVar.end())
    return op.emitError("Could not find input in mapValueToTimeVar : ");
  mapValueToTimeVar[op.getResult()] = mapValueToTimeVar[op.value()];
  mapValueToOffset[op.getResult()] = mapValueToOffset[op.value()];
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(mlir::ConstantOp op) {
  setOfForeverValidValues.insert(op.getResult());
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(mlir::IndexCastOp op) {
  // IndexType values are always constants.
  setOfForeverValidValues.insert(op.getResult());
  return success();
}

LogicalResult RegisterAllocationPass::updateOp(hir::LatchOp op) {
  setOfForeverValidValues.insert(op.getResult());
  return success();
}
namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createRegisterAllocationPass() {
  return std::make_unique<RegisterAllocationPass>();
}
} // namespace hir
} // namespace circt
