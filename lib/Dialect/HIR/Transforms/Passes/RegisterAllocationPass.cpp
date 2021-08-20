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
#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace hir;
namespace {
class RegisterAllocationPass
    : public hir::RegisterAllocationBase<RegisterAllocationPass> {
public:
  void runOnOperation() override {
    hir::FuncOp funcOp = getOperation();
    auto scheduleInfo = ScheduleInfo::createScheduleInfo(funcOp);
    if (!scheduleInfo.hasValue()) {
      signalPassFailure();
      return;
    }
    this->info = &scheduleInfo.getValue();

    WalkResult walk = funcOp.walk([this](Operation *operation) -> WalkResult {
      if (failed(this->dispatch(operation)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (walk.wasInterrupted())
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
private:
  LogicalResult getPipelineBalancedValue(ImplicitLocOpBuilder &, Value &, Value,
                                         Value, int64_t);

private:
  template <typename T>
  LogicalResult visitOpWithNoOperandsDelay(T);
  LogicalResult visitOp(hir::ReturnOp);
  LogicalResult visitOp(hir::YieldOp);
  LogicalResult visitOp(hir::CallOp);
  LogicalResult dispatch(Operation *);

private:
  ScheduleInfo *info;

private:
  size_t numLatcheOpsAdded = 0;
  size_t numDelayOpsAdded = 0;
  size_t peakDelay = 0;
  size_t totalDelayRegistersInserted = 0;
};
} // end anonymous namespace

LogicalResult RegisterAllocationPass::getPipelineBalancedValue(
    ImplicitLocOpBuilder &builder, Value &newValue, Value oldValue,
    Value opTimeVar, int64_t opOffset) {

  // If the value is not a builtin sized type or is always valid then no
  // need to pipeline it.
  if ((!helper::isBuiltinSizedType(oldValue.getType())) ||
      info->isAlwaysValid(oldValue)) {
    newValue = oldValue;
    return success();
  }

  Value operandTimeVar = info->getRootTimeVar(oldValue);
  if (!operandTimeVar)
    return builder.emitError("Could not find a time-var for value of type : ")
           << oldValue.getType();
  int64_t operandOffset = info->getRootTimeOffset(oldValue);

  Value opRootTimeVar = info->getRootTimeVar(opTimeVar);
  if (!opRootTimeVar)
    return builder.emitError("Could not find a time-var for tstart");
  opOffset += info->getRootTimeOffset(opTimeVar);

  if (operandTimeVar == opRootTimeVar) {
    if (opOffset < operandOffset) {
      builder.emitError() << "Op offset(" << opOffset << ") < operand offset("
                          << operandOffset
                          << "). operand type :" << oldValue.getType();
      mlir::emitRemark(opRootTimeVar.getLoc(), "Root time var defined here");
      return failure();
    }
    if (opOffset == operandOffset) {
      newValue = oldValue;
      return success();
    }
    newValue = builder
                   .create<hir::DelayOp>(
                       oldValue.getType(), oldValue,
                       builder.getI64IntegerAttr(opOffset - operandOffset),
                       operandTimeVar, builder.getI64IntegerAttr(operandOffset))
                   .getResult();
    this->numDelayOpsAdded++;
    this->peakDelay = (long)this->peakDelay > (opOffset - operandOffset)
                          ? (long)this->peakDelay
                          : (opOffset - operandOffset);
    this->totalDelayRegistersInserted += (opOffset - operandOffset);
    return success();
  }
  newValue =
      builder
          .create<hir::LatchOp>(oldValue.getType(), oldValue, operandTimeVar,
                                builder.getI64IntegerAttr(operandOffset))
          .getResult();
  this->numLatcheOpsAdded++;
  return success();
}

template <typename T>
LogicalResult RegisterAllocationPass::visitOpWithNoOperandsDelay(T op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto operands = op->getOperands();
  for (size_t i = 0; i < operands.size(); i++) {
    auto operand = operands[i];
    Value newOperand;
    if (failed(getPipelineBalancedValue(builder, newOperand, operand,
                                        op.tstart(), op.offset().getValue())))
      return failure();
    op->setOperand(i, newOperand);
  }
  return success();
}

LogicalResult RegisterAllocationPass::visitOp(hir::ReturnOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto operands = op.operands();
  SmallVector<Value> operandsNew;
  auto parentFuncOp = dyn_cast<hir::FuncOp>(op->getParentOp());
  auto parentFuncTy = parentFuncOp.funcTy().dyn_cast<hir::FuncType>();
  auto resultAttrs = parentFuncTy.getResultAttrs();
  assert(parentFuncOp);
  Value tstartFuncOp = parentFuncOp.getRegionTimeVar();
  for (size_t i = 0; i < operands.size(); i++) {
    if (operands[i].getType().isa<hir::TimeType>()) {
      operandsNew.push_back(operands[i]);
      continue;
    }
    int64_t argDelay = helper::extractDelayFromDict(resultAttrs[i]);
    Value operandNew;
    if (failed(getPipelineBalancedValue(builder, operandNew, operands[i],
                                        tstartFuncOp, argDelay)))
      return failure();
    operandsNew.push_back(operandNew);
  }
  op.operandsMutable().assign(operandsNew);
  return success();
}

LogicalResult RegisterAllocationPass::visitOp(hir::YieldOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto operands = op.operands();
  SmallVector<Value> operandsNew;
  ArrayAttr resultAttrs;
  Value tstartRegion;
  if (auto parentIfOp = dyn_cast<hir::IfOp>(op->getParentOp())) {
    resultAttrs = parentIfOp.result_attrs();
    tstartRegion = parentIfOp.getRegionTimeVar();
  }

  for (size_t i = 0; i < operands.size(); i++) {
    if (operands[i].getType().isa<hir::TimeType>()) {
      operandsNew.push_back(operands[i]);
      continue;
    }

    int64_t argDelay =
        helper::extractDelayFromDict(resultAttrs[i].dyn_cast<DictionaryAttr>());

    Value operandNew;
    if (failed(getPipelineBalancedValue(builder, operandNew, operands[i],
                                        tstartRegion, argDelay)))
      return failure();
    operandsNew.push_back(operandNew);
  }

  op.operandsMutable().assign(operandsNew);
  return success();
}

LogicalResult RegisterAllocationPass::visitOp(hir::CallOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), OpBuilder(op));
  auto operands = op.operands();
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  auto inputAttrs = funcTy.getInputAttrs();
  SmallVector<Value> operandsNew;
  for (size_t i = 0; i < operands.size(); i++) {
    Value operandNew;
    if (!helper::isBuiltinSizedType(operands[i].getType()))
      continue;
    uint64_t delay = helper::extractDelayFromDict(inputAttrs[i]);
    if (failed(getPipelineBalancedValue(builder, operandNew, operands[i],
                                        op.tstart(),
                                        op.offset().getValue() + delay)))
      return failure();
    operandsNew.push_back(operandNew);
  }
  op.operandsMutable().assign(operandsNew);
  return success();
}

LogicalResult RegisterAllocationPass::dispatch(Operation *operation) {
  if (auto op = dyn_cast<hir::FuncOp>(operation)) {
    return success();
  }

  if (auto op = dyn_cast<hir::ForOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::LoadOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::StoreOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::DelayOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::AddIOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::SubIOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::MulIOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::AddFOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::SubFOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::MulFOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::LatchOp>(operation)) {
    if (failed(visitOpWithNoOperandsDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::ReturnOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::YieldOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::CallOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<hir::TimeOp>(operation)) {
    return success();
  } else if (auto op = dyn_cast<mlir::ConstantOp>(operation)) {
    return success();
  } else if (auto op = dyn_cast<mlir::IndexCastOp>(operation)) {
    return success();
  } else if (auto op = dyn_cast<mlir::TruncateIOp>(operation)) {
    return success();
  } else if (auto op = dyn_cast<mlir::SignExtendIOp>(operation)) {
    return success();
  } else {
    assert(!dyn_cast<hir::FuncOp>(operation));
    return operation->emitError("Unsupported op for RegisterAllocationPass.");
  }
  return success();
}
namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createRegisterAllocationPass() {
  return std::make_unique<RegisterAllocationPass>();
}
} // namespace hir
} // namespace circt
