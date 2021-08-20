#include "circt/Dialect/HIR/Analysis/ScheduleAnalysis.h"
//#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/Region.h"
//#include "mlir/IR/Types.h"
//#include "mlir/IR/Value.h"
//#include "mlir/Support/LogicalResult.h"

//#include "mlir/Analysis/Utils.h"
//#include "mlir/Dialect/StandardOps/IR/Ops.h"
//#include "llvm/ADT/TypeSwitch.h"
//#include "llvm/Support/Debug.h"

using namespace circt;
using namespace hir;

// TODO:
// - For fixed II calc how long LatchOp output is valid.
// - Track happens-before relations between root time-vars (using POSET).

namespace {
class ScheduleInfoImpl {
public:
  ScheduleInfoImpl(ScheduleInfo &scheduleInfo) : scheduleInfo(scheduleInfo) {}
  LogicalResult dispatch(Operation *);
  LogicalResult visitRegion(Region &);

private:
  LogicalResult visitOp(FuncOp);
  LogicalResult visitOp(ForOp);
  LogicalResult visitOp(TimeOp);
  LogicalResult visitOp(CallOp);
  LogicalResult visitOp(LatchOp);
  LogicalResult visitOp(RecvOp);
  LogicalResult visitOp(DelayOp);
  LogicalResult visitOp(mlir::ConstantOp);
  LogicalResult visitOp(mlir::IndexCastOp);
  LogicalResult visitOp(mlir::SignExtendIOp);
  LogicalResult visitOp(mlir::TruncateIOp);
  template <typename T>
  LogicalResult visitSingleResultOpWithOptionalDelay(T);

private:
  ScheduleInfo &scheduleInfo;
};
} // namespace

//-----------------------------------------------------------------------------
// ScheduleInfo class methods.
//-----------------------------------------------------------------------------

llvm::Optional<ScheduleInfo> ScheduleInfo::createScheduleInfo(FuncOp op) {
  ScheduleInfo scheduleInfo(op);
  ScheduleInfoImpl scheduleInfoImpl(scheduleInfo);

  if (failed(scheduleInfoImpl.dispatch(op)))
    return llvm::None;

  if (failed(scheduleInfoImpl.visitRegion(op.getFuncBody())))
    return llvm::None;

  return std::move(scheduleInfo);
}

bool ScheduleInfo::isAlwaysValid(Value v) {
  return setOfAlwaysValidValues.contains(v);
}
Value ScheduleInfo::getRootTimeVar(Value v) { return mapValueToRootTimeVar[v]; }
uint64_t ScheduleInfo::getRootTimeOffset(Value v) {
  assert(getRootTimeVar(v));
  return (uint64_t)mapValueToOffset[v];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// ScheduleInfoImpl class methods.
//-----------------------------------------------------------------------------
LogicalResult ScheduleInfoImpl::dispatch(Operation *operation) {
  if (auto op = dyn_cast<FuncOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<ForOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<TimeOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<CallOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<LatchOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<mlir::ConstantOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<LoadOp>(operation)) {
    if (failed(visitSingleResultOpWithOptionalDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<RecvOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<AddIOp>(operation)) {
    if (failed(visitSingleResultOpWithOptionalDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<SubIOp>(operation)) {
    if (failed(visitSingleResultOpWithOptionalDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<MulIOp>(operation)) {
    if (failed(visitSingleResultOpWithOptionalDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<AddFOp>(operation)) {
    if (failed(visitSingleResultOpWithOptionalDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<SubFOp>(operation)) {
    if (failed(visitSingleResultOpWithOptionalDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<MulFOp>(operation)) {
    if (failed(visitSingleResultOpWithOptionalDelay(op)))
      return failure();
  } else if (auto op = dyn_cast<DelayOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<mlir::IndexCastOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<mlir::TruncateIOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<mlir::SignExtendIOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (operation->getNumResults() > 0) {
    return operation->emitError(
        " Failed to create ScheduleInfo. Unknown op has results.");
  }
  return success();
}

LogicalResult ScheduleInfoImpl::visitRegion(Region &region) {
  for (Operation &operation : region.front()) {
    if (failed(dispatch(&operation)))
      return failure();
  }
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(FuncOp op) {
  scheduleInfo.setOfRootTimeVars.insert(op.getRegionTimeVar());
  scheduleInfo.mapValueToRootTimeVar[op.getRegionTimeVar()] =
      op.getRegionTimeVar();
  scheduleInfo.mapValueToOffset[op.getRegionTimeVar()] = 0;
  auto operands = op.getFuncBody().front().getArguments();
  auto inputAttrs = op.funcTy().dyn_cast<hir::FuncType>().getInputAttrs();
  for (size_t i = 0; i < operands.size(); i++) {
    Value operand = operands[i];
    if (helper::isBuiltinSizedType(operand.getType())) {
      scheduleInfo.mapValueToRootTimeVar[operand] = op.getRegionTimeVar();
      scheduleInfo.mapValueToOffset[operand] =
          helper::extractDelayFromDict(inputAttrs[i]);
    } else if (operand.getType().isa<TimeType>()) {
      scheduleInfo.setOfRootTimeVars.insert(operand);
      scheduleInfo.mapValueToRootTimeVar[operand] = op.getRegionTimeVar();
      scheduleInfo.mapValueToOffset[operand] = 0;
    }
  }

  return success();
}
LogicalResult ScheduleInfoImpl::visitOp(ForOp op) {
  if (!op.tstart())
    return op.emitError("Failed to create ScheduleInfo. Operation is not "
                        "scheduled yet.");
  scheduleInfo.setOfRootTimeVars.insert(op.getResult());
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] = op.getResult();
  scheduleInfo.mapValueToOffset[op.getResult()] = 0;

  // Register all the region operands with the root time var.
  scheduleInfo.mapValueToOffset[op.getInductionVar()] = 0;
  for (Value operand : op.getBody()->getArguments()) {
    scheduleInfo.mapValueToRootTimeVar[operand] = op.getIterTimeVar();
    scheduleInfo.mapValueToOffset[operand] = 0;
  }
  for (Value latchedOperand : op.getLatchedInputs()) {
    scheduleInfo.setOfAlwaysValidValues.insert(latchedOperand);
  }
  return visitRegion(op.getLoopBody());
}

LogicalResult ScheduleInfoImpl::visitOp(TimeOp op) {
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.timevar()];
  scheduleInfo.mapValueToOffset[op.getResult()] =
      op.delay() + scheduleInfo.mapValueToOffset[op.timevar()];

  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(CallOp op) {
  if (!op.tstart())
    return op.emitError("Failed to create ScheduleInfo. Operation is not "
                        "scheduled yet.");
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  for (size_t i = 0; i < op.getNumResults(); i++) {
    Value res = op.getResult(i);
    Type resTy = res.getType();
    DictionaryAttr attrDict = funcTy.getResultAttrs()[i];
    if (helper::isBuiltinSizedType(resTy)) {
      uint64_t delay = helper::extractDelayFromDict(attrDict);
      scheduleInfo.mapValueToRootTimeVar[res] =
          scheduleInfo.mapValueToRootTimeVar[op.tstart()];
      scheduleInfo.mapValueToOffset[res] =
          delay + scheduleInfo.mapValueToOffset[op.tstart()];
    } else if (resTy.isa<TimeType>()) {
      scheduleInfo.setOfRootTimeVars.insert(res);
    } else {
      op.emitError()
          << "Failed to create ScheduleInfo. Could not handle return type "
          << resTy << ".";
    }
  }
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(LatchOp op) {
  if (!op.tstart())
    return op.emitError("Failed to create ScheduleInfo. Operation is not "
                        "scheduled yet.");
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.tstart()];
  scheduleInfo.mapValueToOffset[op.getResult()] =
      op.offset().getValue() + scheduleInfo.mapValueToOffset[op.tstart()];
  scheduleInfo.setOfAlwaysValidValues.insert(op.getResult());
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(RecvOp op) {
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.tstart()];
  scheduleInfo.mapValueToOffset[op.getResult()] =
      op.offset().getValue() + scheduleInfo.mapValueToOffset[op.tstart()];
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(DelayOp op) {
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.tstart()];
  scheduleInfo.mapValueToOffset[op.getResult()] =
      op.delay() + op.offset().getValue() +
      scheduleInfo.mapValueToOffset[op.tstart()];
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(mlir::ConstantOp op) {
  scheduleInfo.setOfAlwaysValidValues.insert(op.getResult());
  return success();
}

template <typename T>
LogicalResult ScheduleInfoImpl::visitSingleResultOpWithOptionalDelay(T op) {
  if (!op.tstart())
    return op.emitError("Failed to create ScheduleInfo. Operation is not "
                        "scheduled yet.");
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.tstart()];
  scheduleInfo.mapValueToOffset[op.getResult()] =
      scheduleInfo.mapValueToOffset[op.tstart()] + op.offset().getValue() +
      op.delay().getValue();
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(mlir::IndexCastOp op) {
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.in()];

  scheduleInfo.mapValueToOffset[op.getResult()] =
      scheduleInfo.mapValueToOffset[op.in()];

  return success();
  if (scheduleInfo.setOfAlwaysValidValues.contains(op.in()))
    scheduleInfo.setOfAlwaysValidValues.insert(op.getResult());
}

LogicalResult ScheduleInfoImpl::visitOp(mlir::SignExtendIOp op) {
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.value()];

  scheduleInfo.mapValueToOffset[op.getResult()] =
      scheduleInfo.mapValueToOffset[op.value()];

  return success();
  if (scheduleInfo.setOfAlwaysValidValues.contains(op.value()))
    scheduleInfo.setOfAlwaysValidValues.insert(op.getResult());
}

LogicalResult ScheduleInfoImpl::visitOp(mlir::TruncateIOp op) {
  scheduleInfo.mapValueToRootTimeVar[op.getResult()] =
      scheduleInfo.mapValueToRootTimeVar[op.value()];

  scheduleInfo.mapValueToOffset[op.getResult()] =
      scheduleInfo.mapValueToOffset[op.value()];
  if (scheduleInfo.setOfAlwaysValidValues.contains(op.value()))
    scheduleInfo.setOfAlwaysValidValues.insert(op.getResult());
  return success();
}
//-----------------------------------------------------------------------------
//
