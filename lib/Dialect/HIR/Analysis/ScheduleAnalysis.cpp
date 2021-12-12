#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HIR/Analysis/ScheduleAnalysis.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace hir;

namespace {
class ScheduleInfoImpl {
public:
  ScheduleInfoImpl(ScheduleInfo &scheduleInfo) : scheduleInfo(scheduleInfo) {}
  LogicalResult visitOperation(Operation *);

private:
  LogicalResult visitOp(FuncOp);
  LogicalResult visitOp(ForOp);
  LogicalResult visitOp(WhileOp);
  LogicalResult visitOp(TimeOp);
  LogicalResult visitOp(CallOp);
  LogicalResult visitOp(LoadOp);
  LogicalResult visitOp(BusRecvOp);
  LogicalResult visitOp(DelayOp);
  LogicalResult visitCombOperation(Operation *);
  LogicalResult visitOp(mlir::arith::ConstantOp);
  LogicalResult visitOp(hw::ConstantOp);

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

  auto walkResult = op.walk<mlir::WalkOrder::PreOrder>(
      [&scheduleInfoImpl](Operation *operation) {
        if (failed(scheduleInfoImpl.visitOperation(operation)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

  if (walkResult.wasInterrupted())
    return llvm::None;
  return std::move(scheduleInfo);
}

Value ScheduleInfo::getRootTimeVar(Value v) { return mapValueToRootTimeVar[v]; }
int64_t ScheduleInfo::getRootTimeOffset(Value v) {
  assert(getRootTimeVar(v));
  return mapValueToOffset[v];
}
void ScheduleInfo::mapValueToTime(Value v, Value tstart, int64_t offset) {
  mapValueToRootTimeVar[v] = tstart;
  mapValueToOffset[v] = offset;
}

void ScheduleInfo::mapValueToAlwaysValid(Value v) {
  setOfAlwaysValidValues.insert(v);
}

void ScheduleInfo::setAsRootTimeVar(Value tstart) {
  mapValueToRootTimeVar[tstart] = tstart;
  mapValueToOffset[tstart] = 0;
  setOfRootTimeVars.insert(tstart);
}

void ScheduleInfo::setAsAlwaysValidValue(Value v) {
  setOfAlwaysValidValues.insert(v);
}

bool ScheduleInfo::isValidAtTime(Value v, Value tstart, int64_t offset) {
  if (setOfAlwaysValidValues.contains(v))
    return true;
  // We can't check relation between two time-domains, so we assume that value
  // is valid (optimisitic analysis).
  if (getRootTimeVar(v) != tstart)
    return true;
  if (getRootTimeOffset(v) == offset)
    return true;
  // v.getDefiningOp()->emitError("offset=")
  //    << offset << "rootTimeOffset=" << getRootTimeOffset(v);
  return false;
}

bool ScheduleInfo::isAlwaysValid(Value v) {
  return setOfAlwaysValidValues.contains(v);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// ScheduleInfoImpl class methods.
//-----------------------------------------------------------------------------
LogicalResult ScheduleInfoImpl::visitOperation(Operation *operation) {
  if (auto op = dyn_cast<FuncOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<ForOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<WhileOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<CallOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<TimeOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<LoadOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<BusRecvOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<DelayOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (isa<comb::CombDialect>(operation->getDialect())) {
    if (failed(visitCombOperation(operation)))
      return failure();
  } else if (auto op = dyn_cast<mlir::arith::ConstantOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (auto op = dyn_cast<hw::ConstantOp>(operation)) {
    if (failed(visitOp(op)))
      return failure();
  } else if (operation->getNumResults() > 0) {
    for (auto result : operation->getResults())
      if (helper::isBuiltinSizedType(result.getType()))
        return operation->emitError("Unsupported op for schedule analysis");
  }
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(FuncOp op) {
  scheduleInfo.setAsRootTimeVar(op.getRegionTimeVar());
  auto operands = op.getFuncBody().front().getArguments();
  auto inputAttrs = op.funcTy().dyn_cast<hir::FuncType>().getInputAttrs();
  for (size_t i = 0; i < operands.size(); i++) {
    Value operand = operands[i];
    if (helper::isBuiltinSizedType(operand.getType())) {
      scheduleInfo.mapValueToTime(operand, op.getRegionTimeVar(),
                                  helper::extractDelayFromDict(inputAttrs[i]));
    } else if (operand.getType().isa<TimeType>()) {
      scheduleInfo.setAsRootTimeVar(operand);
    }
  }
  printf("funcOp visited.");
  fflush(stdout);
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(ForOp op) {
  if (!op.tstart())
    return op.emitError("Failed to create ScheduleInfo. Operation is not "
                        "scheduled yet.");

  // register induction var.
  scheduleInfo.mapValueToTime(op.getInductionVar(), op.getIterTimeVar(), 0);

  // Register the region time var as a new root time var.
  scheduleInfo.setAsRootTimeVar(op.getIterTimeVar());

  // Register the t_end time var as a new root time var.
  scheduleInfo.setAsRootTimeVar(op.t_end());

  // Register the iter args and corresponding ForOp results.
  if (op.iter_arg_delays()) {
    auto iterArgDelays = op.iter_arg_delays().getValue();
    for (size_t i = 0; i < iterArgDelays.size(); i++) {
      auto delay = iterArgDelays[i].dyn_cast<mlir::IntegerAttr>().getInt();
      auto iterArg = op.body().getArgument(i);
      scheduleInfo.mapValueToTime(iterArg, op.getIterTimeVar(), delay);
      scheduleInfo.mapValueToTime(op.getResult(i), op.t_end(), delay);
    }
  }
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(WhileOp op) {
  if (!op.tstart())
    return op.emitError("Failed to create ScheduleInfo. Operation is not "
                        "scheduled yet.");

  // Register the region time var as a new root time var.
  scheduleInfo.setAsRootTimeVar(op.getIterTimeVar());

  // Register the t_end time var as a new root time var.
  scheduleInfo.setAsRootTimeVar(op.t_end());

  // Register the iter args and corresponding WhileOp results.
  if (auto iterArgDelays = op.iter_arg_delays().getValue())
    for (size_t i = 0; i < iterArgDelays.size(); i++) {
      auto delay = iterArgDelays[i].dyn_cast<mlir::IntegerAttr>().getInt();
      auto iterArg = op.body().getArgument(i);
      scheduleInfo.mapValueToTime(iterArg, op.getIterTimeVar(), delay);
      scheduleInfo.mapValueToTime(op.getResult(i), op.t_end(), delay);
    }
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(TimeOp op) {
  scheduleInfo.mapValueToTime(
      op.getResult(), scheduleInfo.getRootTimeVar(op.timevar()),
      op.delay() + scheduleInfo.getRootTimeOffset(op.timevar()));
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
      scheduleInfo.mapValueToTime(
          res, scheduleInfo.getRootTimeVar(op.tstart()),
          op.offset().getValue() + scheduleInfo.getRootTimeOffset(op.tstart()) +
              delay);
    } else if (resTy.isa<TimeType>()) {
      scheduleInfo.setAsRootTimeVar(res);
    } else {
      op.emitError()
          << "Failed to create ScheduleInfo. Could not handle return type "
          << resTy << ".";
    }
  }
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(LoadOp op) {
  if (!op.tstart())
    return op.emitError("Failed to create ScheduleInfo. Operation is not "
                        "scheduled yet.");

  if (!scheduleInfo.getRootTimeVar(op.tstart()))
    return op.emitError("Could not find root time var for tstart.");

  scheduleInfo.mapValueToTime(
      op.getResult(), scheduleInfo.getRootTimeVar(op.tstart()),
      op.offset().getValue() + scheduleInfo.getRootTimeOffset(op.tstart()) +
          op.delay());
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(BusRecvOp op) {
  scheduleInfo.mapValueToTime(
      op.getResult(), scheduleInfo.getRootTimeVar(op.tstart()),
      op.offset().getValue() + scheduleInfo.getRootTimeOffset(op.tstart()));
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(DelayOp op) {
  scheduleInfo.mapValueToTime(op.getResult(),
                              scheduleInfo.getRootTimeVar(op.tstart()),
                              op.delay() + op.offset().getValue() +
                                  scheduleInfo.getRootTimeOffset(op.tstart()));
  return success();
}

LogicalResult ScheduleInfoImpl::visitCombOperation(Operation *operation) {
  Value tstart;
  int64_t offset;
  for (auto operand : operation->getOperands())
    if (scheduleInfo.getRootTimeVar(operand)) {
      tstart = scheduleInfo.getRootTimeVar(operand);
      offset = scheduleInfo.getRootTimeOffset(operand);
      break;
    }

  for (auto result : operation->getResults()) {
    if (tstart) {
      scheduleInfo.mapValueToTime(result, tstart, offset);
    } else {
      scheduleInfo.setAsAlwaysValidValue(result);
    }
  }
  return success();
}

LogicalResult ScheduleInfoImpl::visitOp(mlir::arith::ConstantOp op) {
  scheduleInfo.setAsAlwaysValidValue(op.result());
  return success();
}
LogicalResult ScheduleInfoImpl::visitOp(hw::ConstantOp op) {
  scheduleInfo.setAsAlwaysValidValue(op.result());
  return success();
}
