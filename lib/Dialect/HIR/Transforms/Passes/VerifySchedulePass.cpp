//=========- VerifySchedulePass.cpp - Verify schedule of HIR dialect-------===//
//
// This file implements the HIR schedule verifier.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/Analysis/ScheduleAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

#include "../PassDetails.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <list>
#include <stack>

using namespace circt;
using namespace hir;
using namespace llvm;
class VerifySchedulePass : public hir::VerifyScheduleBase<VerifySchedulePass> {
public:
  void runOnOperation() override;

private:
  LogicalResult verifyOp(CallOp op);
  LogicalResult verifyOp(ForOp op);
  LogicalResult verifyOp(WhileOp op);
  LogicalResult verifyOp(NextIterOp op);
  LogicalResult verifyCombOp(Operation *);
  LogicalResult verifyOpWithAllOperandsValidAtStartTime(ScheduledOp);
  LogicalResult verifyOperation(Operation *);

private:
  ScheduleInfo *scheduleInfo;
};

void VerifySchedulePass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  auto schedInfo = ScheduleInfo::createScheduleInfo(funcOp).getValue();
  this->scheduleInfo = &schedInfo;
  funcOp.walk([this](Operation *operation) {
    if (failed(verifyOperation(operation)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
}

LogicalResult VerifySchedulePass::verifyOperation(Operation *operation) {
  if (auto op = dyn_cast<CallOp>(operation))
    return verifyOp(op);
  if (auto op = dyn_cast<ForOp>(operation))
    return verifyOp(op);
  if (auto op = dyn_cast<WhileOp>(operation))
    return verifyOp(op);
  if (auto op = dyn_cast<NextIterOp>(operation))
    return verifyOp(op);
  if (isa<comb::CombDialect>(operation->getDialect()))
    return verifyCombOp(operation);
  if (auto op = dyn_cast<hir::ScheduledOp>(operation))
    return verifyOpWithAllOperandsValidAtStartTime(operation);
  return success();
}

LogicalResult VerifySchedulePass::verifyOp(CallOp op) {
  for (size_t i = 0; i < op.getFuncType().getInputTypes().size(); i++) {
    auto ty = op.getFuncType().getInputTypes()[i];
    auto operand = op.operands()[i];
    auto attr = op.getFuncType().getInputAttrs()[i];
    if (helper::isBuiltinSizedType(ty)) {
      auto offset = helper::extractDelayFromDict(attr) + op.offset();
      if (!scheduleInfo->isValidAtTime(operand, op.tstart(), offset))
        return op.emitError("Error in scheduling of operand.")
                   .attachNote(operand.getLoc())
               << "Operand defined here.";
    }
  }
  return success();
}

LogicalResult VerifySchedulePass::verifyOp(ForOp op) {
  if (op.iter_arg_delays())
    for (size_t i = 0; i < op.iter_arg_delays()->size(); i++) {
      auto offset =
          op.iter_arg_delays().getValue()[i].dyn_cast<IntegerAttr>().getInt() +
          op.offset();
      if (!scheduleInfo->isValidAtTime(op.iter_args()[i], op.tstart(), offset))
        return op.emitError("Error in scheduling of iter_arg.")
                   .attachNote(op.iter_args()[i].getLoc())
               << "iter_arg defined here.";
    }
  return success();
}

LogicalResult VerifySchedulePass::verifyOp(WhileOp op) {
  if (op.iter_arg_delays())
    for (size_t i = 0; i < op.iter_arg_delays()->size(); i++) {
      auto offset =
          op.iter_arg_delays().getValue()[i].dyn_cast<IntegerAttr>().getInt() +
          op.offset();
      if (!scheduleInfo->isValidAtTime(op.iter_args()[i], op.tstart(), offset))
        return op.emitError("Error in scheduling of iter_arg.")
                   .attachNote(op.iter_args()[i].getLoc())
               << "iter_arg defined here.";
    }
  return success();
}

LogicalResult VerifySchedulePass::verifyOp(NextIterOp op) {
  // FIXME: Verify this using the parent for-op delays.
  return success();
}

LogicalResult VerifySchedulePass::verifyCombOp(Operation *operation) {
  Value tstart;
  int64_t offset;
  for (auto operand : operation->getOperands()) {
    if (scheduleInfo->isAlwaysValid(operand))
      continue;
    if (tstart) {
      if (scheduleInfo->isValidAtTime(operand, tstart, offset))
        continue;
      operation->emitError("Error in scheduling of operand.")
              .attachNote(operand.getLoc())
          << "Operand defined here.";
    }
    tstart = scheduleInfo->getRootTimeVar(operand);
    if (!tstart) {
      return operation->emitError(
          "Could not find root time var for this ssa var.");
    }
    offset = scheduleInfo->getRootTimeOffset(operand);
  }
  return success();
}

LogicalResult
VerifySchedulePass::verifyOpWithAllOperandsValidAtStartTime(ScheduledOp op) {
  for (auto operand : op->getOperands()) {
    if (helper::isBuiltinSizedType(operand.getType())) {
      if (!scheduleInfo->isValidAtTime(operand, op.getTimeVar(),
                                       op.getTimeOffset()))
        return op.emitError("Error in scheduling of operand.")
                   .attachNote(operand.getLoc())
               << "Operand defined here. ";
    }
  }
  return success();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createVerifySchedulePass() {
  return std::make_unique<VerifySchedulePass>();
}
} // namespace hir
} // namespace circt
