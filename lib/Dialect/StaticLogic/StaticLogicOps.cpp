//===- StaticLogicOps.h - StaticLogic MLIR Operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the StaticLogic ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/StaticLogic/StaticLogic.h"

#include "circt/Dialect/Scheduling/Scheduler.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace circt;
using namespace circt::staticlogic;

using circt::sched::OperatorInfoAttr;

#define GET_OP_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogic.cpp.inc"

void StaticLogicDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/StaticLogic/StaticLogic.cpp.inc"
      >();
}

LogicalResult
PipelineOp::constructSchedulingProblem(sched::SchedulerBase &scheduler) {
  auto &blockToSchedule = getRegion().getBlocks().front();
  auto &operationsToSchedule = blockToSchedule.getOperations();

  auto unitLatencyOperator = OperatorInfoAttr::get(getContext(), "unit", 1);
  StringRef defaultAttrName = OperatorInfoAttr::getDefaultAttributeName();

  for (Operation &op : operationsToSchedule) {
    OperatorInfoAttr opr;
    if (op.hasAttrOfType<OperatorInfoAttr>(defaultAttrName))
      opr = op.getAttrOfType<OperatorInfoAttr>(defaultAttrName);
    else
      opr = unitLatencyOperator; // fallback if attribute is not present

    if (failed(scheduler.registerOperation(&op)) ||
        failed(scheduler.registerOperators(&op, opr)))
      return failure();
  }

  for (Operation &op : operationsToSchedule) {
    for (Value operand : op.getOperands()) {
      if (operand.isa<BlockArgument>())
        continue; // block arguments are always available w.r.t the schedule
      Operation *operandOp = operand.getDefiningOp();
      if (failed(scheduler.registerDependence(operandOp, 0, &op, 0, 0)))
        return failure();
    }
  }

  return success();
}
