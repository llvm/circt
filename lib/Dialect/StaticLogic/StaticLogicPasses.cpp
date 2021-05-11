//===- StaticLogicPasses.cpp - StaticLogic transformations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the StaticLogic passes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/StaticLogic/StaticLogicPasses.h"

#include "circt/Dialect/Scheduling/Algorithms/ASAPScheduler.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace circt {
namespace staticlogic {
#define GEN_PASS_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogicPasses.h.inc"
} // namespace staticlogic
} // namespace circt

using namespace circt;
using namespace circt::staticlogic;

namespace {
struct SchedulePipelinePass
    : public SchedulePipelineBase<SchedulePipelinePass> {
  void runOnFunction() override;
  LogicalResult schedule(PipelineOp pipeline);
};
} // anonymous namespace

void SchedulePipelinePass::runOnFunction() {
  auto op = getOperation();
  for (auto pl : op.getOps<PipelineOp>()) {
    if (failed(schedule(pl)))
      return signalPassFailure();
  }
}

LogicalResult SchedulePipelinePass::schedule(PipelineOp pipeline) {
  sched::ASAPScheduler scheduler;
  if (failed(pipeline.constructSchedulingProblem(scheduler)) ||
      failed(scheduler.schedule()))
    return failure();

  OpBuilder builder(&getContext());
  pipeline->walk([&](Operation *scheduledOp) {
    auto startTime = scheduler.getStartTime(scheduledOp);
    if (startTime)
      scheduledOp->setAttr("startTime", builder.getIndexAttr(*startTime));
  });
  return success();
}

std::unique_ptr<mlir::Pass> circt::staticlogic::createSchedulePipelinePass() {
  return std::make_unique<SchedulePipelinePass>();
}
