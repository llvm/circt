//===- TestPasses.cpp - Test passes for scheduling algorithms -===============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements test passes for scheduling algorithms.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/ASAPScheduler.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;
using namespace circt::scheduling;

//===----------------------------------------------------------------------===//
// ASAPScheduler
//===----------------------------------------------------------------------===//

namespace {
struct TestASAPSchedulerPass
    : public PassWrapper<TestASAPSchedulerPass, FunctionPass> {
  void runOnFunction() override;
};
} // anonymous namespace

void TestASAPSchedulerPass::runOnFunction() {
  auto func = getFunction();
  OpBuilder builder(func.getContext());

  // instantiate algorithm implementation
  ASAPScheduler scheduler(func);

  // set up catch-all operator type with unit latency
  auto unitOpr = scheduler.getOrInsertOperatorType("unit");
  scheduler.setLatency(unitOpr, 1);

  // parse additional operator type information attached to the test case
  if (auto attr = func->getAttrOfType<ArrayAttr>("operatortypes")) {
    for (auto oprAttr : attr.getAsRange<DictionaryAttr>()) {
      auto name = oprAttr.getAs<StringAttr>("name");
      auto latency = oprAttr.getAs<IntegerAttr>("latency");
      if (!(name && latency))
        continue;

      auto opr = scheduler.getOrInsertOperatorType(name.getValue());
      scheduler.setLatency(opr, latency.getInt());
    }
  }

  // construct problem (consider only the first block)
  auto &operationsToSchedule = func.getBlocks().front().getOperations();
  for (Operation &op : operationsToSchedule) {
    scheduler.insertOperation(&op);

    if (auto oprRefAttr = op.getAttrOfType<StringAttr>("opr")) {
      scheduler.setLinkedOperatorType(
          &op, scheduler.getOrInsertOperatorType(oprRefAttr.getValue()));
    } else {
      scheduler.setLinkedOperatorType(&op, unitOpr);
    }
  }

  if (failed(scheduler.schedule())) {
    func->emitError("scheduling failed");
    return signalPassFailure();
  }

  if (failed(scheduler.verify())) {
    func->emitError("schedule verification failed");
    return signalPassFailure();
  }

  for (Operation &op : operationsToSchedule) {
    unsigned startTime = *scheduler.getStartTime(&op);
    op.emitRemark("start time = " + std::to_string(startTime));
  }
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerSchedulingTestPasses() {
  PassRegistration<TestASAPSchedulerPass> asapTester(
      "test-asap-scheduler", "Emit ASAP scheduler's solution as remarks");
}
} // namespace test
} // namespace circt
