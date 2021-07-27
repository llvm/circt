//===- TestPasses.cpp - Test passes for the scheduling infrastructure -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements test passes for scheduling problems and algorithms.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;
using namespace circt::scheduling;

//===----------------------------------------------------------------------===//
// Construction helper methods
//===----------------------------------------------------------------------===//

static SmallVector<SmallVector<unsigned>> parseArrayOfArrays(ArrayAttr attr) {
  SmallVector<SmallVector<unsigned>> result;
  for (auto elemArr : attr.getAsRange<ArrayAttr>()) {
    result.emplace_back();
    for (auto elem : elemArr.getAsRange<IntegerAttr>())
      result.back().push_back(elem.getInt());
  }
  return result;
}

static void constructProblem(Problem &prob, FuncOp func) {
  // set up catch-all operator type with unit latency
  auto unitOpr = prob.getOrInsertOperatorType("unit");
  prob.setLatency(unitOpr, 1);

  // parse additional operator type information attached to the test case
  if (auto attr = func->getAttrOfType<ArrayAttr>("operatortypes")) {
    for (auto oprAttr : attr.getAsRange<DictionaryAttr>()) {
      auto name = oprAttr.getAs<StringAttr>("name");
      auto latency = oprAttr.getAs<IntegerAttr>("latency");
      if (!(name && latency))
        continue;

      auto opr = prob.getOrInsertOperatorType(name.getValue());
      prob.setLatency(opr, latency.getInt());
    }
  }

  // construct problem (consider only the first block)
  for (auto &op : func.getBlocks().front().getOperations()) {
    prob.insertOperation(&op);

    if (auto oprRefAttr = op.getAttrOfType<StringAttr>("opr")) {
      auto opr = prob.getOrInsertOperatorType(oprRefAttr.getValue());
      prob.setLinkedOperatorType(&op, opr);
    } else {
      prob.setLinkedOperatorType(&op, unitOpr);
    }
  }

  // parse auxiliary dependences in the testcase, encoded as an array of arrays
  // of integer attributes
  if (auto attr = func->getAttrOfType<ArrayAttr>("auxdeps")) {
    auto &ops = prob.getOperations();
    unsigned nOps = ops.size();
    for (auto &elemArr : parseArrayOfArrays(attr)) {
      assert(elemArr.size() >= 2 && elemArr[0] < nOps && elemArr[1] < nOps);
      Operation *from = ops[elemArr[0]];
      Operation *to = ops[elemArr[1]];
      auto res = prob.insertDependence(std::make_pair(from, to));
      assert(succeeded(res));
    }
  }
}

static void constructCyclicProblem(CyclicProblem &prob, FuncOp func) {
  constructProblem(prob, func);

  // parse auxiliary dependences in the testcase (again), in order to set the
  // optional distance in the cyclic problem
  if (auto attr = func->getAttrOfType<ArrayAttr>("auxdeps")) {
    auto &ops = prob.getOperations();
    for (auto &elemArr : parseArrayOfArrays(attr)) {
      if (elemArr.size() < 3)
        continue; // skip this dependence, rather than setting the default value
      Operation *from = ops[elemArr[0]];
      Operation *to = ops[elemArr[1]];
      unsigned dist = elemArr[2];
      prob.setDistance(std::make_pair(from, to), dist);
    }
  }
}

//===----------------------------------------------------------------------===//
// (Basic) Problem
//===----------------------------------------------------------------------===//

namespace {
struct TestProblemPass : public PassWrapper<TestProblemPass, FunctionPass> {
  void runOnFunction() override;
};
} // namespace

void TestProblemPass::runOnFunction() {
  auto func = getFunction();

  Problem prob(func);
  constructProblem(prob, func);

  if (failed(prob.check())) {
    func->emitError("problem check failed");
    return signalPassFailure();
  }

  // get schedule from the test case
  for (auto *op : prob.getOperations())
    if (auto startTimeAttr = op->getAttrOfType<IntegerAttr>("problemStartTime"))
      prob.setStartTime(op, startTimeAttr.getInt());

  if (failed(prob.verify())) {
    func->emitError("problem verification failed");
    return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// CyclicProblem
//===----------------------------------------------------------------------===//

namespace {
struct TestCyclicProblemPass
    : public PassWrapper<TestCyclicProblemPass, FunctionPass> {
  void runOnFunction() override;
};
} // namespace

void TestCyclicProblemPass::runOnFunction() {
  auto func = getFunction();

  CyclicProblem prob(func);
  constructCyclicProblem(prob, func);

  if (failed(prob.check())) {
    func->emitError("problem check failed");
    return signalPassFailure();
  }

  // get II from the test case
  if (auto attr = func->getAttrOfType<IntegerAttr>("problemInitiationInterval"))
    prob.setInitiationInterval(attr.getInt());

  // get schedule from the test case
  for (auto *op : prob.getOperations())
    if (auto startTimeAttr = op->getAttrOfType<IntegerAttr>("problemStartTime"))
      prob.setStartTime(op, startTimeAttr.getInt());

  if (failed(prob.verify())) {
    func->emitError("problem verification failed");
    return signalPassFailure();
  }
}

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

  Problem prob(func);
  constructProblem(prob, func);

  if (failed(prob.check())) {
    func->emitError("problem check failed");
    return signalPassFailure();
  }

  if (failed(scheduleASAP(prob))) {
    func->emitError("scheduling failed");
    return signalPassFailure();
  }

  if (failed(prob.verify())) {
    func->emitError("schedule verification failed");
    return signalPassFailure();
  }

  OpBuilder builder(func.getContext());
  for (auto *op : prob.getOperations()) {
    unsigned startTime = *prob.getStartTime(op);
    op->setAttr("asapStartTime", builder.getI32IntegerAttr(startTime));
  }
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerSchedulingTestPasses() {
  PassRegistration<TestProblemPass> problemTester(
      "test-scheduling-problem", "Import a schedule encoded as attributes");
  PassRegistration<TestCyclicProblemPass> cyclicProblemTester(
      "test-cyclic-problem",
      "Import a solution for the cyclic problem encoded as attributes");
  PassRegistration<TestASAPSchedulerPass> asapTester(
      "test-asap-scheduler", "Emit ASAP scheduler's solution as attributes");
}
} // namespace test
} // namespace circt
