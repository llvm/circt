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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

template <typename T = unsigned>
static SmallVector<std::pair<llvm::StringRef, T>>
parseArrayOfDicts(ArrayAttr attr, StringRef key) {
  SmallVector<std::pair<llvm::StringRef, T>> result;
  for (auto dictAttr : attr.getAsRange<DictionaryAttr>()) {
    auto name = dictAttr.getAs<StringAttr>("name");
    if (!name)
      continue;
    auto intAttr = dictAttr.getAs<IntegerAttr>(key);
    if (intAttr) {
      result.push_back(std::make_pair(name.getValue(), intAttr.getInt()));
      continue;
    }
    auto floatAttr = dictAttr.getAs<FloatAttr>(key);
    if (floatAttr)
      result.push_back(
          std::make_pair(name.getValue(), floatAttr.getValueAsDouble()));
  }
  return result;
}

static void constructProblem(Problem &prob, func::FuncOp func) {
  // set up catch-all operator type with unit latency
  auto unitOpr = prob.getOrInsertOperatorType("unit");
  prob.setLatency(unitOpr, 1);

  // parse additional operator type information attached to the test case
  if (auto attr = func->getAttrOfType<ArrayAttr>("operatortypes")) {
    for (auto &elem : parseArrayOfDicts(attr, "latency")) {
      auto opr = prob.getOrInsertOperatorType(std::get<0>(elem));
      prob.setLatency(opr, std::get<1>(elem));
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
    for (auto &elemArr : parseArrayOfArrays(attr)) {
      assert(elemArr.size() >= 2 && elemArr[0] < ops.size() &&
             elemArr[1] < ops.size());
      Operation *from = ops[elemArr[0]];
      Operation *to = ops[elemArr[1]];
      auto res = prob.insertDependence(std::make_pair(from, to));
      assert(succeeded(res));
      (void)res;
    }
  }
}

static void constructCyclicProblem(CyclicProblem &prob, func::FuncOp func) {
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

static void constructChainingProblem(ChainingProblem &prob, func::FuncOp func) {
  // patch the default operator type to have zero-delay
  auto unitOpr = prob.getOrInsertOperatorType("unit");
  prob.setIncomingDelay(unitOpr, 0.0f);
  prob.setOutgoingDelay(unitOpr, 0.0f);

  // parse operator type info (again) to extract delays
  if (auto attr = func->getAttrOfType<ArrayAttr>("operatortypes")) {
    for (auto &elem : parseArrayOfDicts<float>(attr, "incdelay")) {
      auto opr = prob.getOrInsertOperatorType(std::get<0>(elem));
      prob.setIncomingDelay(opr, std::get<1>(elem));
    }
    for (auto &elem : parseArrayOfDicts<float>(attr, "outdelay")) {
      auto opr = prob.getOrInsertOperatorType(std::get<0>(elem));
      prob.setOutgoingDelay(opr, std::get<1>(elem));
    }
  }
}

static void constructSharedOperatorsProblem(SharedOperatorsProblem &prob,
                                            func::FuncOp func) {
  // parse operator type info (again) to extract optional operator limit
  if (auto attr = func->getAttrOfType<ArrayAttr>("operatortypes")) {
    for (auto &elem : parseArrayOfDicts(attr, "limit")) {
      auto opr = prob.getOrInsertOperatorType(std::get<0>(elem));
      prob.setLimit(opr, std::get<1>(elem));
    }
  }
}

static void emitSchedule(Problem &prob, StringRef attrName,
                         OpBuilder &builder) {
  for (auto *op : prob.getOperations()) {
    unsigned startTime = *prob.getStartTime(op);
    op->setAttr(attrName, builder.getI32IntegerAttr(startTime));
  }
}

//===----------------------------------------------------------------------===//
// (Basic) Problem
//===----------------------------------------------------------------------===//

namespace {
struct TestProblemPass
    : public PassWrapper<TestProblemPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestProblemPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-scheduling-problem"; }
  StringRef getDescription() const override {
    return "Import a schedule encoded as attributes";
  }
};
} // namespace

void TestProblemPass::runOnOperation() {
  auto func = getOperation();

  auto prob = Problem::get(func);
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
    : public PassWrapper<TestCyclicProblemPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCyclicProblemPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-cyclic-problem"; }
  StringRef getDescription() const override {
    return "Import a solution for the cyclic problem encoded as attributes";
  }
};
} // namespace

void TestCyclicProblemPass::runOnOperation() {
  auto func = getOperation();

  auto prob = CyclicProblem::get(func);
  constructProblem(prob, func);
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
// ChainingProblem
//===----------------------------------------------------------------------===//

namespace {
struct TestChainingProblemPass
    : public PassWrapper<TestChainingProblemPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestChainingProblemPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-chaining-problem"; }
  StringRef getDescription() const override {
    return "Import a solution for the chaining problem encoded as attributes";
  }
};
} // namespace

void TestChainingProblemPass::runOnOperation() {
  auto func = getOperation();

  auto prob = ChainingProblem::get(func);
  constructProblem(prob, func);
  constructChainingProblem(prob, func);

  if (failed(prob.check())) {
    func->emitError("problem check failed");
    return signalPassFailure();
  }

  // get schedule and physical start times from the test case
  for (auto *op : prob.getOperations()) {
    if (auto startTimeAttr = op->getAttrOfType<IntegerAttr>("problemStartTime"))
      prob.setStartTime(op, startTimeAttr.getInt());
    if (auto sticAttr = op->getAttrOfType<FloatAttr>("problemStartTimeInCycle"))
      prob.setStartTimeInCycle(op, sticAttr.getValueAsDouble());
  }

  if (failed(prob.verify())) {
    func->emitError("problem verification failed");
    return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// SharedOperatorsProblem
//===----------------------------------------------------------------------===//

namespace {
struct TestSharedOperatorsProblemPass
    : public PassWrapper<TestSharedOperatorsProblemPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSharedOperatorsProblemPass)

  void runOnOperation() override;
  StringRef getArgument() const override {
    return "test-shared-operators-problem";
  }
  StringRef getDescription() const override {
    return "Import a solution for the shared operators problem encoded as "
           "attributes";
  }
};
} // namespace

void TestSharedOperatorsProblemPass::runOnOperation() {
  auto func = getOperation();

  auto prob = SharedOperatorsProblem::get(func);
  constructProblem(prob, func);
  constructSharedOperatorsProblem(prob, func);

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
// ModuloProblem
//===----------------------------------------------------------------------===//

namespace {
struct TestModuloProblemPass
    : public PassWrapper<TestModuloProblemPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestModuloProblemPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-modulo-problem"; }
  StringRef getDescription() const override {
    return "Import a solution for the modulo problem encoded as attributes";
  }
};
} // namespace

void TestModuloProblemPass::runOnOperation() {
  auto func = getOperation();

  auto prob = ModuloProblem::get(func);
  constructProblem(prob, func);
  constructCyclicProblem(prob, func);
  constructSharedOperatorsProblem(prob, func);

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
    : public PassWrapper<TestASAPSchedulerPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestASAPSchedulerPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-asap-scheduler"; }
  StringRef getDescription() const override {
    return "Emit ASAP scheduler's solution as attributes";
  }
};
} // anonymous namespace

void TestASAPSchedulerPass::runOnOperation() {
  auto func = getOperation();

  auto prob = Problem::get(func);
  constructProblem(prob, func);
  assert(succeeded(prob.check()));

  if (failed(scheduleASAP(prob))) {
    func->emitError("scheduling failed");
    return signalPassFailure();
  }

  if (failed(prob.verify())) {
    func->emitError("schedule verification failed");
    return signalPassFailure();
  }

  OpBuilder builder(func.getContext());
  emitSchedule(prob, "asapStartTime", builder);
}

//===----------------------------------------------------------------------===//
// SimplexScheduler
//===----------------------------------------------------------------------===//

namespace {
struct TestSimplexSchedulerPass
    : public PassWrapper<TestSimplexSchedulerPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSimplexSchedulerPass)

  TestSimplexSchedulerPass() = default;
  TestSimplexSchedulerPass(const TestSimplexSchedulerPass &) {}
  Option<std::string> problemToTest{*this, "with", llvm::cl::init("Problem")};
  void runOnOperation() override;
  StringRef getArgument() const override { return "test-simplex-scheduler"; }
  StringRef getDescription() const override {
    return "Emit a simplex scheduler's solution as attributes";
  }
};
} // anonymous namespace

void TestSimplexSchedulerPass::runOnOperation() {
  auto func = getOperation();
  Operation *lastOp = func.getBlocks().front().getTerminator();
  OpBuilder builder(func.getContext());

  if (problemToTest == "Problem") {
    auto prob = Problem::get(func);
    constructProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleSimplex(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    emitSchedule(prob, "simplexStartTime", builder);
    return;
  }

  if (problemToTest == "CyclicProblem") {
    auto prob = CyclicProblem::get(func);
    constructProblem(prob, func);
    constructCyclicProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleSimplex(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    func->setAttr("simplexInitiationInterval",
                  builder.getI32IntegerAttr(*prob.getInitiationInterval()));
    emitSchedule(prob, "simplexStartTime", builder);
    return;
  }

  if (problemToTest == "SharedOperatorsProblem") {
    auto prob = SharedOperatorsProblem::get(func);
    constructProblem(prob, func);
    constructSharedOperatorsProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleSimplex(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    emitSchedule(prob, "simplexStartTime", builder);
    return;
  }

  if (problemToTest == "ModuloProblem") {
    auto prob = ModuloProblem::get(func);
    constructProblem(prob, func);
    constructCyclicProblem(prob, func);
    constructSharedOperatorsProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleSimplex(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    func->setAttr("simplexInitiationInterval",
                  builder.getI32IntegerAttr(*prob.getInitiationInterval()));
    emitSchedule(prob, "simplexStartTime", builder);
    return;
  }

  if (problemToTest == "ChainingProblem") {
    auto prob = ChainingProblem::get(func);
    constructProblem(prob, func);
    constructChainingProblem(prob, func);
    assert(succeeded(prob.check()));

    // get cycle time from the test case
    auto cycleTimeAttr = func->getAttrOfType<FloatAttr>("cycletime");
    assert(cycleTimeAttr);
    float cycleTime = cycleTimeAttr.getValueAsDouble();

    if (failed(scheduleSimplex(prob, lastOp, cycleTime))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    // act like a client that wants to strictly enforce the cycle time
    for (auto *op : prob.getOperations()) {
      float endTimeInCycle =
          *prob.getStartTimeInCycle(op) +
          *prob.getOutgoingDelay(*prob.getLinkedOperatorType(op));
      if (endTimeInCycle > cycleTime) {
        op->emitError("cycle time violated");
        return signalPassFailure();
      }
    }

    emitSchedule(prob, "simplexStartTime", builder);
    for (auto *op : prob.getOperations()) {
      float startTimeInCycle = *prob.getStartTimeInCycle(op);
      op->setAttr("simplexStartTimeInCycle",
                  builder.getF32FloatAttr(startTimeInCycle));
    }
    return;
  }

  llvm_unreachable("Unsupported scheduling problem");
}

//===----------------------------------------------------------------------===//
// LPScheduler
//===----------------------------------------------------------------------===//

#ifdef SCHEDULING_OR_TOOLS

namespace {
struct TestLPSchedulerPass
    : public PassWrapper<TestLPSchedulerPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLPSchedulerPass)

  TestLPSchedulerPass() = default;
  TestLPSchedulerPass(const TestLPSchedulerPass &) {}
  Option<std::string> problemToTest{*this, "with", llvm::cl::init("Problem")};
  void runOnOperation() override;
  StringRef getArgument() const override { return "test-lp-scheduler"; }
  StringRef getDescription() const override {
    return "Emit an LP scheduler's solution as attributes";
  }
};
} // anonymous namespace

void TestLPSchedulerPass::runOnOperation() {
  auto func = getOperation();
  Operation *lastOp = func.getBlocks().front().getTerminator();
  OpBuilder builder(func.getContext());

  if (problemToTest == "Problem") {
    auto prob = Problem::get(func);
    constructProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleLP(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    emitSchedule(prob, "lpStartTime", builder);
    return;
  }

  if (problemToTest == "CyclicProblem") {
    auto prob = CyclicProblem::get(func);
    constructProblem(prob, func);
    constructCyclicProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleLP(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    func->setAttr("lpInitiationInterval",
                  builder.getI32IntegerAttr(*prob.getInitiationInterval()));
    emitSchedule(prob, "lpStartTime", builder);
    return;
  }

  llvm_unreachable("Unsupported scheduling problem");
}

namespace {
struct TestCPSATSchedulerPass
    : public PassWrapper<TestCPSATSchedulerPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCPSATSchedulerPass)

  TestCPSATSchedulerPass() = default;
  TestCPSATSchedulerPass(const TestCPSATSchedulerPass &) {}
  Option<std::string> problemToTest{*this, "with", llvm::cl::init("Problem")};
  void runOnOperation() override;
  StringRef getArgument() const override { return "test-cpsat-scheduler"; }
  StringRef getDescription() const override {
    return "Emit a CPSAT scheduler's solution as attributes";
  }
};
} // anonymous namespace

void TestCPSATSchedulerPass::runOnOperation() {
  auto func = getOperation();
  Operation *lastOp = func.getBlocks().front().getTerminator();
  OpBuilder builder(func.getContext());

  if (problemToTest == "SharedOperatorsProblem") {
    auto prob = SharedOperatorsProblem::get(func);
    constructProblem(prob, func);
    constructSharedOperatorsProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleCPSAT(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    emitSchedule(prob, "cpSatStartTime", builder);
    return;
  }

  llvm_unreachable("Unsupported scheduling problem");
}

#endif // SCHEDULING_OR_TOOLS

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerSchedulingTestPasses() {
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestProblemPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestCyclicProblemPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestChainingProblemPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestSharedOperatorsProblemPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestModuloProblemPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestASAPSchedulerPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestSimplexSchedulerPass>();
  });
#ifdef SCHEDULING_OR_TOOLS
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestLPSchedulerPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestCPSATSchedulerPass>();
  });
#endif
}
} // namespace test
} // namespace circt
