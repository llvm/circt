//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/SynthOps.h"

#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace synth;

namespace {

const char *ir = R"MLIR(
    hw.module private @basic(in %clock : !seq.clock, in %a : i2, in %b : i2, out x : i2, out y: i4) {
      %p = seq.firreg %a clock %clock : i2
      %q = seq.firreg %s clock %clock : i2
      %r = hw.instance "inst" @child(a: %p: i2, b: %b: i2) -> (x: i2)
      %s = synth.aig.and_inv not %p, %q, %r : i2
      %dummy = comb.concat %s, %a : i2, i2
      hw.output %s, %dummy : i2, i4
    } 
    hw.module private @child(in %a : i2, in %b : i2, out x : i2) {
      %r = synth.aig.and_inv not %a, %b {sv.namehint = "child.r"} : i2
      hw.output %r : i2
    }

    hw.module private @nest(in %clock : !seq.clock, in %a: i1, out x: i1) {
      %p = seq.compreg %p, %clock : i1
      hw.output %p: i1
    }

    hw.module private @top(in %clock : !seq.clock, in %a: i1) {
      %0 = hw.instance "inst1" @nest(clock: %clock: !seq.clock, a: %a: i1) -> (x: i1)
      %1 = hw.instance "inst2" @nest(clock: %clock: !seq.clock, a: %a: i1) -> (x: i1)
    }
    )MLIR";

const char *combIR = R"MLIR(
    hw.module private @comb(in %a: i4, in %b: i4, in %c: i4, out x: i4, out y: i4) {
       %0 = comb.add %a, %b : i4
       %1 = comb.mul %0, %c : i4
       hw.output %0, %1 : i4, i4
    }
    )MLIR";

TEST(LongestPathTest, BasicTest) {
  MLIRContext context;
  context.loadDialect<SynthDialect>();
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<seq::SeqDialect>();
  context.loadDialect<comb::CombDialect>();

  // Parse the IR string into a module
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  // Find the 'basic' module
  SymbolTable symbolTable(module.get());
  auto basicModule = symbolTable.lookup<hw::HWModuleOp>("basic");
  ASSERT_TRUE(basicModule);
  auto it = basicModule.getBodyBlock()->begin();
  auto p = cast<seq::FirRegOp>(*it++);
  auto q = cast<seq::FirRegOp>(*it++);
  auto inst = cast<hw::InstanceOp>(*it++);
  Value concat = basicModule.getBodyBlock()->getTerminator()->getOperand(1);
  ModuleAnalysisManager mam(module.get(), nullptr);
  AnalysisManager am(mam);

  // Check global analysis
  {
    LongestPathAnalysis longestPath(module.get(), am,
                                    LongestPathAnalysisOptions(true, false));
    llvm::SmallVector<DataflowPath> results;
    auto closedPath =
        longestPath.getInternalPaths(basicModule.getModuleNameAttr(), results);

    ASSERT_TRUE(succeeded(closedPath));
    auto p = cast<seq::FirRegOp>(*basicModule.getBodyBlock()->begin());
    auto q =
        cast<seq::FirRegOp>(*std::next(basicModule.getBodyBlock()->begin()));

    // There are 4 closed paths: {p[0] -> q[0], p[1] -> q[1], q[0] -> q[0], q[1]
    // -> q[1]}
    DenseSet<std::tuple<Value, size_t, Value, size_t, int64_t>> answerSet({
        {p, 0, q, 0, 3},
        {p, 1, q, 1, 3},
        {q, 0, q, 0, 2},
        {q, 1, q, 1, 2},
    });

    EXPECT_EQ(results.size(), 4u);
    for (auto path : results)
      EXPECT_TRUE(answerSet.erase(
          {path.getStartPoint().value, path.getStartPoint().bitPos,
           path.getEndPointAsObject().value, path.getEndPointAsObject().bitPos,
           path.getDelay()}));

    // Check other API.
    EXPECT_EQ(longestPath.getMaxDelay(concat), 3); // max([3, 3, 0, 0]) = 3
    EXPECT_EQ(longestPath.getAverageMaxDelay(concat),
              2); //  avg([3, 3, 0, 0]) = ceil((3+3+0+0)/4) = 2

    // Check history.
    auto history = results[0].getHistory();
    SmallVector<DebugPoint> points;
    for (auto &point : history)
      points.push_back(point);

    EXPECT_EQ(points.size(), 3u);
    EXPECT_EQ(points[0].comment, "output port"); // inst.r
    EXPECT_EQ(points[1].comment, "namehint");    // child.r
    EXPECT_EQ(points[2].comment, "input port");  // inst.a

    LongestPathAnalysis longestPathWithoutDebug(module.get(), am);
    results.clear();
    EXPECT_TRUE(succeeded(longestPathWithoutDebug.getInternalPaths(
        basicModule.getModuleNameAttr(), results)));
    EXPECT_EQ(results.size(), 4u);
    // No history must be recorded.
    for (auto path : results)
      EXPECT_TRUE(path.getHistory().isEmpty());
  }

  // Check local paths
  {
    auto nestedAm = am.nest(basicModule);
    LongestPathAnalysis longestPath(basicModule, nestedAm,
                                    LongestPathAnalysisOptions(true, false));
    llvm::SmallVector<DataflowPath> results;
    auto closedPath =
        longestPath.getInternalPaths(basicModule.getModuleNameAttr(), results);

    ASSERT_TRUE(succeeded(closedPath));
    // In local analysis, instance results are treated as start point.
    // There are 6 closed paths:
    // {p[0] -> q[0], p[1] -> q[1], q[0] -> q[0], q[1] -> q[1], inst.r[0] ->
    // q[0], inst.r[1] -> q[1]}
    DenseSet<std::tuple<Value, size_t, Value, size_t, int64_t>> answerSet({
        {p, 0, q, 0, 2},
        {p, 1, q, 1, 2},
        {q, 0, q, 0, 2},
        {q, 1, q, 1, 2},
        {inst.getResult(0), 0, q, 0, 2},
        {inst.getResult(0), 1, q, 1, 2},
    });

    EXPECT_EQ(results.size(), 6u);
    for (auto path : results)
      EXPECT_TRUE(answerSet.erase(
          {path.getStartPoint().value, path.getStartPoint().bitPos,
           path.getEndPointAsObject().value, path.getEndPointAsObject().bitPos,
           path.getDelay()}));

    EXPECT_EQ(longestPath.getMaxDelay(concat), 2); // max([2, 2, 0, 0]) = 3
    EXPECT_EQ(longestPath.getAverageMaxDelay(concat),
              1); // avg([2, 2, 0, 0]) = 1
  }
}

TEST(LongestPathTest, ElaborationTest) {
  MLIRContext context;
  context.loadDialect<SynthDialect>();
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<seq::SeqDialect>();
  context.loadDialect<comb::CombDialect>();

  // Parse the IR string into a module
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  // Find the 'top' module
  SymbolTable symbolTable(module.get());
  auto basicModule = symbolTable.lookup<hw::HWModuleOp>("top");
  ASSERT_TRUE(basicModule);
  ModuleAnalysisManager mam(module.get(), nullptr);
  AnalysisManager am(mam);

  LongestPathAnalysis longestPath(module.get(), am,
                                  LongestPathAnalysisOptions(true, false));
  llvm::SmallVector<DataflowPath> elaboratedPaths, unelaboratedPaths;
  auto elaborated =
      longestPath.getInternalPaths(basicModule.getModuleNameAttr(),
                                   elaboratedPaths, /*elaboratePaths=*/true);
  auto unelaborated = longestPath.getInternalPaths(
      basicModule.getModuleNameAttr(), unelaboratedPaths,
      /*elaboratePaths=*/false);

  // In unelaborated representation, there are 1 paths: {p -> p}.
  // In elaborated representation, there are 2 paths: {inst1.p -> inst1.p,
  // inst2.p -> inst2.p}.
  EXPECT_TRUE(succeeded(unelaborated));
  EXPECT_EQ(unelaboratedPaths.size(), 1u);
  EXPECT_TRUE(succeeded(elaborated));
  EXPECT_EQ(elaboratedPaths.size(), 2u);

  EXPECT_EQ(unelaboratedPaths[0].getStartPoint().instancePath.size(), 0u);
  EXPECT_EQ(unelaboratedPaths[0].getRoot().getModuleName().compare("nest"), 0);

  for (auto &path : elaboratedPaths) {
    EXPECT_EQ(path.getStartPoint().instancePath.size(), 1u);
    EXPECT_EQ(path.getRoot().getModuleName().compare("top"), 0);
    EXPECT_TRUE(
        path.getStartPoint().instancePath.leaf().getInstanceName().starts_with(
            "inst"));
  }

  EXPECT_NE(elaboratedPaths[0]
                .getStartPoint()
                .instancePath.leaf()
                .getInstanceNameAttr(),
            elaboratedPaths[1]
                .getStartPoint()
                .instancePath.leaf()
                .getInstanceNameAttr());
}

TEST(LongestPathTest, Incremental) {
  MLIRContext context;
  context.loadDialect<SynthDialect>();
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  // Register prerequisite passes needed to convert Comb IR to Synth
  circt::synth::registerSynthAnalysisPrerequisitePasses();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(combIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto basicModule = symbolTable.lookup<hw::HWModuleOp>("comb");
  ASSERT_TRUE(basicModule);
  ModuleAnalysisManager mam(module.get(), nullptr);
  AnalysisManager am(mam);

  // Create incremental analysis that lazily computes delays
  IncrementalLongestPathAnalysis longestPath(basicModule, am);
  auto it = basicModule.getBodyBlock()->begin();
  auto add = cast<comb::AddOp>(*it++);
  auto mul = cast<comb::MulOp>(*it++);

  // Compute delay for add operation (bit 1)
  // This should trigger analysis and mark the add operation as analyzed
  auto delayAdd = longestPath.getMaxDelay(add.getResult(), 1);
  ASSERT_TRUE(succeeded(delayAdd));
  ASSERT_EQ(*delayAdd, 5);
  // Once analyzed, the add operation cannot be safely mutated since it's been
  // marked as analyzed.
  ASSERT_TRUE(!longestPath.isOperationValidToMutate(add));
  // Mul is still safe to mutate since it hasn't been analyzed yet
  ASSERT_TRUE(longestPath.isOperationValidToMutate(mul));

  // Now compute delay for mul operation (bit 1)
  // This depends on the add result, so it includes add's delay
  auto delayMul = longestPath.getMaxDelay(mul.getResult(), 1);
  ASSERT_TRUE(succeeded(delayMul));
  ASSERT_EQ(*delayMul, 8);
  // After analysis, mul can no longer be safely mutated
  ASSERT_TRUE(!longestPath.isOperationValidToMutate(mul));
}

TEST(LongestPathTest, OpenPaths) {
  MLIRContext context;
  context.loadDialect<SynthDialect>();
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<seq::SeqDialect>();
  context.loadDialect<comb::CombDialect>();

  // Parse the IR string into a module
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  // Find the 'basic' module
  SymbolTable symbolTable(module.get());
  auto basicModule = symbolTable.lookup<hw::HWModuleOp>("basic");
  ASSERT_TRUE(basicModule);
  ModuleAnalysisManager mam(module.get(), nullptr);
  AnalysisManager am(mam);

  LongestPathAnalysis longestPath(module.get(), am,
                                  LongestPathAnalysisOptions(true, false));

  // Test getOpenPathsFromInputPortsToInternal (input-to-register paths)
  llvm::SmallVector<DataflowPath> inputToInternalPaths;
  auto inputToInternalResult = longestPath.getOpenPathsFromInputPortsToInternal(
      basicModule.getModuleNameAttr(), inputToInternalPaths);
  ASSERT_TRUE(succeeded(inputToInternalResult));

  // There should be paths from input ports (a, b) to registers (p, q)
  // a[0] -> p[0], a[1] -> p[1], b[0] -> q[0], b[1] -> q[1]
  EXPECT_GT(inputToInternalPaths.size(), 0u);

  // Verify that these are input-to-register paths
  for (auto &path : inputToInternalPaths) {
    // Start point should be an input port (block argument)
    EXPECT_TRUE(isa<BlockArgument>(path.getStartPoint().value));
    // End point should be a register (Object, not OutputPort)
    EXPECT_TRUE(std::holds_alternative<Object>(path.getEndPoint()));
  }

  // Test getOpenPathsFromInternalToOutputPorts (register-to-output paths)
  llvm::SmallVector<DataflowPath> internalToOutputPaths;
  auto internalToOutputResult =
      longestPath.getOpenPathsFromInternalToOutputPorts(
          basicModule.getModuleNameAttr(), internalToOutputPaths);
  ASSERT_TRUE(succeeded(internalToOutputResult));

  // There should be paths from registers to output ports
  EXPECT_GT(internalToOutputPaths.size(), 0u);

  // Verify that these are register-to-output paths
  for (auto &path : internalToOutputPaths) {
    // End point should be an output port
    EXPECT_TRUE(
        std::holds_alternative<DataflowPath::OutputPort>(path.getEndPoint()));
  }
}

} // namespace
