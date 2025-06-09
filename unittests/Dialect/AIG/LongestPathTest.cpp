//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "circt/Dialect/AIG/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace aig;

namespace {

const char *ir = R"MLIR(
    hw.module private @basic(in %clock : !seq.clock, in %a : i2, in %b : i2, out x : i2, out y: i4) {
      %p = seq.firreg %a clock %clock : i2
      %q = seq.firreg %s clock %clock : i2
      %r = hw.instance "inst" @child(a: %p: i2, b: %b: i2) -> (x: i2)
      %s = aig.and_inv not %p, %q, %r : i2
      %dummy = comb.concat %s, %a : i2, i2
      hw.output %s, %dummy : i2, i4
    } 
    hw.module private @child(in %a : i2, in %b : i2, out x : i2) {
      %r = aig.and_inv not %a, %b {sv.namehint = "child.r"} : i2
      hw.output %r : i2
    }
    )MLIR";

TEST(LongestPathTest, BasicTest) {
  MLIRContext context;
  context.loadDialect<AIGDialect>();
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
    LongestPathAnalysis longestPath(module.get(), am, {true});
    llvm::SmallVector<DataflowPath> results;
    auto closedPath =
        longestPath.getClosedPaths(basicModule.getModuleNameAttr(), results);

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
          {path.getFanIn().value, path.getFanIn().bitPos,
           path.getFanOut().value, path.getFanOut().bitPos, path.getDelay()}));

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
    EXPECT_TRUE(succeeded(longestPathWithoutDebug.getClosedPaths(
        basicModule.getModuleNameAttr(), results)));
    EXPECT_EQ(results.size(), 4u);
    // No history must be recorded.
    for (auto path : results)
      EXPECT_TRUE(path.getHistory().isEmpty());
  }

  // Check local paths
  {
    auto nestedAm = am.nest(basicModule);
    LongestPathAnalysis longestPath(basicModule, nestedAm, {true});
    llvm::SmallVector<DataflowPath> results;
    auto closedPath =
        longestPath.getClosedPaths(basicModule.getModuleNameAttr(), results);

    ASSERT_TRUE(succeeded(closedPath));
    // In local analysis, instance results are treated as fanIn.
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
          {path.getFanIn().value, path.getFanIn().bitPos,
           path.getFanOut().value, path.getFanOut().bitPos, path.getDelay()}));

    EXPECT_EQ(longestPath.getMaxDelay(concat), 2); // max([2, 2, 0, 0]) = 3
    EXPECT_EQ(longestPath.getAverageMaxDelay(concat),
              1); // avg([2, 2, 0, 0]) = 1
  }
}

} // namespace
