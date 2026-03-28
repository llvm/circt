//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SATSolver.h"

#include "gtest/gtest.h"

using namespace circt;
namespace {
struct HeapNode {
  double score = 0.0;
};

struct HeapNodeScore {
  double operator()(const HeapNode &node) const { return node.score; }
};
} // namespace

TEST(SatSolverTest, IndexedMaxHeapPopsInDescendingScoreOrder) {
  llvm::SmallVector<HeapNode, 4> nodes = {{1.0}, {5.0}, {3.0}, {4.0}};
  IndexedMaxHeap<HeapNode, HeapNodeScore> heap(nodes);

  for (unsigned i = 0; i < nodes.size(); ++i)
    heap.insert(i);

  EXPECT_EQ(1u, heap.pop());
  EXPECT_EQ(3u, heap.pop());
  EXPECT_EQ(2u, heap.pop());
  EXPECT_EQ(0u, heap.pop());
  EXPECT_TRUE(heap.empty());
}

TEST(SatSolverTest, IndexedMaxHeapIncreaseReordersExistingEntry) {
  llvm::SmallVector<HeapNode, 4> nodes = {{1.0}, {2.0}, {3.0}};
  IndexedMaxHeap<HeapNode, HeapNodeScore> heap(nodes);

  for (unsigned i = 0; i < nodes.size(); ++i)
    heap.insert(i);

  nodes[0].score = 10.0;
  heap.increase(0);

  EXPECT_EQ(0u, heap.pop());
  EXPECT_EQ(2u, heap.pop());
  EXPECT_EQ(1u, heap.pop());
}

TEST(SatSolverTest, IndexedMaxHeapAvoidsDuplicateInsertions) {
  llvm::SmallVector<HeapNode, 2> nodes = {{1.0}, {2.0}};
  IndexedMaxHeap<HeapNode, HeapNodeScore> heap(nodes);

  heap.insert(0);
  heap.insert(1);
  heap.insert(1);

  EXPECT_EQ(1u, heap.pop());
  EXPECT_EQ(0u, heap.pop());
  EXPECT_TRUE(heap.empty());
}

TEST(SatSolverTest, IndexedMaxHeapClearRemovesEntriesAndAllowsReuse) {
  llvm::SmallVector<HeapNode, 4> nodes = {{1.0}, {5.0}, {3.0}};
  IndexedMaxHeap<HeapNode, HeapNodeScore> heap(nodes);

  for (unsigned i = 0; i < nodes.size(); ++i)
    heap.insert(i);

  heap.clear();

  EXPECT_TRUE(heap.empty());
  for (unsigned i = 0; i < nodes.size(); ++i)
    EXPECT_FALSE(heap.contains(i));

  nodes[0].score = 7.0;
  heap.insert(0);
  heap.insert(2);

  EXPECT_EQ(0u, heap.pop());
  EXPECT_EQ(2u, heap.pop());
  EXPECT_TRUE(heap.empty());
}

// ==----------------------------------------------------------------------===//
// Z3 solver tests
// ==----------------------------------------------------------------------===//

TEST(SatSolverTest, UnitClauseAndAssumption) {
  // -DLLVM_ENABLE_Z3_SOLVER=ON is required to run this test.
  auto solver = createZ3SATSolver();
  if (!solver)
    GTEST_SKIP() << "Z3 is not available in this build.";

  // (x1) AND (!x1 OR x2)
  // This should be satisfiable with x1=true, x2=true.
  // But if we assume !x1, it should become unsatisfiable.
  solver->addClause({1});
  solver->addClause({-1, 2});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));

  solver->assume(-1);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, AssumptionsAreScopedToSolve) {
  // -DLLVM_ENABLE_Z3_SOLVER=On is required to run this test.
  auto solver = createZ3SATSolver();
  if (!solver)
    GTEST_SKIP() << "Z3 is not available in this build.";

  solver->addClause({1});
  solver->addClause({2});

  solver->assume(-1);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));
}

TEST(SatSolverTest, CadicalConflictLimitCanBeConfigured) {
  CadicalSATSolverOptions options;
  options.config = CadicalSATSolverOptions::CadicalSolverConfig::Plain;
  auto solver = createCadicalSATSolver(options);
  if (!solver)
    GTEST_SKIP() << "CaDiCaL is not available in this build.";

  // Build a small pigeonhole instance that requires search to prove UNSAT:
  // 5 pigeons placed into 4 holes.
  constexpr int pigeons = 5;
  constexpr int holes = 4;
  auto var = [=](int pigeon, int hole) { return pigeon * holes + hole + 1; };

  solver->reserveVars(pigeons * holes);

  // Each pigeon must be in at least one hole.
  for (int pigeon = 0; pigeon < pigeons; ++pigeon)
    solver->addClause(
        {var(pigeon, 0), var(pigeon, 1), var(pigeon, 2), var(pigeon, 3)});

  // No hole can contain two pigeons.
  for (int hole = 0; hole < holes; ++hole) {
    for (int lhs = 0; lhs < pigeons; ++lhs) {
      for (int rhs = lhs + 1; rhs < pigeons; ++rhs)
        solver->addClause({-var(lhs, hole), -var(rhs, hole)});
    }
  }

  // Fail with a very low conflict limit, but succeed with no conflict limit.
  solver->setConflictLimit(0);
  EXPECT_EQ(IncrementalSATSolver::kUNKNOWN, solver->solve());
  solver->setConflictLimit(-1);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}
