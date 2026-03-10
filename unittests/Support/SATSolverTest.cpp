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

TEST(SatSolverTest, UnitClauseAndAssumption) {
  auto solver = createZ3SATSolver();
  if (!solver)
    GTEST_SKIP() << "Z3 is not available in this build.";

  // (x1) AND (!x1 OR x2)
  // This should be satisfiable with x1=true, x2=true.
  // But if we assume !x1, it should become unsatisfiable.
  solver->add(1);
  solver->add(0);
  solver->add(-1);
  solver->add(2);
  solver->add(0);

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));

  solver->assume(-1);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, AssumptionsAreScopedToSolve) {
  auto solver = createZ3SATSolver();
  if (!solver)
    GTEST_SKIP() << "Z3 is not available in this build.";

  solver->add(1);
  solver->add(0);
  solver->add(2);
  solver->add(0);

  solver->assume(-1);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));
}
