//===- LPSchedulers.cpp - Schedulers using external LP solvers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of linear programming-based schedulers using external solvers
// via OR-Tools.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Operation.h"

#include "ortools/linear_solver/linear_solver.h"

using namespace circt;
using namespace circt::scheduling;
using namespace operations_research;

LogicalResult scheduling::scheduleLP(Problem &prob, Operation *lastOp) {
  Operation *containingOp = prob.getContainingOp();

  MPSolver::OptimizationProblemType problemType;
  if (!MPSolver::ParseSolverType("GLOP_LINEAR_PROGRAMMING", &problemType) ||
      !MPSolver::SupportsProblemType(problemType))
    return containingOp->emitError("Solver is unvailable");

  MPSolver solver("Problem", problemType);
  double infinity = solver.infinity();

  // Create start time variables.
  DenseMap<Operation *, MPVariable *> vars;
  unsigned i = 0;
  for (auto *op : prob.getOperations()) {
    vars[op] = solver.MakeNumVar(0, infinity, (Twine("t_") + Twine(i)).str());
    ++i;
  }

  // The objective is to minimize the start time of the last operation.
  MPObjective *objective = solver.MutableObjective();
  objective->SetCoefficient(vars[lastOp], 1);
  objective->SetMinimization();

  // Construct a linear constraint for each dependence.
  for (auto *op : prob.getOperations())
    for (auto dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      Operation *dst = dep.getDestination();
      if (src == dst)
        return containingOp->emitError() << "dependence cycle detected";

      //     t_src + t.linkedOperatorType.latency <= t_dst
      // <=> 1 * t_src + -1 * t_dst <= -latency
      unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
      MPConstraint *constraint =
          solver.MakeRowConstraint(-infinity, -((double)latency));
      constraint->SetCoefficient(vars[src], 1);
      constraint->SetCoefficient(vars[dst], -1);
    }

  // Invoke solver. The LP is infeasible if the scheduling problem contained
  // dependence cycles. Otherwise, we expect the result to be optimal.
  MPSolver::ResultStatus result = solver.Solve();
  if (result == MPSolver::INFEASIBLE)
    return containingOp->emitError() << "dependence cycle detected";
  assert(result == MPSolver::OPTIMAL);

  // Retrieve start times.
  for (auto *op : prob.getOperations())
    prob.setStartTime(op, std::round(vars[op]->solution_value()));

  return success();
}
