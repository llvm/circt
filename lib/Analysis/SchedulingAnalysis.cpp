//===- SchedulingAnalysis.cpp - scheduling analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis involving scheduling.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"

using namespace mlir;

/// CyclicSchedulingAnalysis constructs a CyclicProblem for each AffineForOp by
/// performing a memory dependence analysis and inserting dependences into the
/// problem. The client should retrieve the partially complete problem to add
/// and associate operator types.
circt::analysis::CyclicSchedulingAnalysis::CyclicSchedulingAnalysis(
    Operation *op, AnalysisManager &am) {
  auto funcOp = cast<FuncOp>(op);

  MemoryDependenceAnalysis &memoryAnalysis =
      am.getAnalysis<MemoryDependenceAnalysis>();

  funcOp.walk([&](AffineForOp forOp) { analyzeForOp(forOp, memoryAnalysis); });
}

void circt::analysis::CyclicSchedulingAnalysis::analyzeForOp(
    AffineForOp forOp, MemoryDependenceAnalysis memoryAnalysis) {
  // Only consider innermost AffineForOps.
  if (isa<AffineForOp>(forOp.getBody()->front()))
    return;

  // Create a cyclic scheduling problem.
  CyclicProblem problem(forOp);

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<MemoryDependence> dependences = memoryAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (MemoryDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      assert(succeeded(problem.insertDependence(dep)));

      // Find the greatest distance lower bound from any loop and use that for
      // this dependence.
      unsigned distance = 0;
      for (DependenceComponent comp : memoryDep.dependenceComponents)
        if (comp.lb.getValue() > distance)
          distance = comp.lb.getValue();

      problem.setDistance(dep, distance);
    }
  });

  // Insert conditional dependences into the problem.
  forOp.getBody()->walk([&](AffineIfOp op) {
    // No special handling required for control-only `if`s.
    if (op.getNumResults() == 0)
      return WalkResult::skip();

    // Model the implicit value flow from the `yield` to the `if`'s result(s).
    Problem::Dependence depThen(op.getThenBlock()->getTerminator(), op);
    assert(succeeded(problem.insertDependence(depThen)));

    if (op.hasElse()) {
      Problem::Dependence depElse(op.getElseBlock()->getTerminator(), op);
      assert(succeeded(problem.insertDependence(depElse)));
    }

    return WalkResult::advance();
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](AffineWriteOpInterface op) {
    Problem::Dependence dep(op, anchor);
    assert(succeeded(problem.insertDependence(dep)));
  });

  // Handle explicitly computed loop-carried values, i.e. excluding the
  // induction variable. Insert inter-iteration dependences from the definers of
  // "iter_args" to their users.
  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
    auto iterArgs = forOp.getRegionIterArgs();
    for (unsigned i = 0; i < nIterArgs; ++i) {
      Operation *iterArgDefiner = anchor->getOperand(i).getDefiningOp();
      // If it's not an operation, we don't need to model the dependence.
      if (!iterArgDefiner)
        continue;

      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
        Problem::Dependence dep(iterArgDefiner, iterArgUser);
        assert(succeeded(problem.insertDependence(dep)));

        // Values always flow between subsequent iterations.
        problem.setDistance(dep, 1);
      }
    }
  }

  // Store the partially complete problem.
  problems.insert(std::pair<Operation *, CyclicProblem>(forOp, problem));
}

CyclicProblem
circt::analysis::CyclicSchedulingAnalysis::getProblem(AffineForOp forOp) {
  auto problem = problems.find(forOp);
  assert(problem != problems.end() && "expected problem to exist");
  return problem->second;
}
