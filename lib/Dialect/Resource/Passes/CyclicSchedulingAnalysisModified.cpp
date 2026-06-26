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

#include "circt/Dialect/Resource/Passes/CyclicSchedulingAnalysisModified.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/AnalysisManager.h"

using namespace mlir;
using namespace mlir::affine;
using namespace circt::scheduling;

/// CyclicSchedulingAnalysis constructs a CyclicProblem for each AffineForOp by
/// performing a memory dependence analysis and inserting dependences into the
/// problem. The client should retrieve the partially complete problem to add
/// and associate operator types.
circt::hls_analysis::CyclicSchedulingAnalysis::CyclicSchedulingAnalysis(
    Operation *op, AnalysisManager &am) {
  auto funcOp = cast<func::FuncOp>(op);

  analysis::MemoryDependenceAnalysis &memoryAnalysis =
      am.getAnalysis<analysis::MemoryDependenceAnalysis>();

  // Find the innermost loop of every perfectly nested band, regardless of
  // enclosing control flow (scf.if/scf.for, etc.). Walk pre-order and skip
  // loops already absorbed into a band we've processed, so each innermost
  // loop is analyzed exactly once.
  DenseSet<Operation *> visited;
  funcOp.walk<WalkOrder::PreOrder>([&](AffineForOp root) {
    if (visited.contains(root))
      return;
    SmallVector<AffineForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, root);
    for (AffineForOp l : nestedLoops)
      visited.insert(l);
    analyzeForOp(nestedLoops.back(), memoryAnalysis);
  });
}

void circt::hls_analysis::CyclicSchedulingAnalysis::analyzeForOp(
    AffineForOp forOp, analysis::MemoryDependenceAnalysis memoryAnalysis) {
  // Create a cyclic scheduling problem.
  CyclicProblem problem(forOp);

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<analysis::MemoryDependence> dependences = memoryAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (analysis::MemoryDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. This
      // assumes outer loops execute sequentially, i.e. one iteration of the
      // inner loop completes before the next iteration is initiated. With
      // proper analysis and lowerings, this can be relaxed.
      unsigned distance = *memoryDep.dependenceComponents.back().lb;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Insert conditional dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    Block *thenBlock = nullptr;
    Block *elseBlock = nullptr;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      thenBlock = ifOp.thenBlock();
      elseBlock = ifOp.elseBlock();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      thenBlock = ifOp.getThenBlock();
      if (ifOp.hasElse())
        elseBlock = ifOp.getElseBlock();
    } else {
      return WalkResult::advance();
    }

    // No special handling required for control-only `if`s.
    if (op->getNumResults() == 0)
      return WalkResult::skip();

    // Model the implicit value flow from the `yield` to the `if`'s result(s).
    Problem::Dependence depThen(thenBlock->getTerminator(), op);
    auto depInserted = problem.insertDependence(depThen);
    assert(succeeded(depInserted));
    (void)depInserted;

    if (elseBlock) {
      Problem::Dependence depElse(elseBlock->getTerminator(), op);
      depInserted = problem.insertDependence(depElse);
      assert(succeeded(depInserted));
      (void)depInserted;
    }

    return WalkResult::advance();
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](Operation *op) {
    if (!isa<AffineStoreOp, memref::StoreOp>(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
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
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;

        // Values always flow between subsequent iterations.
        problem.setDistance(dep, 1);
      }
    }
  }

  // Store the partially complete problem.
  problems.insert(std::pair<Operation *, CyclicProblem>(forOp, problem));
}

CyclicProblem &
circt::hls_analysis::CyclicSchedulingAnalysis::getProblem(AffineForOp forOp) {
  auto problem = problems.find(forOp);
  assert(problem != problems.end() && "expected problem to exist");
  return problem->second;
}
