//===- PresburgerSchedulers.cpp -  Presburger lib based schedulers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of linear programming-based schedulers with the Presburger
// library Simplex.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::scheduling;

using namespace mlir::presburger;

namespace {

/// The Solver finds the smallest II that satisfies the constraints and
/// minimizes the objective function. The solver also finds when a particular
/// operation should be scheduled.
class Solver : private LexSimplex {
public:
  Solver(Problem &prob, unsigned numObj)
      : LexSimplex(1 + numObj + prob.getOperations().size()), prob(prob) {
    // Offsets for variable types.
    unsigned problemVarOffset = numObj + 1;

    // Map each operation to a variable representing its start time and make
    // their start time positive.
    unsigned var = problemVarOffset;
    for (Operation *op : prob.getOperations()) {
      opToVar[op] = var;
      addLowerBound(var, MPInt(0));
      ++var;
    }
  }

  /// Get the number of columns in the constraint system.
  unsigned getNumCols() const { return LexSimplex::getNumVariables() + 1; }

  using LexSimplex::addEquality;
  using LexSimplex::addInequality;

  /// Get the index of the operation in the solver.
  unsigned getOpIndex(Operation *op) const { return opToVar.lookup(op); }

  /// Create a latency constraint representing the given dependence.
  /// The constraint is represented as:
  /// dstOpStartTime >= srcOpStartTime + latency.
  void createLatencyConstraint(MutableArrayRef<MPInt> row,
                               Problem::Dependence dep) {
    // Constraint: dst >= src + latency.
    Operation *src = dep.getSource();
    Operation *dst = dep.getDestination();
    unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
    row.back() = -latency; // note the negation
    if (src !=
        dst) { // note that these coefficients just zero out in self-arcs.
      row[opToVar[src]] = -1;
      row[opToVar[dst]] = 1;
    }
  }

  /// Create a cyclic latency constraint representing the given dependence and
  /// the given dependence distance.
  /// The constraint is represented as:
  /// dstOpStartTime >= srcOpStartTime + latency + II * distance.
  void createCyclicLatencyConstraint(MutableArrayRef<MPInt> row,
                                     Problem::Dependence dep,
                                     const MPInt &distance) {
    createLatencyConstraint(row, dep);
    row[0] = distance;
  }

  /// Add a constant lower bound on the variable at position `var`, representing
  /// the constraint: `var >= bound`.
  void addLowerBound(unsigned var, const MPInt &bound) {
    SmallVector<MPInt, 8> row(getNumCols());
    row[var] = 1;
    row.back() = -bound;
    addInequality(row);
  }

  /// Fix the variable at position `var` to a constant, representing the
  /// constraint: `var == bound`.
  void addEqBound(unsigned var, const MPInt &bound) {
    SmallVector<MPInt, 8> row(getNumCols());
    row[var] = 1;
    row.back() = -bound;
    addEquality(row);
  }

  // Solve the problem, keeping II integer, but allowing the solutions can be
  // rational. We use a rational lexicographic simplex solver to do this.
  // To keep II integer, if we obtain a rational II, we fix the II to the
  // ceiling of the rational II. Since any II greater than the minimum II
  // is valid, this is a valid solution.
  MaybeOptimum<SmallVector<Fraction, 8>> solveRationally() {
    MaybeOptimum<SmallVector<Fraction, 8>> sample =
        LexSimplex::findRationalLexMin();

    if (!sample.isBounded())
      return sample;

    ArrayRef<Fraction> res = *sample;

    // If we have an integer II, we can just return the solution.
    Fraction ii = res[0];
    if (ii.num % ii.den == 0) {
      return sample;
    }

    // We have a rational solution for II. We fix II to the ceiling of the
    // given solution.
    addEqBound(0, ceil(ii));

    sample = LexSimplex::findRationalLexMin();
    assert(sample.isBounded() && "Rounded up II should be feasible");

    return sample;
  }

private:
  // Reference to the problem.
  Problem &prob;

  /// A mapping from operation to their index in the simplex.
  DenseMap<Operation *, unsigned> opToVar;
};

}; // namespace

LogicalResult scheduling::schedulePresburger(Problem &prob, Operation *lastOp) {
  Solver solver(prob, 1);

  // II = 0 for acyclic problems.
  solver.addEqBound(0, MPInt(0));

  // There is a single objective, minimize the last operation.
  {
    SmallVector<MPInt, 8> row(solver.getNumCols());
    row[1] = 1;
    row[solver.getOpIndex(lastOp)] = -1;
    solver.addEquality(row);
  }

  // Setup constraints for dependencies.
  {
    SmallVector<MPInt, 8> row(solver.getNumCols());
    for (auto *op : prob.getOperations()) {
      for (auto &dep : prob.getDependences(op)) {
        solver.createLatencyConstraint(row, dep);
        solver.addInequality(row);
        std::fill(row.begin(), row.end(), MPInt(0));
      }
    }
  }

  // The constraints from dependence are built in a way that the solution always
  // has integer rational start times. So, we can solve rationally keeping II
  // integer.
  auto res = solver.solveRationally();
  if (!res.isBounded())
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  auto sample = *res;

  for (auto *op : prob.getOperations())
    prob.setStartTime(op,
                      int64_t(sample[solver.getOpIndex(op)].getAsInteger()));

  return success();
}

LogicalResult scheduling::schedulePresburger(CyclicProblem &prob,
                                             Operation *lastOp) {
  Solver solver(prob, 1);

  // II >= 1 for cyclic problems.
  solver.addLowerBound(0, MPInt(1));

  // There is a single objective, minimize the last operation.
  {
    SmallVector<MPInt, 8> row(solver.getNumCols());
    row[1] = 1;
    row[solver.getOpIndex(lastOp)] = -1;
    solver.addEquality(row);
  }

  // Setup constraints for dependencies.
  {
    SmallVector<MPInt, 8> row(solver.getNumCols());
    for (auto *op : prob.getOperations()) {
      for (auto &dep : prob.getDependences(op)) {
        if (auto dist = prob.getDistance(dep))
          solver.createCyclicLatencyConstraint(row, dep, MPInt(*dist));
        else
          solver.createLatencyConstraint(row, dep);
        solver.addInequality(row);
        std::fill(row.begin(), row.end(), MPInt(0));
      }
    }
  }

  // The constraints from dependence are built in a way that the solution always
  // has integer rational start times. So, we can solve rationally keeping II
  // integer.
  auto res = solver.solveRationally();
  if (!res.isBounded())
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  auto sample = *res;

  prob.setInitiationInterval(int64_t(sample[0].getAsInteger()));
  for (auto *op : prob.getOperations())
    prob.setStartTime(op,
                      int64_t(sample[solver.getOpIndex(op)].getAsInteger()));

  return success();
}
