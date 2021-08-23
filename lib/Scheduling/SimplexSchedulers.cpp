//===- SimplexSchedulers.cpp - Linear programming-based schedulers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of linear programming-based schedulers with a built-in simplex
// solver.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "simplex-schedulers"

using namespace circt;
using namespace circt::scheduling;

using llvm::dbgs;
using llvm::format;

namespace {

/// This class provides a framework to model certain scheduling problems as
/// lexico-parametric linear programs (LP), which are then solved with an
/// extended version of the dual simplex algorithm.
///
/// The approach is described in:
///   B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///   Instruction Scheduling", PRISM 1994.22, 1994.
///
/// Resource-free scheduling problems (called "central problems" in the paper)
/// have an *integer* linear programming formulation with a totally unimodular
/// constraint matrix. Such ILPs can however be solved optimally in polynomial
/// time with a (non-integer) LP solver (such as the simplex algorithm), as the
/// LP solution is guaranteed to be integer. Note that this is the same idea as
/// used by SDC-based schedulers.
class SimplexSchedulerBase {
protected:
  /// The objective is to minimize the start time of this operation.
  Operation *lastOp;

  /// S is part of a mechanism to assign fixed values to the LP variables.
  int parameterS;

  /// T represents the initiation interval (II). Its minimally-feasible value is
  /// computed by the algorithm.
  int parameterT;

  /// The simplex tableau is the algorithm's main data structure.
  /// The dashed parts always contain the zero respectively the identity matrix,
  /// and therefore are not stored explicitly.
  ///
  ///                        ◀─────nColumns──▶
  ///                       ┌─────┬───────────┬ ─ ─ ─ ─ ┐
  ///        objectiveRow > │. . .│. . ... . .│    0        ▲
  ///                       ├─────┼───────────┼ ─ ─ ─ ─ ┤   │
  ///  firstConstraintRow > │. . .│. . ... . .│1            │
  ///                       │. . .│. . ... . .│  1      │   │nRows
  ///                       │. . .│. . ... . .│    1        │
  ///                       │. . .│. . ... . .│      1  │   │
  ///                       │. . .│. . ... . .│        1    ▼
  ///                       └─────┴───────────┴ ─ ─ ─ ─ ┘
  ///       parameter1Column ^
  ///         parameterSColumn ^
  ///           parameterTColumn ^
  ///  firstNonBasicVariableColumn ^
  ///                              ─────────── ──────────
  ///                       nonBasicVariables   basicVariables
  SmallVector<SmallVector<int>> tableau;

  /// During the pivot operation, one column in the elided part of the tableau
  /// is modified; this vector temporarily catches the changes.
  SmallVector<int> implicitBasicVariableColumnVector;

  /// The linear program models the operations' start times as variables, which
  /// we identify here as 0, ..., |ops|-1.
  /// Additionally, for each dependence (precisely, the inequality modeling the
  /// precedence constraint), a slack variable is required; these are identified
  /// as |ops|, ..., |ops|+|deps|-1.
  ///
  /// This vector stores the numeric IDs of non-basic variables. A variable's
  /// index *i* in this vector corresponds to the tableau *column*
  /// `firstNonBasicVariableColumn`+*i*.
  SmallVector<unsigned> nonBasicVariables;

  /// This vector store the numeric IDs of basic variables. A variable's index
  /// *i* in this vector corresponds to the tableau *row*
  /// `firstConstraintRow`+*i*.
  SmallVector<unsigned> basicVariables;

  /// Used to conveniently retrieve an operation's start time variable. The
  /// alternative would be to find the op's index in the problem's list of
  /// operations.
  DenseMap<Operation *, unsigned> startTimeVariables;

  /// This vector keeps track of the current locations (i.e. row or column) of
  /// a start time variable in the tableau. We encode column numbers as positive
  /// integers, and row numbers as negative integers. We do not track the slack
  /// variables.
  SmallVector<int> startTimeLocations;

  /// Non-basic variables can be "frozen", which prevents them from being
  /// pivoted into basis again.
  DenseSet<unsigned> frozenVariables;

  /// Number of rows in the tableau = 1 + |deps|.
  unsigned nRows;
  /// Number of explicitly stored columns in the tableau = 3 + |ops|.
  unsigned nColumns;

  /// The first row encodes the LP's objective function.
  static constexpr unsigned objectiveRow = 0;
  /// All other rows encode linear constraints.
  static constexpr unsigned firstConstraintRow = 1;

  /// The first column corresponds to the always-one "parameter" in u = (1,S,T).
  static constexpr unsigned parameter1Column = 0;
  /// The second column corresponds to the variable-freezing parameter S.
  static constexpr unsigned parameterSColumn = 1;
  /// The third column corresponds to the parameter T, i.e. the current II.
  static constexpr unsigned parameterTColumn = 2;
  /// All other (explicitly stored) columns represent non-basic variables.
  static constexpr unsigned firstNonBasicVariableColumn = 3;

  virtual Problem &getProblem() = 0;
  virtual void fillConstraintRow(SmallVector<int> &row, detail::Dependence dep);
  void buildTableau();
  Optional<unsigned> findPivotRow();
  Optional<unsigned> findPivotColumn(unsigned pivotRow,
                                     bool allowPositive = false);
  void multiplyRow(unsigned row, int factor);
  void addMultipleOfRow(unsigned sourceRow, int factor, unsigned targetRow);
  void pivot(unsigned pivotRow, unsigned pivotColumn);
  LogicalResult solveTableau();
  bool isInBasis(unsigned startTimeVariable);
  unsigned freeze(unsigned startTimeVariable);
  void translate(unsigned column, int factor1, int factorS, int factorT);
  LogicalResult scheduleAt(unsigned startTimeVariable, unsigned timeStep);
  void storeStartTimes();

  void dumpTableau();

public:
  explicit SimplexSchedulerBase(Operation *lastOp) : lastOp(lastOp) {}
  virtual ~SimplexSchedulerBase() = default;
  virtual LogicalResult schedule() = 0;
};

/// This class solves the basic, acyclic `Problem`.
class SimplexScheduler : public SimplexSchedulerBase {
private:
  Problem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SimplexScheduler(Problem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}

  LogicalResult schedule() override;
};

/// This class solves the resource-free `CyclicProblem`.  The optimal initiation
/// interval (II) is determined as a side product of solving the parametric
/// problem, and corresponds to the "RecMII" (= recurrence-constrained minimum
/// II) usually considered as one component in the lower II bound used by modulo
/// schedulers.
class CyclicSimplexScheduler : public SimplexSchedulerBase {
private:
  CyclicProblem &prob;

protected:
  Problem &getProblem() override { return prob; }
  void fillConstraintRow(SmallVector<int> &row,
                         detail::Dependence dep) override;

public:
  CyclicSimplexScheduler(CyclicProblem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SimplexSchedulerBase
//===----------------------------------------------------------------------===//

void SimplexSchedulerBase::fillConstraintRow(SmallVector<int> &row,
                                             detail::Dependence dep) {
  auto &prob = getProblem();
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();
  unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
  row[parameter1Column] = -latency; // note the negation
  row[startTimeLocations[startTimeVariables[src]]] = 1;
  row[startTimeLocations[startTimeVariables[dst]]] = -1;
}

void SimplexSchedulerBase::buildTableau() {
  auto &prob = getProblem();

  // The initial tableau is constructed so that operations' start time variables
  // are out of basis, whereas all slack variables are in basis. We will number
  // them accordingly.
  unsigned var = 0;

  // Assign column and variable numbers to the operations' start times.
  for (auto *op : prob.getOperations()) {
    nonBasicVariables.push_back(var);
    startTimeVariables[op] = var;
    startTimeLocations.push_back(firstNonBasicVariableColumn + var);
    ++var;
  }

  // one column for each parameter (1,S,T), and for all operations
  nColumns = 3 + nonBasicVariables.size();

  // Helper to grow both the tableau and the implicit column vector.
  auto addRow = [&]() -> SmallVector<int> & {
    implicitBasicVariableColumnVector.push_back(0);
    return tableau.emplace_back(nColumns, 0);
  };

  // Set up the objective row.
  auto &objRowVec = addRow();
  objRowVec[startTimeLocations[startTimeVariables[lastOp]]] = 1;

  // Now set up rows/constraints for the dependences.
  for (auto *op : prob.getOperations()) {
    for (auto &dep : prob.getDependences(op)) {
      auto &consRowVec = addRow();
      fillConstraintRow(consRowVec, dep);
      basicVariables.push_back(var);
      ++var;
    }
  }

  // `objectiveRow` + one row per dependence
  nRows = tableau.size();
}

Optional<unsigned> SimplexSchedulerBase::findPivotRow() {
  // Find the first row for which the dot product ~B_p u is negative.
  for (unsigned row = firstConstraintRow; row < nRows; ++row) {
    auto &rowVec = tableau[row];
    int rowVal = rowVec[parameter1Column] +
                 rowVec[parameterSColumn] * parameterS +
                 rowVec[parameterTColumn] * parameterT;
    if (rowVal < 0)
      return row;
  }

  return None;
}

Optional<unsigned> SimplexSchedulerBase::findPivotColumn(unsigned pivotRow,
                                                         bool allowPositive) {
  Optional<int> maxQuot;
  Optional<unsigned> pivotCol;
  // Look for negative entries in the ~A part of the tableau. If multiple
  // candidates exist, take the one with maximum of the quotient:
  // tableau[objectiveRow][col] / pivotCand
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    if (frozenVariables.count(
            nonBasicVariables[col - firstNonBasicVariableColumn]))
      continue;
    int pivotCand = tableau[pivotRow][col];
    if (pivotCand < 0 || (allowPositive && pivotCand > 0)) {
      // The ~A part of the tableau has only {-1, 0, 1} entries by construction.
      assert(pivotCand * pivotCand == 1);
      int quot = tableau[objectiveRow][col] / pivotCand;
      if (!maxQuot || quot > *maxQuot) {
        maxQuot = quot;
        pivotCol = col;
      }
    }
  }

  return pivotCol;
}

void SimplexSchedulerBase::multiplyRow(unsigned row, int factor) {
  assert(factor != 0);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[row][col] *= factor;
  // Also multiply the corresponding entry in the temporary column vector.
  implicitBasicVariableColumnVector[row] *= factor;
}

void SimplexSchedulerBase::addMultipleOfRow(unsigned sourceRow, int factor,
                                            unsigned targetRow) {
  assert(factor != 0 && sourceRow != targetRow);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[targetRow][col] += tableau[sourceRow][col] * factor;
  // Again, perform row operation on the temporary column vector as well.
  implicitBasicVariableColumnVector[targetRow] +=
      implicitBasicVariableColumnVector[sourceRow] * factor;
}

/// The pivot operation applies elementary row operations to the tableau in
/// order to make the \p pivotColumn (corresponding to a non-basic variable) a
/// unit vector (only the \p pivotRow'th entry is 1). Then, a basis exchange is
/// performed: the non-basic variable is swapped with the basic variable
/// associated with the pivot row.
void SimplexSchedulerBase::pivot(unsigned pivotRow, unsigned pivotColumn) {
  // The implicit columns are part of an identity matrix.
  implicitBasicVariableColumnVector[pivotRow] = 1;

  int pivotElem = tableau[pivotRow][pivotColumn];
  // The ~A part of the tableau has only {-1, 0, 1} entries by construction.
  assert(pivotElem * pivotElem == 1);
  // Make `tableau[pivotRow][pivotColumn]` := 1
  multiplyRow(pivotRow, 1 / pivotElem);

  for (unsigned row = 0; row < nRows; ++row) {
    if (row == pivotRow)
      continue;

    int elem = tableau[row][pivotColumn];
    if (elem == 0)
      continue; // nothing to do

    // Make `tableau[row][pivotColumn]` := 0.
    addMultipleOfRow(pivotRow, -elem, row);
  }

  // Swap the pivot column with the implicitly constructed column vector.
  // We really only need to copy in one direction here, as the former pivot
  // column is a unit vector, which is not stored explicitly.
  for (unsigned row = 0; row < nRows; ++row) {
    tableau[row][pivotColumn] = implicitBasicVariableColumnVector[row];
    implicitBasicVariableColumnVector[row] = 0; // Reset for next pivot step.
  }

  // Look up numeric IDs of variables involved in this pivot operation.
  unsigned &nonBasicVar =
      nonBasicVariables[pivotColumn - firstNonBasicVariableColumn];
  unsigned &basicVar = basicVariables[pivotRow - firstConstraintRow];

  // Keep track of where start time variables are; ignore slack variables.
  if (nonBasicVar < startTimeLocations.size())
    startTimeLocations[nonBasicVar] = -pivotRow; // ...going into basis.
  if (basicVar < startTimeLocations.size())
    startTimeLocations[basicVar] = pivotColumn; // ...going out of basis.

  // Record the swap in the variable lists.
  std::swap(nonBasicVar, basicVar);
}

LogicalResult SimplexSchedulerBase::solveTableau() {
  // Iterate as long as we find rows to pivot on (~B_p u is negative), otherwise
  // an optimal solution has been found.
  while (auto pivotRow = findPivotRow()) {
    // Look for pivot elements.
    if (auto pivotCol = findPivotColumn(*pivotRow)) {
      pivot(*pivotRow, *pivotCol);
      continue;
    }

    // If we did not find a pivot column, then the entire row contained only
    // positive entries, and the problem is in principle infeasible. However, if
    // the entry in the `parameterTColumn` is positive, we can make the LP
    // feasible again by increasing the II.
    int entry1Col = tableau[*pivotRow][parameter1Column];
    int entryTCol = tableau[*pivotRow][parameterTColumn];
    if (entryTCol > 0) {
      // The negation of `entry1Col` is not in the paper. I think this is an
      // oversight, because `entry1Col` certainly is negative (otherwise the row
      // would not have been a valid pivot row), and without the negation, the
      // new II would be negative.
      assert(entry1Col < 0);
      parameterT = (-entry1Col - 1) / entryTCol + 1;

      LLVM_DEBUG(dbgs() << "Increased II to " << parameterT << '\n');

      continue;
    }

    // Otherwise, the linear program is infeasible.
    return failure();
  }

  // Optimal solution found!
  return success();
}

bool SimplexSchedulerBase::isInBasis(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());
  int loc = startTimeLocations[startTimeVariable];
  if (-loc >= (int)firstConstraintRow)
    return true;
  if (loc >= (int)firstNonBasicVariableColumn)
    return false;
  llvm_unreachable("Invalid variable location");
}

// The following `freeze`, `translate` and `scheduleAt` methods are implemented
// based on a follow-up publication to the paper mentioned above:
//
//   B. D. de Dinechin, "Fast Modulo Scheduling Under the Simplex Scheduling
//   Framework", PRISM 1995.01, 1995.

unsigned SimplexSchedulerBase::freeze(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  // Mark variable.
  frozenVariables.insert(startTimeVariable);

  if (!isInBasis(startTimeVariable))
    // That's all for non-basic variables.
    return startTimeLocations[startTimeVariable];

  // We need to pivot this variable one out of basis.
  unsigned pivotRow = -startTimeLocations[startTimeVariable];

  // Here, positive pivot elements can be considered as well, hence finding a
  // suitable column should not fail.
  auto pivotCol = findPivotColumn(pivotRow, /* allowPositive= */ true);
  assert(pivotCol);

  // Perform the exchange.
  pivot(pivotRow, *pivotCol);

  // `startTimeVariable` is now represented by `pivotCol`.
  return *pivotCol;
}

void SimplexSchedulerBase::translate(unsigned column, int factor1, int factorS,
                                     int factorT) {
  for (unsigned row = 0; row < nRows; ++row) {
    auto &rowVec = tableau[row];
    int elem = rowVec[column];
    if (elem == 0)
      continue;

    rowVec[parameter1Column] += -elem * factor1;
    rowVec[parameterSColumn] += -elem * factorS;
    rowVec[parameterTColumn] += -elem * factorT;
  }
}

LogicalResult SimplexSchedulerBase::scheduleAt(unsigned startTimeVariable,
                                               unsigned timeStep) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  // Freeze variable and translate its column by parameter S.
  unsigned frozenCol = freeze(startTimeVariable);
  translate(frozenCol, /* factor1= */ 0, /* factorS= */ 1, /* factorT= */ 0);

  // Temporarily set S to the desired value, and attempt to solve.
  parameterS = timeStep;
  auto res = solveTableau();
  parameterS = 0;

  if (failed(res)) {
    // The LP is infeasible with the new constraint. We could try other values
    // for S, but for now, we just roll back and signal failure to the driver.
    translate(frozenCol, /* factor1= */ 0, /* factorS= */ -1, /* factorT= */ 0);
    frozenVariables.erase(startTimeVariable);
    res = solveTableau();
    assert(succeeded(res));
    return failure();
  }

  // Translate S by the other parameter(s). Currently, this means setting
  // `factor1` to `timeStep`. For cyclic problems, one would perform a modulo
  // decomposition: S = `factor1` + `factorT` * T, with `factor1` < T.
  //
  // This translation does not change the value of ~B_p u (the dot product of
  // the the first three columns with parameters), hence we do not need to solve
  // the tableau again.
  //
  // Note: I added a negation of the factors here, which is not mentioned in the
  // paper's text, but apparently used in the example. Without it, the intended
  // effect, i.e. making the S-column all-zero again, is not achieved.
  assert(parameterT == 0);
  translate(parameterSColumn, -timeStep, /* factorS= */ 1, /* factorT= */ 0);

  // Record the start time.
  Problem &prob = getProblem();
  prob.setStartTime(prob.getOperations()[startTimeVariable], timeStep);

  return success();
}

void SimplexSchedulerBase::storeStartTimes() {
  auto &prob = getProblem();

  for (auto *op : prob.getOperations()) {
    unsigned startTimeVar = startTimeVariables[op];
    if (!isInBasis(startTimeVar)) {
      // Non-basic variables that are not already fixed to a specific time step
      // are 0 at the end of the simplex algorithm.
      if (!frozenVariables.contains(startTimeVar))
        prob.setStartTime(op, 0);
      continue;
    }
    // For the variables currently in basis, we look up the solution in the ~B
    // part of the tableau.
    auto &rowVec = tableau[-startTimeLocations[startTimeVar]];
    unsigned startTime = rowVec[parameter1Column] +
                         rowVec[parameterSColumn] * parameterS +
                         rowVec[parameterTColumn] * parameterT;
    prob.setStartTime(op, startTime);
  }
}

void SimplexSchedulerBase::dumpTableau() {
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  for (unsigned i = 0; i < nRows; ++i) {
    if (i == firstConstraintRow) {
      for (unsigned j = 0; j < nColumns; ++j) {
        if (j == firstNonBasicVariableColumn)
          dbgs() << "-+";
        dbgs() << "----";
      }
      dbgs() << '\n';
    }
    for (unsigned j = 0; j < nColumns; ++j) {
      if (j == firstNonBasicVariableColumn)
        dbgs() << " |";
      dbgs() << format(" %3d", tableau[i][j]);
    }
    if (i >= firstConstraintRow)
      dbgs() << format(" |< %2d", basicVariables[i - firstConstraintRow]);
    dbgs() << '\n';
  }
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  dbgs() << format(" %3d %3d %3d | ", 1, parameterS, parameterT);
  for (unsigned j = firstNonBasicVariableColumn; j < nColumns; ++j)
    dbgs() << format(" %2d^",
                     nonBasicVariables[j - firstNonBasicVariableColumn]);
  dbgs() << '\n';
}

//===----------------------------------------------------------------------===//
// SimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult SimplexScheduler::schedule() {
  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Optimal solution found with start time of last operation = "
             << -tableau[objectiveRow][parameter1Column] << '\n');

  storeStartTimes();
  return success();
}

//===----------------------------------------------------------------------===//
// CyclicSimplexScheduler
//===----------------------------------------------------------------------===//

void CyclicSimplexScheduler::fillConstraintRow(SmallVector<int> &row,
                                               detail::Dependence dep) {
  SimplexSchedulerBase::fillConstraintRow(row, dep);
  if (auto dist = prob.getDistance(dep))
    row[parameterTColumn] = *dist;
}

LogicalResult CyclicSimplexScheduler::schedule() {
  parameterS = 0;
  parameterT = 1;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Optimal solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -tableau[objectiveRow][parameter1Column] << '\n');

  prob.setInitiationInterval(parameterT);
  storeStartTimes();
  return success();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult scheduling::scheduleSimplex(Problem &prob, Operation *lastOp) {
  SimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(CyclicProblem &prob,
                                          Operation *lastOp) {
  CyclicSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(SharedPipelinedOperatorsProblem &prob,
                                          Operation *lastOp) {
  (void)prob;
  (void)lastOp;
  return failure();
}
