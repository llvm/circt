//===- SimplexScheduler.cpp - Linear programming-based scheduler ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of a scheduler that applies the simplex algorithm to a linear
// program formulation of the resource-free cyclic scheduling problem.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "simplex-scheduler"

using namespace circt;
using namespace circt::scheduling;

using llvm::dbgs;
using llvm::format;

namespace {

/// This class models the cyclic scheduling problem as a lexico-parametric
/// linear program (LP), and solves it with an extended version of the dual
/// simplex algorithm.
///
/// The approach is described in:
///   B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///   Instruction Scheduling", PRISM 1994.22, 1994.
///
/// Our `CyclicProblem` corresponds to de Dinechin's initial, resource-free
/// "central problem", and results in an *integer* linear programming
/// formulation with a totally unimodular constraint matrix. Such ILPs can
/// however be solved optimally in polynomial time with a (non-integer) LP
/// solver (such as the simplex algorithm), as the LP solution is guaranteed to
/// be integer.
/// (Note that this is the same idea as used by SDC-based schedulers.)
///
/// The LP's objective is to minimize the start time of the client-provided
/// "last" operation. The optimal initiation interval (II) is determined as a
/// side product of solving the parametric problem, and corresponds to the
/// "RecMII" (= recurrence-constrained minimum II) usually considered as one
/// component in the lower II bound used by modulo schedulers.
class SimplexScheduler {
private:
  CyclicProblem &prob;
  Operation *lastOp;

  /// The minimally-feasible initiation interval is computed by the algorithm.
  int parameterII;

  /// The simplex tableau is the algorithm's main data structure.
  /// The dashed parts always contain the zero respectively the identity matrix,
  /// and therefore are not stored explicitly.
  ///
  ///                          ◀───nColumns──▶
  ///                         ┌───┬───────────┬ ─ ─ ─ ─ ┐
  ///          objectiveRow > │~Z │. . ~C^T. .│    0        ▲
  ///                         ├───┼───────────┼ ─ ─ ─ ─ ┤   │
  ///    firstConstraintRow > │. .│. . . . . .│1            │
  ///                         │. .│. . . . . .│  1      │   │nRows
  ///                         │~B |. . ~A  . .│    1        │
  ///                         │. .│. . . . . .│      1  │   │
  ///                         │. .│. . . . . .│        1    ▼
  ///                         └───┴───────────┴ ─ ─ ─ ─ ┘
  ///       parameterOneColumn ^
  ///          parameterIIColumn ^
  ///  firstNonBasicVariableColumn ^
  ///                              ─────────── ──────────
  ///                       nonBasicVariables   basicVariables
  SmallVector<SmallVector<int>> tableau;

  /// During the pivot operation, one column in the elided part of the tableau
  /// is modified; this vector temporarily catches the changes.
  SmallVector<int> implicitBasicVariableColumnVector;

  /// The linear program models the operations' start times as variables, which
  /// we identify here as 0, ..., |ops|-1.
  /// Additionally, for each depedence (precisely, the inequality modeling the
  /// precedence constraint), a slack variable is required; these are identified
  /// as |ops|, ..., |ops|+|deps|-1.
  ///
  /// This vector stores the numeric IDs of non-basic variables. A variable's
  /// index *i* in this vector corresponds to the tableau *column*
  /// `firstNonBasicVariableColum`+*i*.
  SmallVector<unsigned> nonBasicVariables;

  /// This vector store the numeric IDs of basic variables. A variable's index
  /// *i* in this vector corresponds to the tableau *row*
  /// `firstConstraintRow`+*i*.
  SmallVector<unsigned> basicVariables;

  /// Number of rows in the tableau = 1 + |deps|.
  unsigned nRows;
  /// Number of explicitly stored columns in the tableau = 2 + |ops|.
  unsigned nColumns;

  /// The first row encodes the LP's objective function.
  static constexpr unsigned objectiveRow = 0;
  /// All other rows encode linear constraints.
  static constexpr unsigned firstConstraintRow = 1;

  /// The first column corresponds to the always-one "parameter" in u = (1,S,T).
  static constexpr unsigned parameterOneColumn = 0;
  /// The second column corresponds to the parameter T, i.e. the initiation
  /// interval. Note that we do not model the parameter S yet.
  static constexpr unsigned parameterIIColumn = 1;
  /// All other (explicitly stored) columns represent non-basic variables.
  static constexpr unsigned firstNonBasicVariableColumn = 2;

  void buildTableau();
  Optional<unsigned> findPivotRow();
  Optional<unsigned> findPivotColumn(unsigned pivotRow);
  void multiplyRow(unsigned row, int factor);
  void addMultipleOfRow(unsigned sourceRow, int factor, unsigned targetRow);
  void pivot(unsigned pivotRow, unsigned pivotColumn);

  void dumpTableau();

public:
  SimplexScheduler(CyclicProblem &prob, Operation *lastOp)
      : prob(prob), lastOp(lastOp) {}
  LogicalResult schedule();
};

} // anonymous namespace

void SimplexScheduler::buildTableau() {
  // Helper map to lookup an operation's column number in the tableau.
  SmallDenseMap<Operation *, unsigned> opCols;

  // The initial tableau is constructed so that operations' start time variables
  // are out of basis, whereas all slack variables are in basis. We will number
  // them accordingly.
  unsigned varNum = 0;

  // Assign column and variable numbers to the operations' start times.
  for (auto *op : prob.getOperations()) {
    opCols[op] = firstNonBasicVariableColumn + varNum;
    nonBasicVariables.push_back(varNum++);
  }

  // `parameterOneColumn` + `parameterIIColumn` + one column per operation
  nColumns = 2 + nonBasicVariables.size();

  // Helper to grow both the tableau and the implicit column vector.
  auto addRow = [&]() -> SmallVector<int> & {
    implicitBasicVariableColumnVector.push_back(0);
    return tableau.emplace_back(nColumns, 0);
  };

  // Set up the objective row.
  auto &objRowVec = addRow();
  objRowVec[opCols[lastOp]] = 1;

  // Now set up rows/constraints for the dependences.
  for (auto *op : prob.getOperations()) {
    for (auto &dep : prob.getDependences(op)) {
      auto &consRowVec = addRow();
      basicVariables.push_back(varNum++);

      Operation *src = dep.getSource();
      Operation *dst = dep.getDestination();
      unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
      unsigned distance = prob.getDistance(dep).getValueOr(0);
      consRowVec[parameterOneColumn] = -latency; // note the negation
      consRowVec[parameterIIColumn] = distance;
      consRowVec[opCols[src]] = 1;
      consRowVec[opCols[dst]] = -1;
    }
  }

  // `objectiveRow` + one row per dependence
  nRows = tableau.size();
}

Optional<unsigned> SimplexScheduler::findPivotRow() {
  // Find the first row for which the dot product "~B_p u" is negative.
  for (unsigned row = firstConstraintRow; row < nRows; ++row) {
    int rowVal = tableau[row][parameterOneColumn] +
                 tableau[row][parameterIIColumn] * parameterII;
    if (rowVal < 0)
      return row;
  }

  return None;
}

Optional<unsigned> SimplexScheduler::findPivotColumn(unsigned pivotRow) {
  Optional<int> maxQuot;
  Optional<unsigned> pivotCol;
  // Look for negative entries in the ~A part of the tableau. If multiple
  // candidates exist, take the one with maximum of the quotient:
  // tableau[objectiveRow][col] / pivotCand
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    int pivotCand = tableau[pivotRow][col];
    if (pivotCand < 0) {
      // The ~A part of the tableau has only {-1, 0, 1} entries by construction.
      assert(pivotCand == -1);
      int quot = -tableau[objectiveRow][col];
      // Quotient in general: tableau[objectiveRow][col] / pivotCand
      if (!maxQuot || quot > *maxQuot) {
        maxQuot = quot;
        pivotCol = col;
      }
    }
  }

  return pivotCol;
}

void SimplexScheduler::multiplyRow(unsigned row, int factor) {
  assert(factor != 0);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[row][col] *= factor;
  // Also multiply the corresponding entry in the temporary column vector.
  implicitBasicVariableColumnVector[row] *= factor;
}

void SimplexScheduler::addMultipleOfRow(unsigned sourceRow, int factor,
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
void SimplexScheduler::pivot(unsigned pivotRow, unsigned pivotColumn) {
  // The implicit columns are part of an identity matrix.
  implicitBasicVariableColumnVector[pivotRow] = 1;

  int pivotElem = tableau[pivotRow][pivotColumn];
  // The ~A part of the tableau has only {-1, 0, 1} entries by construction.
  // The pivot element must be negative, so it can only be -1.
  assert(pivotElem == -1);
  // Make `tableau[pivotRow][pivotColumn]` := 1
  multiplyRow(pivotRow, -1); // Factor in general: 1 / pivotElement

  for (unsigned row = 0; row < nRows; ++row) {
    if (row == pivotRow)
      continue;

    int elem = tableau[row][pivotColumn];
    if (elem == 0)
      continue; // nothing to do

    // Make `tableau[row][pivotColumn]` := 0.
    // Factor in general: -elem / pivotElem
    addMultipleOfRow(pivotRow, -elem, row);
  }

  // Swap the pivot column with the implictly constructed column vector.
  // We really only need to copy in one direction here, as the former pivot
  // column is a unit vector, which is not stored explictly.
  for (unsigned row = 0; row < nRows; ++row) {
    tableau[row][pivotColumn] = implicitBasicVariableColumnVector[row];
    implicitBasicVariableColumnVector[row] = 0; // Reset for next pivot step.
  }

  // Record the swap in the variable lists.
  std::swap(nonBasicVariables[pivotColumn - firstNonBasicVariableColumn],
            basicVariables[pivotRow - firstConstraintRow]);
}

void SimplexScheduler::dumpTableau() {
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
  dbgs() << "          ";
  for (unsigned j = firstNonBasicVariableColumn; j < nColumns; ++j)
    dbgs() << format(" %2d^",
                     nonBasicVariables[j - firstNonBasicVariableColumn]);
  dbgs() << '\n';
}

LogicalResult SimplexScheduler::schedule() {
  // Initialize data structures.
  parameterII = 1;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n");
  LLVM_DEBUG(dumpTableau());

  // Iterate as long as we find rows to pivot on (~B_p u is negative), otherwise
  // an optimal solution has been found.
  while (auto pivotRow = findPivotRow()) {
    // Look for pivot elements.
    if (auto pivotCol = findPivotColumn(*pivotRow)) {
      pivot(*pivotRow, *pivotCol);

      LLVM_DEBUG(dbgs() << "Pivoted with " << *pivotRow << ',' << *pivotCol
                        << ":\n");
      LLVM_DEBUG(dumpTableau());

      continue;
    }

    // If we did not find a pivot column, then the entire row contained only
    // positive entries, and the problem is in principle infeasible. However, if
    // the entry in the `parameterIIColumn` is positive, we can make the LP
    // feasible again by increasing the II.
    int entryOneCol = tableau[*pivotRow][parameterOneColumn];
    int entryIICol = tableau[*pivotRow][parameterIIColumn];
    if (entryIICol > 0) {
      // The negation of `entryOneCol` is not in the paper. I think this is an
      // oversight, because `entryOneCol` certainly is negative (otherwise the
      // row would not have been a valid pivot row), and without the negation,
      // the new II would be negative.
      assert(entryOneCol < 0);
      parameterII = (-entryOneCol - 1) / entryIICol + 1;

      LLVM_DEBUG(dbgs() << "Increased II to " << parameterII << '\n');

      continue;
    }

    // Otherwise, there is nothing we can do.
    return prob.getContainingOp()->emitError("problem is infeasible");
  }

  LLVM_DEBUG(dbgs() << "Optimal solution found with II = " << parameterII
                    << " and start time of last operation = "
                    << -tableau[objectiveRow][parameterOneColumn] << '\n');

  // Store solution in the problem object.
  prob.setInitiationInterval(parameterII);

  auto &ops = prob.getOperations();
  unsigned nOps = ops.size();
  // For the start time variables currently in basis, we look up the solution
  // in the ~B part of the tableau. The slack variables (IDs >= |ops|) are
  // ignored.
  for (unsigned i = 0; i < basicVariables.size(); ++i) {
    unsigned varNum = basicVariables[i];
    if (varNum < nOps) {
      unsigned startTime =
          tableau[firstConstraintRow + i][parameterOneColumn] +
          tableau[firstConstraintRow + i][parameterIIColumn] * parameterII;
      prob.setStartTime(ops[varNum], startTime);
    }
  }

  // Non-basic variables are 0 at the end of the simplex algorithm.
  for (unsigned i = 0; i < nonBasicVariables.size(); ++i) {
    unsigned varNum = nonBasicVariables[i];
    if (varNum < nOps)
      prob.setStartTime(ops[varNum], 0);
  }

  return success();
}

LogicalResult scheduling::scheduleSimplex(CyclicProblem &prob,
                                          Operation *lastOp) {
  SimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}
