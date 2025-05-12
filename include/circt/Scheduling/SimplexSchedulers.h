//===- SimplexSchedulers.h - Linear programming-based schedulers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SIMPLEX_SCHEDULERS_H
#define CIRCT_SIMPLEX_SCHEDULERS_H

#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Schedulers.h"
#include "circt/Scheduling/Utilities.h"
#include "mlir/IR/Operation.h"

namespace circt::scheduling {

/// This class provides a framework to model certain scheduling problems as
/// lexico-parametric linear programs (LP), which are then solved with an
/// extended version of the dual simplex algorithm.
///
/// The approach is described in:
///  [1] B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///      Instruction Scheduling", PRISM 1994.22, 1994.
///  [2] B. D. de Dinechin, "Fast Modulo Scheduling Under the Simplex Scheduling
///      Framework", PRISM 1995.01, 1995.
///
/// Resource-free scheduling problems (called "central problems" in the papers)
/// have an *integer* linear programming formulation with a totally unimodular
/// constraint matrix. Such ILPs can however be solved optimally in polynomial
/// time with a (non-integer) LP solver (such as the simplex algorithm), as the
/// LP solution is guaranteed to be integer. Note that this is the same idea as
/// used by SDC-based schedulers.
template <class Derived, typename P>
class SimplexSchedulerBase {
protected:
  /// S is part of a mechanism to assign fixed values to the LP variables.
  int parameterS;

  /// T represents the initiation interval (II). Its minimally-feasible value is
  /// computed by the algorithm.
  int parameterT;

  /// The simplex tableau is the algorithm's main data structure.
  /// The dashed parts always contain the zero respectively the identity matrix,
  /// and therefore are not stored explicitly.
  ///
  ///                        ◀───nColumns────▶
  ///           nParameters────┐
  ///                        ◀─┴─▶
  ///                       ┌─────┬───────────┬ ─ ─ ─ ─ ┐
  ///                      ▲│. . .│. . ... . .│    0        ▲
  ///           nObjectives││. . .│. . ... . .│         │   │
  ///                      ▼│. . .│. . ... . .│             │
  ///                       ├─────┼───────────┼ ─ ─ ─ ─ ┤   │
  ///  firstConstraintRow > │. . .│. . ... . .│1            │nRows
  ///                       │. . .│. . ... . .│  1      │   │
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

  /// Non-basic variables can be "frozen" to a specific value, which prevents
  /// them from being pivoted into basis again.
  DenseMap<unsigned, unsigned> frozenVariables;

  /// Number of rows in the tableau = |obj| + |deps|.
  unsigned nRows;
  /// Number of explicitly stored columns in the tableau = |params| + |ops|.
  unsigned nColumns;

  // Number of objective rows.
  unsigned nObjectives;
  /// All other rows encode linear constraints.
  unsigned &firstConstraintRow = nObjectives;

  // Number of parameters (fixed for now).
  static constexpr unsigned nParameters = 3;
  /// The first column corresponds to the always-one "parameter" in u = (1,S,T).
  static constexpr unsigned parameter1Column = 0;
  /// The second column corresponds to the variable-freezing parameter S.
  static constexpr unsigned parameterSColumn = 1;
  /// The third column corresponds to the parameter T, i.e. the current II.
  static constexpr unsigned parameterTColumn = 2;
  /// All other (explicitly stored) columns represent non-basic variables.
  static constexpr unsigned firstNonBasicVariableColumn = nParameters;

  /// Allow subclasses to collect additional constraints that are not part of
  /// the input problem, but should be modeled in the linear problem.
  SmallVector<Problem::Dependence> additionalConstraints;

  virtual LogicalResult checkLastOp(Problem &problem, Operation *lastOp);
  virtual bool fillObjectiveRow(Problem &problem, Operation *lastOp,
                                SmallVector<int> &row, unsigned obj);
  virtual void fillConstraintRow(Problem &problem, SmallVector<int> &row,
                                 Problem::Dependence dep);
  virtual void fillAdditionalConstraintRow(Problem &problem,
                                           SmallVector<int> &row,
                                           Problem::Dependence dep);

  virtual void buildTableau(P &problem, Operation *lastOp);

  int getParametricConstant(unsigned row);
  SmallVector<int> getObjectiveVector(unsigned column);
  std::optional<unsigned> findDualPivotRow();
  std::optional<unsigned> findDualPivotColumn(unsigned pivotRow,
                                              bool allowPositive = false);
  std::optional<unsigned> findPrimalPivotColumn();
  std::optional<unsigned> findPrimalPivotRow(unsigned pivotColumn);
  void multiplyRow(unsigned row, int factor);
  void addMultipleOfRow(unsigned sourceRow, int factor, unsigned targetRow);
  void pivot(unsigned pivotRow, unsigned pivotColumn);
  LogicalResult solveTableau();
  LogicalResult restoreDualFeasibility();
  bool isInBasis(unsigned startTimeVariable);
  unsigned freeze(unsigned startTimeVariable, unsigned timeStep);
  void translate(unsigned column, int factor1, int factorS, int factorT);
  LogicalResult scheduleAt(unsigned startTimeVariable, unsigned timeStep);
  void moveBy(unsigned startTimeVariable, unsigned amount);
  unsigned getStartTime(unsigned startTimeVariable);

  void dumpTableau();

public:
  explicit SimplexSchedulerBase() = default;
  virtual ~SimplexSchedulerBase() = default;
};

/// This class solves the basic, acyclic `Problem`.
class SimplexScheduler : public SimplexSchedulerBase<SimplexScheduler, Problem>,
                         public Scheduler<Problem> {
  friend class SimplexSchedulerBase<SimplexScheduler, Problem>;

protected:
  using SimplexSchedulerBase<SimplexScheduler, Problem>::fillObjectiveRow;
  using SimplexSchedulerBase<SimplexScheduler, Problem>::fillConstraintRow;
  using SimplexSchedulerBase<SimplexScheduler,
                             Problem>::fillAdditionalConstraintRow;

public:
  SimplexScheduler() : SimplexSchedulerBase() {}

  LogicalResult schedule(Problem &problem, Operation *lastOp) override;
};

/// This class solves the resource-free `CyclicProblem`.  The optimal initiation
/// interval (II) is determined as a side product of solving the parametric
/// problem, and corresponds to the "RecMII" (= recurrence-constrained minimum
/// II) usually considered as one component in the lower II bound used by modulo
/// schedulers.
class CyclicSimplexScheduler
    : public SimplexSchedulerBase<CyclicSimplexScheduler, CyclicProblem>,
      public Scheduler<CyclicProblem> {
  friend class SimplexSchedulerBase<CyclicSimplexScheduler, CyclicProblem>;

protected:
  using SimplexSchedulerBase<CyclicSimplexScheduler,
                             CyclicProblem>::fillObjectiveRow;
  using SimplexSchedulerBase<CyclicSimplexScheduler,
                             CyclicProblem>::fillConstraintRow;
  using SimplexSchedulerBase<CyclicSimplexScheduler,
                             CyclicProblem>::fillAdditionalConstraintRow;

  void fillConstraintRow(CyclicProblem &problem, SmallVector<int> &row,
                         Problem::Dependence dep);

public:
  CyclicSimplexScheduler() : SimplexSchedulerBase() {}
  LogicalResult schedule(CyclicProblem &problem, Operation *lastOp) override;
};

// This class solves acyclic, resource-constrained `SharedOperatorsProblem` with
// a simplified version of the iterative heuristic presented in [2].
class SharedOperatorsSimplexScheduler
    : public SimplexSchedulerBase<SharedOperatorsSimplexScheduler,
                                  SharedOperatorsProblem>,
      public Scheduler<SharedOperatorsProblem> {
  friend class SimplexSchedulerBase<SharedOperatorsSimplexScheduler,
                                    SharedOperatorsProblem>;

protected:
  using SimplexSchedulerBase<SharedOperatorsSimplexScheduler,
                             SharedOperatorsProblem>::fillObjectiveRow;
  using SimplexSchedulerBase<SharedOperatorsSimplexScheduler,
                             SharedOperatorsProblem>::fillConstraintRow;
  using SimplexSchedulerBase<
      SharedOperatorsSimplexScheduler,
      SharedOperatorsProblem>::fillAdditionalConstraintRow;

public:
  SharedOperatorsSimplexScheduler() = default;
  LogicalResult schedule(SharedOperatorsProblem &problem,
                         Operation *lastOp) override;
};

// This class solves the `ModuloProblem` using the iterative heuristic presented
// in [2].
class ModuloSimplexScheduler
    : public CyclicSimplexScheduler,
      public SimplexSchedulerBase<ModuloSimplexScheduler, ModuloProblem>,
      public Scheduler<ModuloProblem> {
  friend class SimplexSchedulerBase<ModuloSimplexScheduler, ModuloProblem>;

private:
  struct MRT {
    ModuloSimplexScheduler &sched;

    using TableType = SmallDenseMap<unsigned, DenseSet<Operation *>>;
    using ReverseTableType = SmallDenseMap<Operation *, unsigned>;
    SmallDenseMap<Problem::OperatorType, TableType> tables;
    SmallDenseMap<Problem::OperatorType, ReverseTableType> reverseTables;

    explicit MRT(ModuloSimplexScheduler &sched) : sched(sched) {}
    LogicalResult enter(ModuloProblem &problem, Operation *op,
                        unsigned timeStep);
    void release(ModuloProblem &problem, Operation *op);
  };

  SmallVector<unsigned> asapTimes, alapTimes;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;

protected:
  using SimplexSchedulerBase<ModuloSimplexScheduler,
                             ModuloProblem>::checkLastOp;

  using CyclicSimplexScheduler::buildTableau;
  using CyclicSimplexScheduler::checkLastOp;
  using CyclicSimplexScheduler::dumpTableau;
  using CyclicSimplexScheduler::fillAdditionalConstraintRow;
  using CyclicSimplexScheduler::fillConstraintRow;
  using CyclicSimplexScheduler::fillObjectiveRow;
  using CyclicSimplexScheduler::getParametricConstant;
  using CyclicSimplexScheduler::getStartTime;
  using CyclicSimplexScheduler::moveBy;
  using CyclicSimplexScheduler::multiplyRow;
  using CyclicSimplexScheduler::parameterS;
  using CyclicSimplexScheduler::parameterT;
  using CyclicSimplexScheduler::restoreDualFeasibility;
  using CyclicSimplexScheduler::schedule;
  using CyclicSimplexScheduler::scheduleAt;
  using CyclicSimplexScheduler::solveTableau;
  using CyclicSimplexScheduler::startTimeLocations;
  using CyclicSimplexScheduler::startTimeVariables;

  LogicalResult checkLastOp(ModuloProblem &problem, Operation *lastOp);
  enum { OBJ_LATENCY = 0, OBJ_AXAP /* i.e. either ASAP or ALAP */ };
  bool fillObjectiveRow(Problem &problem, Operation *lastOp,
                        SmallVector<int> &row, unsigned obj) override;
  void updateMargins();
  void scheduleOperation(ModuloProblem &problem, Operation *n);
  unsigned computeResMinII(SharedOperatorsProblem &problem);

public:
  ModuloSimplexScheduler()
      : CyclicSimplexScheduler(),
        SimplexSchedulerBase<ModuloSimplexScheduler, ModuloProblem>(),
        mrt(*this) {}
  LogicalResult schedule(ModuloProblem &problem, Operation *lastOp) override;
};

// This class solves the `ChainingProblem` by relying on pre-computed
// chain-breaking constraints.
class ChainingSimplexScheduler
    : public SimplexSchedulerBase<ChainingSimplexScheduler, ChainingProblem>,
      public Scheduler<ChainingProblem> {
  friend SimplexSchedulerBase<ChainingSimplexScheduler, ChainingProblem>;

protected:
  using SimplexSchedulerBase<ChainingSimplexScheduler,
                             ChainingProblem>::fillObjectiveRow;
  using SimplexSchedulerBase<ChainingSimplexScheduler,
                             ChainingProblem>::fillConstraintRow;
  using SimplexSchedulerBase<ChainingSimplexScheduler,
                             ChainingProblem>::fillAdditionalConstraintRow;
  void fillAdditionalConstraintRow(Problem &problem, SmallVector<int> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingSimplexScheduler() : SimplexSchedulerBase() {}
  LogicalResult schedule(ChainingProblem &problem, Operation *lastOp) override;
};

// This class solves the resource-free `ChainingCyclicProblem` by relying on
// pre-computed chain-breaking constraints. The optimal initiation interval (II)
// is determined as a side product of solving the parametric problem, and
// corresponds to the "RecMII" (= recurrence-constrained minimum II) usually
// considered as one component in the lower II bound used by modulo schedulers.
class ChainingCyclicSimplexScheduler
    : public SimplexSchedulerBase<ChainingCyclicSimplexScheduler,
                                  ChainingCyclicProblem>,
      public Scheduler<ChainingCyclicProblem> {

  friend class SimplexSchedulerBase<ChainingCyclicSimplexScheduler,
                                    ChainingCyclicProblem>;

protected:
  using SimplexSchedulerBase<ChainingCyclicSimplexScheduler,
                             ChainingCyclicProblem>::fillObjectiveRow;
  using SimplexSchedulerBase<ChainingCyclicSimplexScheduler,
                             ChainingCyclicProblem>::fillConstraintRow;
  void fillConstraintRow(ChainingCyclicProblem &problem, SmallVector<int> &row,
                         Problem::Dependence dep);
  void fillAdditionalConstraintRow(Problem &problem, SmallVector<int> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingCyclicSimplexScheduler() : SimplexSchedulerBase() {}
  LogicalResult schedule(ChainingCyclicProblem &problem,
                         Operation *lastOp) override;
};

}; // namespace circt::scheduling

#endif