//===- Problems.h - Modeling of scheduling problems -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static scheduling algorithms for use in HLS flows all solve similar problems.
// The classes in this file serve as an interface between clients and algorithm
// implementations, and model a basic scheduling problem and some commonly used
// extensions (e.g. modulo scheduling). This includes problem-specific
// verification methods.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_PROBLEMS_H
#define CIRCT_SCHEDULING_PROBLEMS_H

#include "circt/Scheduling/DependenceIterator.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Identifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace scheduling {

/// This class models the most basic scheduling problem.
///
/// A problem instance is comprised of:
///
///  - *Operations*: The vertices in the data-flow graph to be scheduled.
///  - *Dependences*: The edges in the data-flow graph to be scheduled, modeling
///    a precedence relation between the involved operations.
///  - *Operator types*: An abstraction of the characteristics of the target
///    representation, e.g. modules in the  HLS component library, available
///    functional units, etc. -- operations are executed on instances/units of
///    an operator type.
///
/// Operations and operator types are stored explicitly. The registered
/// operations induce a subgraph of the SSA graph. We implicitly include the
/// dependences corresponding to its *def-use* relationships in the problem,
/// e.g. if operation *y*'s second operand uses the first result produced by
/// *x*, we'd have a dependence `x:0 --> y:1`. Clients can additionally register
/// explicit, *auxiliary* dependence between operations, e.g. to encode memory
/// dependencies or other ordering constraints. Auxiliary dependences do not
/// distinguish specific operands/results. The differences between the flavors
/// are transparent to concrete algorithms.
///
/// All components of the problem (operations, dependences, operator types, as
/// well as the problem itself) can be annotated with *properties*. In this
/// basic problem, we model
///
/// - `linkedOperatorType` maps operations to their operator types.
/// - `latency`, an operator type-property denoting the number of time steps
///   after which the operator's result is available.
/// - `startTime`, an operation-property for the time step in which an operation
///   is started. Together, the start times for all operations represent the
///   problem's solution, i.e. the schedule.
///
/// Subclasses, i.e. corresponding to more complex scheduling problems, can
/// declare additional properties as needed. The `clearSolution` method must be
/// overridden if the new properties are part of the solution.
//
/// The `check...` methods perform validity checks before scheduling, e.g. that
/// all operations have an associated operator type, etc.
///
/// The `verify...` methods check the correctness of the solution determined by
/// a concrete scheduling algorithm, e.g. that there are start times available
/// for each registered operation, and the precedence constraints as modeled by
/// the dependences are satisfied.
class Problem {
public:
  /// Initialize a scheduling problem corresponding to \p containingOp.
  Problem(Operation *containingOp) : containingOp(containingOp) {}
  virtual ~Problem() = default;

  friend detail::DependenceIterator;

  //===--------------------------------------------------------------------===//
  // Aliases for the problem components
  //===--------------------------------------------------------------------===//
public:
  /// A thin wrapper to allow a uniform handling of def-use and auxiliary
  /// dependences.
  using Dependence = detail::Dependence;

  /// Operator types are distinguished by name (chosen by the client).
  using OperatorType = mlir::Identifier;

  //===--------------------------------------------------------------------===//
  // Aliases for containers storing the problem components and properties
  //===--------------------------------------------------------------------===//
public:
  using OperationSet = llvm::SetVector<Operation *>;
  using DependenceRange = llvm::iterator_range<detail::DependenceIterator>;
  using OperatorTypeSet = llvm::SetVector<OperatorType>;

protected:
  using AuxDependenceMap =
      llvm::DenseMap<Operation *, llvm::SmallSetVector<Operation *, 4>>;

  template <typename T>
  using OperationProperty = llvm::DenseMap<Operation *, Optional<T>>;
  template <typename T>
  using DependenceProperty = llvm::DenseMap<Dependence, Optional<T>>;
  template <typename T>
  using OperatorTypeProperty = llvm::DenseMap<OperatorType, Optional<T>>;
  template <typename T>
  using ProblemProperty = Optional<T>;

  //===--------------------------------------------------------------------===//
  // Containers for problem components and properties
  //===--------------------------------------------------------------------===//
private:
  // Operation containing the ops for this scheduling problem. Used for its
  // MLIRContext and to emit diagnostics.
  Operation *containingOp;

  // Problem components
  OperationSet operations;
  AuxDependenceMap auxDependences;
  OperatorTypeSet operatorTypes;

  // Operation properties
  OperationProperty<OperatorType> linkedOperatorType;
  OperationProperty<unsigned> startTime;

  // Operator type properties
  OperatorTypeProperty<unsigned> latency;

  //===--------------------------------------------------------------------===//
  // Problem construction
  //===--------------------------------------------------------------------===//
public:
  /// Include \p op in this scheduling problem.
  void insertOperation(Operation *op) { operations.insert(op); }

  /// Include \p dep in the scheduling problem. Return failure if \p dep does
  /// not represent a valid def-use or auxiliary dependence between operations.
  /// The endpoints become registered operations w.r.t. the problem.
  LogicalResult insertDependence(Dependence dep);

  /// Include \p opr in this scheduling problem.
  void insertOperatorType(OperatorType opr) { operatorTypes.insert(opr); }

  /// Retrieves the operator type identified by the client-specific \p name. The
  /// operator type is automatically registered in the scheduling problem.
  OperatorType getOrInsertOperatorType(StringRef name);

  //===--------------------------------------------------------------------===//
  // Access to problem components
  //===--------------------------------------------------------------------===//
public:
  /// Return the operation containing this problem, e.g. to emit diagnostics.
  Operation *getContainingOp() { return containingOp; }

  /// Return true if \p op is part of this problem.
  bool hasOperation(Operation *op) { return operations.contains(op); }
  /// Return the set of operations.
  const OperationSet &getOperations() { return operations; }

  /// Return a range object to transparently iterate over \p op's *incoming*
  ///  1) implicit def-use dependences (backed by the SSA graph), and then
  ///  2) explictly added auxiliary dependences.
  ///
  /// In other words, this yields dependences whose destination operation is
  /// \p op, and whose source operations are \p op's predecessors in the problem
  /// graph.
  ///
  /// To iterate over all of the scheduling problem's dependences, simply
  /// process the ranges for all registered operations.
  DependenceRange getDependences(Operation *op);

  /// Return true if \p opr is part of this problem.
  bool hasOperatorType(OperatorType opr) { return operatorTypes.contains(opr); }
  /// Return the set of operator types.
  const OperatorTypeSet &getOperatorTypes() { return operatorTypes; }

  //===--------------------------------------------------------------------===//
  // Access to properties
  //===--------------------------------------------------------------------===//
public:
  /// The linked operator type provides the runtime characteristics for \p op.
  Optional<OperatorType> getLinkedOperatorType(Operation *op) {
    return linkedOperatorType.lookup(op);
  }
  void setLinkedOperatorType(Operation *op, OperatorType opr) {
    linkedOperatorType[op] = opr;
  }

  /// The latency is the number of cycles \p opr needs to compute its result.
  Optional<unsigned> getLatency(OperatorType opr) {
    return latency.lookup(opr);
  }
  void setLatency(OperatorType opr, unsigned val) { latency[opr] = val; }

  /// Return the start time for \p op, as computed by the scheduler.
  /// These start times comprise the basic problem's solution, i.e. the
  /// *schedule*.
  Optional<unsigned> getStartTime(Operation *op) {
    return startTime.lookup(op);
  }
  void setStartTime(Operation *op, unsigned val) { startTime[op] = val; }

  /// Clear all properties that are part of the solution.
  virtual void clearSolution() { startTime.clear(); }

  //===--------------------------------------------------------------------===//
  // Hooks to check/verify the different problem components
  //===--------------------------------------------------------------------===//
protected:
  virtual LogicalResult checkOperation(Operation *op);
  virtual LogicalResult checkDependence(Dependence dep);
  virtual LogicalResult checkOperatorType(OperatorType opr);
  virtual LogicalResult checkProblem();

  virtual LogicalResult verifyOperation(Operation *op);
  virtual LogicalResult verifyDependence(Dependence dep);
  virtual LogicalResult verifyOperatorType(OperatorType opr);
  virtual LogicalResult verifyProblem();

public:
  /// Return success if the constructed scheduling problem is valid.
  LogicalResult check();
  /// Return success if the computed solution is valid.
  LogicalResult verify();
};

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_PROBLEMS_H
