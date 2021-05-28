//===- Scheduler.h - Common interface for scheduling algorithms -*- C++ -*-===//
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
// extensions (e.g. modulo scheduling). This includes problem-specific utility
// and verification methods.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SCHEDULING_SCHEDULER_H
#define CIRCT_SUPPORT_SCHEDULING_SCHEDULER_H

#include "circt/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace sched {

/// This class models the most basic scheduling problem.
///
/// A problem instance is comprised of:
///
///  - *Operations*: The vertices in the data-flow graph to be scheduled.
///    Using `mlir::Operation*` is the obvious mapping, but other data types can
///    be used as the scheduler treats them as opaque handles. We use the
///    abbreviation `op` to mean operation (handle) in the scheduling API.
///  - *Dependences*: The edges in the data-flow graph to be scheduled, modeling
///    a precedence relation between the involved operations. Individual operand
///    and result indices can be distinguished. We use the abbreviation `dep` to
///    in the scheduling API.
///  - *Operator types*: An abstraction of the characteristics of the target
///    representation, e.g. modules in the  HLS component library, available
///    functional units, etc. -- operations are executed on instances/units of
///    an operator type. Again, these are opaque handles for the scheduler. We
///    use `opr` to denote operator type handles in the API.
///
/// All components of the problem (ops, deps, oprs, as well as the problem
/// itself) can be annotated with *properties*. In this basic problem, we model
///
/// - `assocOpr`, an operation-property that stores the operator type associated
///   with each operation.
/// - `latency`, an operator type-property denoting the number of time steps
///   after which the operator's result is available.
/// - `startTime`, an operation-property for the time step in which an operation
///   is started. Together, the start times for all operations represent the
///   problem's solution, i.e. the schedule.
///
/// Subclasses, i.e. base classes for more complex scheduling problems as well
/// as concrete algorithms, can declare additional properties as needed. The
/// top-level entrypoint for algorithms is the pure virtual `schedule()` method.
//
/// The `check...` methods perform validity checks before scheduler, e.g. all
/// operation handles referenced in dependences are registered, all ops have an
/// associated operator type, etc.
///
/// The `verify...` methods check the correctness of the solution determined by
/// a concrete scheduling algorithm, e.g. that there are start times available
/// for each registered operation, and the precedence constraints as modeled by
/// the dependences are satisfied.
class Scheduler {
  //===--------------------------------------------------------------------===//
  // Aliases for API types
  //===--------------------------------------------------------------------===//
public:
  /// Opaque handle for operations
  using OperationHandle = void *;
  /// Dependences are tuples: (from, fromResultIdx, to, toOperandIdx)
  using Dependence =
      std::tuple<OperationHandle, unsigned, OperationHandle, unsigned>;
  /// Opaque handle for operator types
  using OperatorTypeHandle = unsigned;

  static Dependence makeDependence(OperationHandle from, unsigned fromResultIdx,
                                   OperationHandle to, unsigned toOperandIdx) {
    return std::make_tuple(from, fromResultIdx, to, toOperandIdx);
  }

  static Dependence makeDependence(OperationHandle from, OperationHandle to) {
    return makeDependence(from, 0, to, 0);
  }

  virtual ~Scheduler() = default;

  //===--------------------------------------------------------------------===//
  // Aliases for containers storing the problem components and properties
  //===--------------------------------------------------------------------===//
protected:
  using OpSet = llvm::SetVector<OperationHandle>;
  using DepSet = llvm::SetVector<Dependence>;
  using OprSet = llvm::SetVector<OperatorTypeHandle>;

  template <typename T>
  using OpProp = llvm::DenseMap<OperationHandle, T>;
  template <typename T>
  using DepProp = llvm::DenseMap<Dependence, T>;
  template <typename T>
  using OprProp = llvm::DenseMap<OperatorTypeHandle, T>;
  template <typename T>
  using ProbProp = Optional<T>;

  //===--------------------------------------------------------------------===//
  // Containers for problem components and properties
  //===--------------------------------------------------------------------===//
private:
  // problem components
  OpSet ops;
  DepSet deps;
  OprSet oprs;

  // operation properties
  OpProp<OperatorTypeHandle> assocOpr;
  OpProp<unsigned> startTime;

  // operator type properties
  OprProp<unsigned> latency;

  //===--------------------------------------------------------------------===//
  // Client interface to construct problem
  //===--------------------------------------------------------------------===//
public:
  void registerOperation(OperationHandle op) { ops.insert(op); }
  void registerDependence(Dependence dep) { deps.insert(dep); }
  void registerOperatorType(OperatorTypeHandle opr) { oprs.insert(opr); }

  //===--------------------------------------------------------------------===//
  // Subclass access to problem components
  //===--------------------------------------------------------------------===//
protected:
  bool hasOperation(OperationHandle op) { return ops.contains(op); }
  const OpSet &getOperations() { return ops; }

  bool hasDependence(Dependence dep) { return deps.contains(dep); }
  const DepSet &getDependences() { return deps; }

  bool hasOperatorType(OperatorTypeHandle opr) { return oprs.contains(opr); }
  const OprSet &getOperatorTypes() { return oprs; }

  //===--------------------------------------------------------------------===//
  // Subclass access to properties: retrieve problem, set solution
  //===--------------------------------------------------------------------===//
protected:
  bool hasAssociatedOperatorType(OperationHandle op) {
    return assocOpr.count(op);
  }
  OperatorTypeHandle getAssociatedOperatorType(OperationHandle op) {
    return assocOpr.lookup(op);
  }

  bool hasLatency(OperatorTypeHandle opr) { return latency.count(opr); }
  unsigned getLatency(OperatorTypeHandle opr) { return latency.lookup(opr); }

  void setStartTime(OperationHandle op, unsigned val) { startTime[op] = val; }

  //===--------------------------------------------------------------------===//
  // Client access to properties: specify problem, retrieve solution
  //===--------------------------------------------------------------------===//
public: //
  void setAssociatedOperatorType(OperationHandle op, OperatorTypeHandle opr) {
    assocOpr[op] = opr;
  }

  void setLatency(OperatorTypeHandle opr, unsigned val) { latency[opr] = val; }

  bool hasStartTime(OperationHandle op) { return startTime.count(op); }
  unsigned getStartTime(OperationHandle op) { return startTime.lookup(op); }

  //===--------------------------------------------------------------------===//
  // Hooks to check/verify the different problem components
  //===--------------------------------------------------------------------===//
protected:
  virtual LogicalResult checkOp(OperationHandle op);
  virtual LogicalResult checkDep(Dependence dep);
  virtual LogicalResult checkOpr(OperatorTypeHandle opr);
  virtual LogicalResult checkProb();

  virtual LogicalResult verifyOp(OperationHandle op);
  virtual LogicalResult verifyDep(Dependence dep);
  virtual LogicalResult verifyOpr(OperatorTypeHandle opr);
  virtual LogicalResult verifyProb();

public:
  /// Return success if the constructed scheduling problem is valid.
  LogicalResult check();
  /// Return success if the computed solution is valid.
  LogicalResult verify();

  //===--------------------------------------------------------------------===//
  // Entry point to actual algorithm
  //===--------------------------------------------------------------------===//
public:
  // Attempt to schedule the constructed problem.
  virtual LogicalResult schedule() = 0;
};

} // namespace sched
} // namespace circt

#endif // CIRCT_SUPPORT_SCHEDULING_SCHEDULER_H
