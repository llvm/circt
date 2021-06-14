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

#ifndef CIRCT_SCHEDULING_SCHEDULER_H
#define CIRCT_SCHEDULING_SCHEDULER_H

#include "circt/Support/LLVM.h"

#include "mlir/Support/StorageUniquer.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace sched {

class Dependence;
class OperatorType;

/// This class models the most basic scheduling problem.
///
/// A problem instance is comprised of:
///
///  - *Operations*: The vertices in the data-flow graph to be scheduled. We use
///    the abbreviation `op` in the scheduling API.
///  - *Dependences*: The edges in the data-flow graph to be scheduled, modeling
///    a precedence relation between the involved operations. Individual operand
///    and result indices can be distinguished. Abbreviation: `dep`
///  - *Operator types*: An abstraction of the characteristics of the target
///    representation, e.g. modules in the  HLS component library, available
///    functional units, etc. -- operations are executed on instances/units of
///    an operator type. Abbreviation: `opr`
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
/// The `check...` methods perform validity checks before scheduling, e.g. all
/// operation handles referenced in dependences are registered, all ops have an
/// associated operator type, etc.
///
/// The `verify...` methods check the correctness of the solution determined by
/// a concrete scheduling algorithm, e.g. that there are start times available
/// for each registered operation, and the precedence constraints as modeled by
/// the dependences are satisfied.
class Scheduler {
public:
  /// Initialize a scheduling problem corresponding to \p containingOp.
  Scheduler(Operation *containingOp);
  virtual ~Scheduler() = default;

  //===--------------------------------------------------------------------===//
  // Aliases for containers storing the problem components and properties
  //===--------------------------------------------------------------------===//
protected:
  using OpSet = llvm::SetVector<Operation *>;
  using DepSet = llvm::SetVector<Dependence *>;
  using OprSet = llvm::SetVector<OperatorType *>;

  template <typename T>
  using OpProp = llvm::DenseMap<Operation *, T>;
  template <typename T>
  using DepProp = llvm::DenseMap<Dependence *, T>;
  template <typename T>
  using OprProp = llvm::DenseMap<OperatorType *, T>;
  template <typename T>
  using ProbProp = Optional<T>;

  //===--------------------------------------------------------------------===//
  // Containers for problem components and properties
  //===--------------------------------------------------------------------===//
private:
  // operation containing the ops for this scheduling problem. Used solely to
  // emit diagnostics.
  Operation *containingOp;

  // storage for dependences and operator types
  mlir::StorageUniquer uniquer;

  // problem components
  OpSet ops;
  DepSet deps;
  OprSet oprs;

  // operation properties
  OpProp<OperatorType *> assocOpr;
  OpProp<unsigned> startTime;

  // operator type properties
  OprProp<unsigned> latency;

  //===--------------------------------------------------------------------===//
  // Client interface to construct problem
  //===--------------------------------------------------------------------===//
public:
  /// Add \p op to the scheduling problem.
  void addOperation(Operation *op) { ops.insert(op); }

  /// Construct a dependence that distinguishes result and operands indices.
  Dependence *getOrInsertDependence(Operation *src, unsigned srcIdx,
                                    Operation *dst, unsigned dstIdx);

  /// Construct a dependence that uses the default result and operand indices.
  Dependence *getOrInsertDependence(Operation *src, Operation *dst) {
    return getOrInsertDependence(src, 0, dst, 0);
  }

  /// Register an operator type identified by \p name.
  OperatorType *getOrInsertOperatorType(StringRef name);

  //===--------------------------------------------------------------------===//
  // Subclass access to problem components
  //===--------------------------------------------------------------------===//
protected:
  bool hasOperation(Operation *op) { return ops.contains(op); }
  const OpSet &getOperations() { return ops; }

  bool hasDependence(Dependence *dep) { return deps.contains(dep); }
  const DepSet &getDependences() { return deps; }

  bool hasOperatorType(OperatorType *opr) { return oprs.contains(opr); }
  const OprSet &getOperatorTypes() { return oprs; }

  //===--------------------------------------------------------------------===//
  // Subclass access to properties: retrieve problem, set solution
  //===--------------------------------------------------------------------===//
protected:
  bool hasAssociatedOperatorType(Operation *op) { return assocOpr.count(op); }
  OperatorType *getAssociatedOperatorType(Operation *op) {
    return assocOpr.lookup(op);
  }

  bool hasLatency(OperatorType *opr) { return latency.count(opr); }
  unsigned getLatency(OperatorType *opr) { return latency.lookup(opr); }

  void setStartTime(Operation *op, unsigned val) { startTime[op] = val; }

  //===--------------------------------------------------------------------===//
  // Client access to properties: specify problem, retrieve solution
  //===--------------------------------------------------------------------===//
public:
  void setAssociatedOperatorType(Operation *op, OperatorType *opr) {
    assocOpr[op] = opr;
  }

  void setLatency(OperatorType *opr, unsigned val) { latency[opr] = val; }

  bool hasStartTime(Operation *op) { return startTime.count(op); }
  unsigned getStartTime(Operation *op) { return startTime.lookup(op); }

  //===--------------------------------------------------------------------===//
  // Hooks to check/verify the different problem components
  //===--------------------------------------------------------------------===//
protected:
  virtual LogicalResult checkOp(Operation *op);
  virtual LogicalResult checkDep(Dependence *dep);
  virtual LogicalResult checkOpr(OperatorType *opr);
  virtual LogicalResult checkProb();

  virtual LogicalResult verifyOp(Operation *op);
  virtual LogicalResult verifyDep(Dependence *dep);
  virtual LogicalResult verifyOpr(OperatorType *opr);
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

/// This class models a *dependence* from the `srcIdx`'th result of `src` to the
/// `destIdx`'th operand of `dst`. In other words, it represents an edge in a
/// scheduling problem.
class Dependence : public mlir::StorageUniquer::BaseStorage {
public:
  using KeyTy = std::tuple<Operation *, Operation *, unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return key == std::make_tuple(src, dst, srcIdx, dstIdx);
  }

  static Dependence *
  construct(mlir::StorageUniquer::StorageAllocator &allocator,
            const KeyTy &key);

  Operation *getSource() { return src; }
  unsigned getSourceIndex() { return srcIdx; }

  Operation *getDestination() { return dst; }
  unsigned getDestinationIndex() { return dstIdx; }

private:
  Dependence() = default;
  Operation *src, *dst;
  unsigned srcIdx, dstIdx;
};

/// This class models a distinct *operator type* in the context of a scheduling
/// problem. It is identified by name and serves as a handle to access the
/// problem-specific property maps.
class OperatorType : public mlir::StorageUniquer::BaseStorage {
public:
  using KeyTy = StringRef;

  bool operator==(const KeyTy &key) const { return key == name; }

  static OperatorType *
  construct(mlir::StorageUniquer::StorageAllocator &allocator,
            const KeyTy &key);

  StringRef getName() { return name; }

private:
  OperatorType() = default;
  StringRef name;
};

} // namespace sched
} // namespace circt

#endif // CIRCT_SCHEDULING_SCHEDULER_H
