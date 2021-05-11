//===- Scheduler.h - Common interface for scheduling algorithms -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the base class and utilities for scheduling algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHEDULING_SCHEDULER_H
#define CIRCT_DIALECT_SCHEDULING_SCHEDULER_H

#include "circt/Dialect/Scheduling/SchedulingAttributes.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace sched {

/// This is the base class for all scheduling algorithms, providing a common
/// interface for clients to construct a scheduling problem, and retrieve the
/// computed schedule afterwards.
class SchedulerBase {
public:
  virtual ~SchedulerBase() = default;

  /// Register a new operation in the scheduling problem.
  /// Returns failure if `op` is already registered.
  virtual LogicalResult registerOperation(Operation *op) = 0;

  /// Register a new dependence between `src` and `dst`:
  /// `src`'s result `srcIdx` must be available before `dst`'s operand `dstIdx`
  /// is required.
  /// A `distance` > 0 indicates a inter-iteration dependence, i.e. `dst`
  /// depends on the result of the `src` operation `distance` iterations/samples
  /// ago.
  /// Fails if `src` or `dst` are unregistered, or in case the implementation
  /// does not support operand/indices or inter-iteration dependences.
  virtual LogicalResult registerDependence(Operation *src, unsigned srcIdx,
                                           Operation *dst, unsigned dstIdx,
                                           unsigned distance) = 0;

  /// Register one or more suitable operators for `op`.
  /// Returns failure if `op` is unregistered, of in case the implementation
  /// does not support to choose from the given operators.
  virtual LogicalResult
  registerOperators(Operation *op,
                    ArrayRef<OperatorInfoAttr> operatorInfos) = 0;

  /// Compute the schedule.
  virtual LogicalResult schedule() = 0;

  /// Retrieves the computed start time for `op`, or `None` if `op` is unknown.
  virtual Optional<unsigned> getStartTime(Operation *op) = 0;
};

} // namespace sched
} // namespace circt

#endif // CIRCT_DIALECT_SCHEDULING_SCHEDULER_H
