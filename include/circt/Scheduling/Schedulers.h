//===- Schedulers.h - Scheduler trait -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULERS_H
#define CIRCT_SCHEDULERS_H

#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/LogicalResult.h"
#include <type_traits>

namespace circt::scheduling {

/// This class provides an interface for schedulers. Schedulers are used to
/// schedule subclasses of `Problem`. A `Scheduler` can handle any number of
/// `Problem` types and declare these problem classes as template parameters.
/// For example: `class MyScheduler : public Scheduler<ChainingProblem,
/// CyclicProblem>` can schedule instances of both `ChainingProblem` and
/// `CyclicProblem`. A `Scheduler` must be able to handle at least one type of
/// `Problem`.
///
/// For every `Problem` class `P` managed by a `Scheduler`, the `Scheduler`
/// must define a method `LogicalResult schedule(P &problem, Operation *lastOp)`
/// to schedule an instance of `P`. The `lastOp` parameter represents the last
/// operation of the `Problem` instance and is used by predefined schedulers
/// in their schedule objectives.
///
/// Implementations of `Scheduler` are free to add additional parameters to the
/// `schedule` method if needed, for example, to support custom objective
/// functions. However, a common method signature is used for interoperability.
/// Functions that schedule a specific problem can accept the `Scheduler` as an
/// argument and be specialized accordingly. For example:
/// ```cpp
/// LogicalResult MyFunction(Scheduler<MyProblem> &scheduler) { ... }
/// ```
/// In this function, the `scheduler` argument can be used to schedule any
/// instance of `MyProblem` and can be called with any `Scheduler` that handles
/// `MyProblem`.
template <typename... Ps>
class Scheduler : public Scheduler<Ps>... {
  static_assert(
      sizeof...(Ps) > 0,
      "A scheduler must be able to schedule at least one class of Problem.");
};

template <typename P>
class Scheduler<P> {
  static_assert((std::is_base_of_v<Problem, P>),
                "Elements scheduled by a Scheduler must be deried from the "
                "Problem class.");

public:
  virtual ~Scheduler() = default;
  virtual LogicalResult schedule(P &problem, Operation *lastOp) = 0;
};

}; // namespace circt::scheduling

#endif