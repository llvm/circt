//===- ASAPScheduler.h - As-soon-as-possible list scheduler -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an as-soon-as-possible list scheduler for acyclic problems.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_ASAPSCHEDULER_H
#define CIRCT_SCHEDULING_ASAPSCHEDULER_H

#include "circt/Scheduling/Scheduler.h"

namespace circt {
namespace scheduling {

/// This is a simple list scheduler for solving the basic scheduling problem.
/// Its objective is to assign each operation its earliest possible start time,
/// or in other words, to schedule each operation as soon as possible (hence the
/// name).
///
/// The dependence graph must not contain cycles.
struct ASAPScheduler : public virtual Scheduler {
  using Scheduler::Scheduler;
  LogicalResult schedule() override;
};

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_ASAPSCHEDULER_H
