//===- ASAPScheduler.h - As-soon-as-possible list scheduler -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an acyclic, as-soon-as-possible list scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SCHEDULING_ASAPSCHEDULER_H
#define CIRCT_SUPPORT_SCHEDULING_ASAPSCHEDULER_H

#include "circt/Support/Scheduling/Scheduler.h"

namespace circt {
namespace sched {

struct ASAPScheduler : public virtual Scheduler {
  LogicalResult schedule() override;
};

} // namespace sched
} // namespace circt

#endif // CIRCT_SUPPORT_SCHEDULING_ASAPSCHEDULER_H
