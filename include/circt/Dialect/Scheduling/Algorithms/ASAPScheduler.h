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

#ifndef CIRCT_DIALECT_SCHEDULING_ALGORITHMS_ASAPSCHEDULER_H
#define CIRCT_DIALECT_SCHEDULING_ALGORITHMS_ASAPSCHEDULER_H

#include "circt/Dialect/Scheduling/Scheduler.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace sched {

class ASAPScheduler : public SchedulerBase {
private:
  llvm::SetVector<Operation *> operations;
  llvm::DenseMap<Operation *, SmallVector<Operation *, 4>> dependences;
  llvm::DenseMap<Operation *, OperatorInfoAttr> operators;
  llvm::DenseMap<Operation *, unsigned> startTimes;

public:
  LogicalResult registerOperation(Operation *op) override;
  LogicalResult registerDependence(Operation *src, unsigned int srcIdx,
                                   Operation *dst, unsigned int dstIdx,
                                   unsigned distance) override;
  LogicalResult
  registerOperators(Operation *op,
                    ArrayRef<OperatorInfoAttr> operatorInfos) override;
  LogicalResult schedule() override;
  Optional<unsigned> getStartTime(Operation *op) override;
};

} // namespace sched
} // namespace circt

#endif // CIRCT_DIALECT_SCHEDULING_ALGORITHMS_ASAPSCHEDULER_H
