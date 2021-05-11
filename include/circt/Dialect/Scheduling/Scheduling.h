//===- Scheduling.h - Scheduling dialect definition -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Scheduling dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHEDULING_H
#define CIRCT_DIALECT_SCHEDULING_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include <memory>

#include "circt/Dialect/Scheduling/SchedulingDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Scheduling/SchedulingAttributes.h.inc"

#include "circt/Dialect/Scheduling/SchedulableOpInterface.h.inc"

namespace circt {
namespace sched {

//===----------------------------------------------------------------------===//
// Interface definition for scheduling algorithms
//===----------------------------------------------------------------------===//

class SchedulerBase {
public:
  virtual ~SchedulerBase() = default;

  virtual mlir::LogicalResult
  schedule(circt::sched::SchedulableOpInterface schedulableOp) = 0;
  virtual mlir::Optional<unsigned>
  getStartTime(mlir::Operation *scheduledOp) = 0;
};

//===----------------------------------------------------------------------===//
// Concrete scheduler implementations
//===----------------------------------------------------------------------===//

std::unique_ptr<SchedulerBase> createASAPScheduler();

} // namespace sched
} // namespace circt

#endif // CIRCT_DIALECT_SCHEDULING_H
