//===- SchedulingInterfaces.td - Interface definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Scheduling dialect's interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHEDULING_SCHEDULINGINTERFACES_TD
#define CIRCT_DIALECT_SCHEDULING_SCHEDULINGINTERFACES_TD

include "mlir/IR/OpBase.td"

def SchedulableOpInterface : OpInterface<"SchedulableOpInterface"> {
  let description = [{
    Intended for operations that contain a graph region which should be
    statically scheduled.
  }];
  let cppNamespace = "::circt::sched";

  let methods = [
    InterfaceMethod<[{
        Construct the abstract scheduling problem (in terms of operations and
        their precedence relations) in the given scheduler implementation.
      }],
      "mlir::LogicalResult", "constructSchedulingProblem",
      (ins "circt::sched::SchedulerBase &":$scheduler)
    >
  ];
}

#endif // CIRCT_DIALECT_SCHEDULING_SCHEDULINGINTERFACES_TD
