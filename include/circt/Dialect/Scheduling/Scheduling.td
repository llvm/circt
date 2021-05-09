//===- Scheduling.td - Scheduling dialect definition -------*- tablegen -*-===//
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

#ifndef CIRCT_DIALECT_SCHEDULING_TD
#define CIRCT_DIALECT_SCHEDULING_TD

include "mlir/IR/OpBase.td"

def Scheduling_Dialect : Dialect {
  let name = "sched";
  let summary = "The `sched` dialect provides an interface and infrastructure for static operation schedulers.";
  let description = [{
    This is a work-in-progress.

    * `SchedulableOpInterface` is implemented by operations containing a graph
      region that should be scheduled. Such a schedulable operation must provide
      _operator_ and _dependence_ information for their nested operations.

    * `SchedulerTrait` is attached to implementations of concrete scheduling
      algorithms, and provides a consistent interface as well as commonly needed
      utilities.
  }];

  let cppNamespace = "::circt::sched";
}

def Scheduling_OperatorInfo : AttrDef<Scheduling_Dialect, "OperatorInfo"> {
  let mnemonic = "opr";
  let description = [{
    This attribute captures operator information relevant for static schedulers.

    Example:
    ```
    %2 = addf %0, %1 {opr = #sched.opr<"mul", latency=6>} : f32
    ```
  }];

  let parameters = (ins
   StringRefParameter<"the operator's name">:$name,
   AttrParameter<"unsigned", "the operator's result is available after `latency` cycles">:$latency
  );
}

#endif // CIRCT_DIALECT_SCHEDULING_TD
