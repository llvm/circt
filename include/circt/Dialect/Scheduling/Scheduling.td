//===- Scheduling.td - Scheduling dialect definition -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Scheduling dialect and its attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHEDULING_SCHEDULING_TD
#define CIRCT_DIALECT_SCHEDULING_SCHEDULING_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Scheduling_Dialect : Dialect {
  let name = "sched";
  let description = [{
    The `sched` dialect provides infrastructure for static operation schedulers.
  }];

  let cppNamespace = "::circt::sched";
}

//===----------------------------------------------------------------------===//
// Attribute definitions
//===----------------------------------------------------------------------===//

def Scheduling_OperatorInfo : AttrDef<Scheduling_Dialect, "OperatorInfo"> {
  let mnemonic = "opr";
  let description = [{
    This attribute captures operator information relevant for static schedulers.

    Example:
    ```
    %2 = mulf %0, %1 {opr = #sched.opr<"mul", latency=6>} : f32
    ```
  }];

  let parameters = (ins
   StringRefParameter<"the operator's name">:$name,
   AttrParameter<"unsigned", "the operator's result is available after "
                             "`latency` cycles">:$latency
  );

  let extraClassDeclaration = [{
    static llvm::StringRef getDefaultAttributeName() {
      return "opr";
    }
  }];
}

#endif // CIRCT_DIALECT_SCHEDULING_SCHEDULING_TD
