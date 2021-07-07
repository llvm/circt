//===-- CalyxPasses.td - Calyx pass definition file --------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Calyx dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXPASSES_TD
#define CIRCT_DIALECT_CALYX_CALYXPASSES_TD

include "mlir/Pass/PassBase.td"

def GoInsertion : Pass<"calyx-go-insertion", "calyx::ComponentOp"> {
  let summary = "Inserts `go` signals into the guards of a group's non-hole assignments";
  let description = [{
    This passes inserts a group's `go` signal into the guards of all
    assignments housed within the group with exception of the "calyx.done"
    terminator. For example,

    Before:
    ```mlir
    calyx.group @Group1 {
      calyx.assign %1 = %2, %3 ? : i8
      calyx.done %4 : i1
    }
    ```

    After:
    ```mlir
    %0 = hw.constant 0 : i1
    ...
    calyx.group @Group1 {
      // The `go` assignment takes on a placeholder
      // value until the CompileControl pass.
      %5 = calyx.go %0 : i1

      %6 = comb.and %5, %3 : i1
      calyx.assign %1 = %2, %6 ? : i8

      calyx.done %4 : i1
    }
    ```
  }];
  let constructor = "circt::calyx::createGoInsertionPass()";
}

#endif // CIRCT_DIALECT_CALYX_CALYXPASSES_TD
