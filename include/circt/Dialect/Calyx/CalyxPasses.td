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

def RemoveGroups : Pass<"calyx-remove-groups", "calyx::ComponentOp"> {
  let summary = "Inlines the groups in a Calyx component.";
  let description = [{
    This pass removes the Group interface from the Calyx program, and inlines all
    assignments. This is done in the following manner:
    1. Assign values to the 'done' signal of the component, corresponding with the
       top-level control group's DoneOp. Add the 'go' signal of the component to all
       assignments.
    2. TODO(Calyx): If there are multiple writes to a signal, replace the reads with
       the disjunction.
    3. Remove all groups.
  }];
  let dependentDialects = ["comb::CombDialect"];
  let constructor = "circt::calyx::createRemoveGroupsPass()";
}

def CompileControl : Pass<"calyx-compile-control", "calyx::ComponentOp"> {
  let summary = "Generates latency-insensitive finite state machines to realize control.";
  let description = [{
    This pass performs a bottom-up traversal of the control program and does the following:
    1. For each control statement such as "calyx.seq", create a new GroupOp to contain all
       the structure to realize the schedule.
    2. Implement the schedule by setting the constituent groups' GoOp and DoneOp.
    3. Replace the control statement in the control program with the corresponding
       compilation group.
  }];
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect"];
  let constructor = "circt::calyx::createCompileControlPass()";
}

def GoInsertion : Pass<"calyx-go-insertion", "calyx::ComponentOp"> {
  let summary = "Insert go signals into the guards of a group's non-hole assignments";
  let description = [{
    This pass inserts the operation "calyx.group_go" into the guards of all
    assignments housed in the group, with the exception of the "calyx.group_done"
    terminator. For example,

    Before:
    ```mlir
    calyx.group @Group1 {
      calyx.assign %in = %out1, %guard ? : i8
      %done = calyx.group_done %out2 : i1
    }
    ```

    After:
    ```mlir
    // The `go` assignment takes on an undefined
    // value until the Compile Control pass.
    %undef = calyx.undef : i1
    ...
    calyx.group @Group1 {
      %go = calyx.group_go %undef : i1

      %and = comb.and %guard, %go : i1
      calyx.assign %in = %out1, %and ? : i8

      %done = calyx.group_done %out2 : i1
    }
    ```
  }];
  let dependentDialects = ["comb::CombDialect"];
  let constructor = "circt::calyx::createGoInsertionPass()";
}

#endif // CIRCT_DIALECT_CALYX_CALYXPASSES_TD
