//===--- Passes.td - Moore pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREPASSES_TD
#define CIRCT_DIALECT_MOORE_MOOREPASSES_TD

include "mlir/Pass/PassBase.td"

def SimplifyProcedures : Pass<"moore-simplify-procedures", "moore::SVModuleOp"> {
    let summary = "Simplify procedures";
    let description = [{
      Because we want to introduce mem2reg in the moore dialect to eliminate the
      local temporary variables, if the local variabels exist in the procedure
      body, it can be promoted by mem2reg. But global/module-level variables
      don't be promoted. So this pass is aimed at inserting a local "shadow"
      variable in always blocks for every module-level variable that the
      process modifies. Then perform mem2reg to eliminate the local "shadow"
      variable.
    }];
    let constructor = "circt::moore::createSimplifyProceduresPass()";
}

def SimplifyRefs : Pass<"moore-simplify-refs", "moore::SVModuleOp"> {
    let summary = "Lower moore.concat_ref ops and moore.dyn_queue_ref_element";
    let description = [{
      It has two purposes:
      - To disassemble the moore.concat_ref. Which is tricky to lower
      directly. For example, disassemble "{a, b} = c" onto "a = c[7:3]"
      and "b = c[2:0]".
      - To eliminate moore.dyn_queue_ref_element in the case where the reference
      immediately has a value assigned via blocking assignment, replacing it with
      moore.queue.set. Queue element references are tricky to lower into LLHD,
      so it's best to get rid of them.
    }];
    let constructor = "circt::moore::createSimplifyRefsPass()";
}

def CreateVTables : Pass<"moore-create-vtables", "mlir::ModuleOp"> {
    let summary = "Lower moore.vtable ops";
    let description = [{
        It's used to compute vtables from classdeclops.
        Needs to be run before any Symbol DCE pass is run.
    }];
    let constructor = "circt::moore::createVTablesPass()";
}


#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD
