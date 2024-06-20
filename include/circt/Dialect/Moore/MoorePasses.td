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

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD
