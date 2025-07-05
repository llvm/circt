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

def LowerConcatRef : Pass<"moore-lower-concatref", "moore::SVModuleOp"> {
    let summary = "Lower moore.concat_ref ops";
    let description = [{
      It's used to disassemble the LHS of assignments that have a form like
      "{a, b} = c" onto "a = c[9001:42];" and "b = c[41:0]". Aimed at
      conveniently lowering this kind of assignment.
    }];
    let constructor = "circt::moore::createLowerConcatRefPass()";
}

def Vectorization : Pass<"moore-vectorization", "mlir::ModuleOp"> {
  let summary = "Vectorize consecutive scalar assignments";
  let description = [{
    This pass identifies sequences of `moore.continuous_assign` operations
    that represent a scalarized vector assignment (i.e., assigning bit by bit).
    It replaces these sequences with a single, more efficient vector-level
    `moore.continuous_assign` operation.
  }];
  let constructor = "circt::moore::createVectorizationPass()";
  let dependentDialects = ["circt::moore::MooreDialect"];
}

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD
