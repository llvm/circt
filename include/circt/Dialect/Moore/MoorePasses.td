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

def DeleteLocalVar : Pass<"delete-local-var", "moore::SVModuleOp"> {
  let summary = "Delete local temporary variables";
  let description = [{
    This pass is aimed to delete local temporary variables and replace all
    uses by directly using their value.
  }];
  let constructor = "circt::moore::createDeleteLocalVarPass()";
}

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD
