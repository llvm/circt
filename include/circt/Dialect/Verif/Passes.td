//===-- Passes.td - Verif pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the Verif dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VERIF_PASSES_TD
#define CIRCT_DIALECT_VERIF_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrepareForFormalPass : Pass<"prepare-for-formal", "hw::HWModuleOp"> {
  let summary = "Prepares a given top-level circuit for formal verification";
  let description = [{
    Prepares a circuit for formal verification. This performs semantic-retaining 
    transformations that make model checking easier, e.g. flattening wires,
    handling formal contracts, removing unwanted information such as OM Classes,
    etc.
  }];
}

def VerifyClockedAssertLikePass : Pass<"verify-clocked-assert-like",
                                       "hw::HWModuleOp"> {
  let summary = "Check that clocked-assert-like are valid";
  let description = [{
    Checks that every `clocked_assert`, `clocked_assume` or `clocked_cover` op
    does not contain a nested `ltl.clock` or `ltl.disable` operation in its
    operand tree.
  }];
}

def LowerFormalToHWPass : Pass<"lower-formal-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower verif.formal ops to hw.module ops";
  let description = [{
    Converts `verif.formal` ops into `hw.module` ops.
  }];
}

def LowerContractsPass : Pass<"lower-contracts", "mlir::ModuleOp"> {
  let summary = "Lower contracts into formal tests";
  let description = [{
    Converts `hw.module` ops containing a `verif.contract` into a
    `verif.formal` op.
  }];
}

def SimplifyAssumeEqPass : Pass<"simplify-assume-eq", "hw::HWModuleOp"> {
  let summary = "Use assume equal statements to simplify symbolic values";
  let description = [{
    When a symbolic value is assumed equal to another value, the symbolic value
    is replaced with its equal value to simplify.
  }];
}

#endif // CIRCT_DIALECT_VERIF_PASSES_TD
