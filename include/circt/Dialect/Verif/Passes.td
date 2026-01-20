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

def CombineAssertLikePass : Pass<"combine-assert-like", "hw::HWModuleOp"> {
  let summary = "Combines all assertions and assumptions into single operations";
  let description = [{
    Combines all assertions and assumptions by conjoining their conditions into a
    single assertion and assumption respectively, e.g.
    ```mlir
    // [...]
    verif.assume %pred0 : i1
    verif.assume %pred1 : i1
    // [...]
    verif.assert %cond0 : i1
    verif.assert %cond1 : i1
    ```
    will be converted into  
    ```
    // [...]
    %pred = comb.and bin %pred0, %pred1 : i1
    verif.assume %pred : i1
    // [...]
    %cond = comb.and bin %cond0, %cond1 : i1
    verif.assert %cond : i1
    ```
    This is already implicitly done in many of down stream tools, and being 
    able to simplify assertions early can help will other verification tasks 
    such as contract inference.

    The conjunctions are created on a per-block basis, so asserlike ops in 
    separate blocks will not be merged.
  }];
}

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

def LowerTestsPass : Pass<"verif-lower-tests", "mlir::ModuleOp"> {
  let summary = "Convert Verif tests into HW modules";
  let description = [{
    Converts `verif.formal` and `verif.simulation` operations into top-level
    `hw.module` operations that can be fed to formal verification and simulation
    tools.
  }];
}

def LowerContractsPass : Pass<"lower-contracts", "mlir::ModuleOp"> {
  let summary = "Lower contracts into formal tests";
  let description = [{
    For `hw.module` ops containing any `verif.contract` ops, apply the
    contracts in the `hw.module` body and emit a `verif.formal` op that checks
    each applied contract.

    Contracts are checked by turning `require`s into `assume`s and `ensure`s
    into `asserts`s.  Inputs to the `hw.module` op are replaced with
    `verif.symbolic_value` ops and other contracts are assumed to hold by
    inlining the contract body.

    Contracts are applied by turning `require`s into `assert`s and `ensure`s
    into `assume`s in the original `hw.module` op body.  Results of the
    contract are replaced with `verif.symbolic_value` ops.  In some cases,
    using the `SimplifyAssumeEqPass` after lowering contracts and before
    canonicalization and cse can result in a greatly simplified `hw.module` op
    body.
  }];
}

def LowerSymbolicValuesPass : Pass<"verif-lower-symbolic-values",
                                   "mlir::ModuleOp"> {
  let summary = "Lower symbolic values to blackbox instances or wires";
  let description = [{
    Converts `verif.symbolic_value` ops into one of the following
    representations that formal tools can work with:

    - Instances of an external module, which most formal tools can treat as an
      internal port boundary with symbolic values assigned to the module
      outputs.
    - Wire declarations marked with `(* anyseq *)`, which Yosys treats like a
      top-level input port.
  }];
  let options = [
    Option<
      "mode", "mode",
      "SymbolicValueLowering", "SymbolicValueLowering::ExtModule",
      "Control how symbolic values are lowered",
      "symbolicValueLoweringCLValues()"
    >,
  ];
  let dependentDialects = [
    "hw::HWDialect",
    "sv::SVDialect",
  ];
}

def SimplifyAssumeEqPass : Pass<"simplify-assume-eq", "hw::HWModuleOp"> {
  let summary = "Use assume equal statements to simplify symbolic values";
  let description = [{
    When a symbolic value is assumed equal to another value, the symbolic value
    is replaced with its equal value to simplify.
  }];
}

def StripContractsPass : Pass<"strip-contracts"> {
  let summary = "Remove contracts from the IR";
  let description = [{
    Replaces all `verif.contract` ops in the IR with their operands, making them
    act as simple passthroughs.
  }];
}

#endif // CIRCT_DIALECT_VERIF_PASSES_TD
