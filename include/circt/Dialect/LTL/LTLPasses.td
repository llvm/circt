//===-- Passes.td - LTL pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the LTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_PASSES_TD
#define CIRCT_DIALECT_LTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def InferLTLClocks : Pass<"ltl-infer-clocks", "hw::HWModuleOp"> {
  let summary =
      "Infer clock for ltl.delay operations from surrounding ltl.clock ops";
  let description = [{
    This pass propagates clock information from `ltl.clock` operations down
    to `ltl.delay` operations in their input chains. After FIRRTLToHW lowering,
    `ltl.delay` operations may have placeholder clocks (constant true) because
    the FIRRTL delay intrinsic doesn't carry explicit clock information.
    
    This pass walks through each `ltl.clock` operation, finds all `ltl.delay`
    operations in its input chain that have placeholder clocks, and replaces
    them with correctly clocked versions using the clock from the enclosing
    `ltl.clock` operation.
    
    Example transformation:
    ```mlir
    // Before:
    %true = hw.constant true
    %d = ltl.delay %true, posedge, %a, 1, 0 : i1
    %p = ltl.clock %d, posedge %clk : !ltl.sequence
    
    // After:
    %d = ltl.delay %clk, posedge, %a, 1, 0 : i1
    %p = ltl.clock %d, posedge %clk : !ltl.sequence
    ```
  }];
  let dependentDialects = ["hw::HWDialect", "ltl::LTLDialect"];
}

#endif // CIRCT_DIALECT_LTL_PASSES_TD
