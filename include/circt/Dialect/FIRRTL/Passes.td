//===-- Passes.td - FIRRTL pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_PASSES_TD
#define CIRCT_DIALECT_FIRRTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerFIRRTLTypes : Pass<"firrtl-lower-types", "firrtl::CircuitOp"> {
  let summary = "Lower FIRRTL types to ground types";
  let description = [{
    Lower firrtl.module port types to ground types.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLTypesPass()";
}

def IMConstProp : Pass<"firrtl-imconstprop", "firrtl::CircuitOp"> {
  let summary = "Intermodule constant propagation and dead code elimination";
  let description = [{
    Use optimistic constant propagation to delete ports and unreachable IR.
  }];
  let constructor = "circt::firrtl::createIMConstPropPass()";
}

def BlackboxMemory : Pass<"firrtl-blackbox-memory", "firrtl::CircuitOp"> {
  let summary = "Replace all FIRRTL memories with an external module blackbox.";
  let description = [{
    This pass replaces all sequential memory operations with an external module
    blackbox.  For each memory operation, it creates a FModuleOp which
    replicates the return types of the memory operation, and replaces the
    MemoryOp with an instance of the module.  Inside the new module, an
    instance of an FExtModule blackbox is created.  The blackbox module must
    use the same parameter naming conventions used by the ReplaceSeqMemories
    pass in the Scala FIRRTL compiler.
  }];
  let constructor = "circt::firrtl::createBlackBoxMemoryPass()";
  let options = [
    Option<"emitWrapper", "emit-wrapper", "bool", "true",
           "Create a wrapper module around the blackbox external module.">
  ];
}

def ExpandWhens : Pass<"firrtl-expand-whens", "firrtl::FModuleOp"> {
  let summary = "Remove all when conditional blocks.";
  let description = [{
    This pass will:
    1. Resolve last connect semantics.
    2. Remove all when operations.

    When a wire has multiple connections, only the final connection is used,
    all previous connections are overwritten. When there is a conditional
    connect, the previous connect is only overwritten when the condition
    holds:

    ```firrtl
    w <= a
    when c :
      w <= b

    ; Equivalent to:
    w <= mux(c, b, a)
    ```

    This pass requires that all connects are expanded.
  }];
  let constructor = "circt::firrtl::createExpandWhensPass()";
}

#endif // CIRCT_DIALECT_FIRRTL_PASSES_TD
