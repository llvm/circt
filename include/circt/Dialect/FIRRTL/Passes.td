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

#endif // CIRCT_DIALECT_FIRRTL_PASSES_TD
