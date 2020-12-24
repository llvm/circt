//===-- Passes.td - FIRRTL pass definition file ------------*- tablegen -*-===//
//
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

def LowerFIRRTLTypes : Pass<"firrtl-lower-types", "firrtl::FModuleOp"> {
  let summary = "Lower FIRRTL types to ground types";
  let description = [{
    Lower firrtl.module port types to ground types.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLTypesPass()";
}

#endif // CIRCT_DIALECT_FIRRTL_PASSES_TD
