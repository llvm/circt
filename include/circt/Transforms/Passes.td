//===-- SVPasses.td - SV pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVPASSES
#define CIRCT_DIALECT_SV_SVPASSES

include "mlir/Pass/PassBase.td"

def SimpleCanonicalizer : Pass<"simple-canonicalizer"> {
  let summary = "Run operation canonicalization patterns";
  let description = [{
      This is a lighter-weight version of the standard MLIR canonicalization
      pass that doesn't do CFG optimizations and has other differences.
  }];

  let constructor = "circt::createSimpleCanonicalizerPass()";
}

#endif // CIRCT_DIALECT_SV_SVPASSES
