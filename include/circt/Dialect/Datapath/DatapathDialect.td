//===- DatapathDialect.td - Datapath dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Datapath dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_DIALECT_TD
#define CIRCT_DIALECT_DATAPATH_DIALECT_TD

include "mlir/IR/OpBase.td"

def DatapathDialect : Dialect {
  let name = "datapath";

  let summary = "Datapath Optimization";
  let description = [{
    This dialect defines the `datapath` dialect, which is intended to be a 
    representation of low-level datapath optimization constructs.
  }];
  // let hasConstantMaterializer = 1;
  let dependentDialects = ["circt::comb::CombDialect"];
  let cppNamespace = "::circt::datapath";
}

#endif // CIRCT_DIALECT_DATAPATH_DIALECT_TD
