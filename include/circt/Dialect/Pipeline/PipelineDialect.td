//===- Pipeline.td - Pipeline dialect definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIPELINE_DIALECT_TD
#define CIRCT_DIALECT_PIPELINE_DIALECT_TD

include "mlir/IR/OpBase.td"

def Pipeline_Dialect : Dialect {
  let name = "pipeline";
  let cppNamespace = "::circt::pipeline";

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  let usePropertiesForAttributes = 0;

  let dependentDialects = [
    "circt::seq::SeqDialect"
  ];
}

#endif // CIRCT_DIALECT_PIPELINE_DIALECT_TD
