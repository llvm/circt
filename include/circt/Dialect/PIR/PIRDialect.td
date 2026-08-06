//===- PIRDialect.td - Property IR dialect definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIR_PIRDIALECT_TD
#define CIRCT_DIALECT_PIR_PIRDIALECT_TD

include "mlir/IR/OpBase.td"

def PIRDialect : Dialect {
  let name = "pir";
  let summary = "Property IR ingress dialect, used to encode SystemVerilog Assertion properties and sequences.";
  // See `docs/Dialect/PIR.md` for detailed dialect documentation.
  let cppNamespace = "circt::pir";
}

#endif // CIRCT_DIALECT_PIR_PIRDIALECT_TD
