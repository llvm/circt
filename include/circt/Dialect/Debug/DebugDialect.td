//===- DebugDialect.td - Debug dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGDIALECT_TD
#define CIRCT_DIALECT_DEBUG_DEBUGDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def DebugDialect : Dialect {
  let name = "dbg";
  let summary = "Facilities to represent debug information";
  let cppNamespace = "circt::debug";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerOps();
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGDIALECT_TD
