//===- SMTDialect.td - SMT dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTDIALECT_TD
#define CIRCT_DIALECT_SMT_SMTDIALECT_TD

include "mlir/IR/DialectBase.td"

def SMTDialect : Dialect {
  let name = "smt";
  let summary = "a dialect that models satisfiability modulo theories";
  let cppNamespace = "circt::smt";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_SMT_SMTDIALECT_TD
