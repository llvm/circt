//===- SMTTypes.td - SMT dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTTYPES_TD
#define CIRCT_DIALECT_SMT_SMTTYPES_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "mlir/IR/AttrTypeBase.td"

class SMTTypeDef<string name> : TypeDef<SMTDialect, name> { }

def BoolType : SMTTypeDef<"Bool"> {
  let mnemonic = "bool";
  let assemblyFormat = "";
}

#endif // CIRCT_DIALECT_SMT_SMTTYPES_TD
