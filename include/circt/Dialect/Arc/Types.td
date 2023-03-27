//===- Types.td - Arc dialect types ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_TYPES_TD
#define CIRCT_DIALECT_ARC_TYPES_TD

include "circt/Dialect/Arc/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class ArcTypeDef<string name> : TypeDef<ArcDialect, name> { }

def MemoryType : ArcTypeDef<"Memory"> {
  let mnemonic = "memory";
  let parameters = (ins "unsigned":$numWords, "::mlir::IntegerType":$wordType,
    OptionalParameter<"unsigned">:$stride);
  let assemblyFormat = "`<` $numWords `x` $wordType (`,` $stride^)? `>`";
}

#endif // CIRCT_DIALECT_ARC_TYPES_TD
