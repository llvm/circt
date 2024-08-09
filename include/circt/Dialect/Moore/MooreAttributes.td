//===- MooreAttributes.td - Moore attribute definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
#define CIRCT_DIALECT_MOORE_MOOREATTRIBUTES

include "circt/Dialect/Moore/MooreDialect.td"
include "mlir/IR/AttrTypeBase.td"

def FVIntegerAttr : AttrDef<MooreDialect, "FVInteger"> {
  let mnemonic = "fvint";
  let summary = "An attribute containing a four-valued integer";
  let parameters = (ins "FVInt":$value);
  let hasCustomAssemblyFormat = 1;
  let convertFromStorage = "$_self.getValue()";
  let returnType = "circt::FVInt";
}

#endif // CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
