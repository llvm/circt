//===- ArcAttributes.td - Attributes for Arc dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCATTRIBUTES_TD
#define CIRCT_DIALECT_ARC_ARCATTRIBUTES_TD

include "circt/Dialect/Arc/ArcDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def TraceTapAttr : AttrDef<ArcDialect, "TraceTap"> {
  let mnemonic = "trace_tap";
  let summary = "TODO";
  let parameters = (ins "mlir::TypeAttr":$sigType, "uint64_t":$stateOffset,
      "mlir::ArrayAttr":$names);
  let assemblyFormat = "`<` $sigType `,` $stateOffset `,` $names `>`";
}

def TraceTapArrayAttr : TypedArrayAttrBase<TraceTapAttr, "TODO"> {}

#endif // CIRCT_DIALECT_ARC_ARCATTRIBUTES_TD
