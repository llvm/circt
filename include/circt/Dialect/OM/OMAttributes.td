//===- OMAttributes.td - Object Model dialect attributes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Object Model dialect type definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMATTRIBUTES_TD
#define CIRCT_DIALECT_OM_OMATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def ReferenceAttr : AttrDef<OMDialect, "Reference", [TypedAttrInterface]> {
  let summary = "An attribute that wraps a #hw.innerNameRef with !om.ref type";

  let mnemonic = "ref";

  let parameters = (ins
    "circt::hw::InnerRefAttr":$innerRef
  );

  let assemblyFormat = [{
    `<` $innerRef `>`
  }];

  let extraClassDeclaration = [{
    mlir::Type getType();
  }];
}

#endif // CIRCT_DIALECT_OM_OMATTRIBUTES_TD
