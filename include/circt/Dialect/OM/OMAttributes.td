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

def OMSymbolRefAttr : AttrDef<OMDialect, "SymbolRef", [TypedAttrInterface]> {
  let summary = "An attribute that wraps a FlatSymbolRefAttr type";

  let mnemonic = "sym_ref";

  let parameters = (ins
    "mlir::FlatSymbolRefAttr":$ref
  );

  let assemblyFormat = [{
    `<` $ref `>`
  }];

  let extraClassDeclaration = [{
    mlir::Type getType();
    // Get the SymbolRefAttr to the symbol represented by this operation.
    static om::SymbolRefAttr get(mlir::Operation *op);
    // Get the SymbolRefAttr to this symbol name.
    static om::SymbolRefAttr get(mlir::StringAttr symName);
  }];
}

#endif // CIRCT_DIALECT_OM_OMATTRIBUTES_TD
