//===- SVAttributes.td - Attributes for SV dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SV dialect specific attributes.
//
//===----------------------------------------------------------------------===//

def MacroIdentAttr : AttrDef<SVDialect, "MacroIdent"> {
  let summary = "Represents a reference to a macro identifier";
  let parameters = (ins "::mlir::StringAttr":$ident);
  let mnemonic = "macro.ident";

  let assemblyFormat = "$ident";

  let builders = [
    AttrBuilder<(ins "::llvm::StringRef":$ident), [{
      return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, ident));
    }]>,
  ];
  let extraClassDeclaration = [{
    ::llvm::StringRef getName() { return getIdent().getValue(); }
  }];
}
