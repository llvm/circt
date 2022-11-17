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
  let summary = "Macro identifier";
  let description = [{
    Represents a reference to a macro identifier.
  }];
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

def SVAttributeAttr : AttrDef<SVDialect, "SVAttribute"> {
  let summary = "a Verilog Attribute";
  let mnemonic = "attribute";
  let description = [{
    This attribute is used to encode a Verilog _attribute_.  A Verilog attribute
    (not to be confused with an LLVM or MLIR attribute) is a syntactic mechanism
    for adding metadata to specific declarations, statements, and expressions in
    the Verilog language.  _There are no "standard" attributes_.  Specific tools
    define and handle their own attributes.

    Verilog attributes have a mandatory name and an optional constant
    expression.  This is encoded as a key (name) value (expression) pair.
    Multiple attributes may be specified, either with multiple separate
    attributes or by comman-separating name--expression pairs.

    Currently, SV attributes don't block most optimizations; therefore, users should
    not expect that sv attributes always appear in the output verilog.
    However, we must block optimizations that updating ops in-place
    since it is mostly invalid to transfer SV attributes one to another.

    For more information, refer to Section 5.12 of the SystemVerilog (1800-2017)
    specification.
  }];
  let parameters = (ins "::mlir::StringAttr":$name,
                        OptionalParameter<"::mlir::StringAttr">:$expression);

  let assemblyFormat = [{ `<` $name ( `=` $expression^ )? `>` }];
  let extraClassDeclaration = [{
    static inline llvm::StringRef
        getSVAttributesAttrName() { return "sv.attributes"; }
  }];
}

def SVAttributesAttr : AttrDef<SVDialect, "SVAttributes"> {
  let summary = "A container of system verilog attributes";
  let mnemonic = "attributes";
  let description = [{
    This attribute is used to store SV attributes. The `attributes` field
    represents SV attributes we want to annotate. The `emitAsComments` field
    controls its emission style: SV spec defines the syntax of SV attributes as
    `(* identifier (= identifer)? *)`. However, unfortunately many vendor-specific
    pragmas violate the syntax and they are intended to be attached as comments.
    Hence we emit given attributes as comments if `emitAsComments` field is true.
  }];
  let parameters = (ins "mlir::ArrayAttr":$attributes,
                      DefaultValuedParameter<"mlir::BoolAttr",
                      "mlir::BoolAttr::get($_ctxt, false)">:$emitAsComments);
  let builders = [
    AttrBuilder<(ins "llvm::ArrayRef<llvm::StringRef>":$strings,
                     CArg<"bool", "false">: $emitAsComments), [{
      SmallVector<Attribute> attrs;
      for (auto str : strings)
        attrs.push_back(sv::SVAttributeAttr::get(
            $_ctxt, ::mlir::StringAttr::get($_ctxt, str), ::mlir::StringAttr()));
      return $_get($_ctxt, ::mlir::ArrayAttr::get($_ctxt, attrs),
                   ::mlir::BoolAttr::get($_ctxt, emitAsComments));
    }]>,
    AttrBuilder<(ins
      "llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>>":$keyValuePairs,
       CArg<"bool", "false">: $emitAsComments)>];
  let hasCustomAssemblyFormat = true;
}
