//===- SVAttributes.td - Attributes for HW dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SystemVerilog dialect-specific attributes.
//
//===----------------------------------------------------------------------===//

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

    For more information, refer to Section 5.12 of the SystemVerilog (1800-2017)
    specification.
  }];
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::StringAttr":$expression);
}
