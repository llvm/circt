//===- SMTAttributes.td - Attributes for SMT dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SMT dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTATTRIBUTES_TD
#define CIRCT_DIALECT_SMT_SMTATTRIBUTES_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def BitVectorAttr : AttrDef<SMTDialect, "BitVector", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "bv";
  let description = [{
    This attribute represents a constant value of the `(_ BitVec width)` sort as
    described in the [SMT bit-vector
    theory](https://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml).

    The constant is parsed and printed as #bX (binary) or #xX (hexadecimal)
    where X is the value in the corresponding format without any further
    prefixing just as in SMT-LIB. The number of digits determines the bit-width
    of the bit-vector. This means, leading zeros are important!

    Examples:
    ```mlir
    #smt.bv<"#b0101"> : !smt.bv<4>
    #smt.bv<"#x5c"> : !smt.bv<8>
    ```
    The explicit type-suffix is optional.

    The bit-width must be greater than zero (i.e., at least one digit as to be
    present).
  }];

  let parameters = (ins "llvm::APInt":$value);

  let hasCustomAssemblyFormat = true;
  let genVerifyDecl = true;

  let builders = [
    AttrBuilder<(ins "llvm::StringRef":$value)>,
    AttrBuilder<(ins "unsigned":$value, "unsigned":$width)>,
  ];

  let extraClassDeclaration = [{
    /// Return the bit-vector constant as a SMT-LIB formatted string.
    std::string getValueAsString(bool prefix = true) const;
  }];
}

#endif // CIRCT_DIALECT_SMT_SMTATTRIBUTES_TD
