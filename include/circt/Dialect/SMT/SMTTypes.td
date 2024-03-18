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

def IntType : SMTTypeDef<"Int"> {
  let mnemonic = "int";
  let description = [{
    This type represents the `Int` sort as described in the
    [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
    SMT-LIB 2.6 standard.
  }];
  let assemblyFormat = "";
}

def BitVectorType : SMTTypeDef<"BitVector"> {
  let mnemonic = "bv";
  let description = [{
    This type represents the `(_ BitVec width)` sort as described in the
    [SMT bit-vector
    theory](https://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml).

    The bit-width must be strictly greater than zero.
  }];

  let parameters = (ins "uint64_t":$width);
  let assemblyFormat = "`<` $width `>`";

  let genVerifyDecl = true;
}

def ArrayType : SMTTypeDef<"Array"> {
  let mnemonic = "array";
  let description = [{
    This type represents the `(Array X Y)` sort, where X and Y are any
    sort/type, as described in the
    [SMT ArrayEx theory](https://smtlib.cs.uiowa.edu/Theories/ArraysEx.smt2) of
    the SMT-LIB standard 2.6.
  }];

  let parameters = (ins "mlir::Type":$domainType, "mlir::Type":$rangeType);
  let assemblyFormat = "`<` `[` $domainType `->` $rangeType `]` `>`";

  let genVerifyDecl = true;
}

def AnySMTType : Type<CPred<"smt::isAnySMTValueType($_self)">,
                      "any SMT value type">;
def AnyNonSMTType : Type<Neg<AnySMTType.predicate>, "any non-smt type">;

#endif // CIRCT_DIALECT_SMT_SMTTYPES_TD
