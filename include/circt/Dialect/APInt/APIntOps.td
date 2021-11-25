//===- APIntOps.td - arbitrary precision integer ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for lossless arbitrary precision integer
// arithmetic.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Arithmetic and Logical Operations
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for binary operators.
class BinOp<string mnemonic, list<OpTrait> traits = []> :
      APIntOp<mnemonic, traits # [NoSideEffect]> {
  let arguments = (ins APIntType:$lhs, APIntType:$rhs);
  let results = (outs APIntType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

// Binary operator with result type inference.
class TIBinOp<string mnemonic, list<OpTrait> traits = []> :
      BinOp<mnemonic, traits # [InferTypeOpInterface]> {

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          Optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

// Arithmetic and Logical Operations.
def AddOp : TIBinOp<"add", [Commutative]>;
def MulOp : TIBinOp<"mul", [Commutative]>;

def ConstantOp : APIntOp<"constant",
    [ConstantLike, NoSideEffect, FirstAttrDerivedResultType]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a sign aware constant value.
    ```
      %result = apint.constant 42 : t1
    ```
    }];

  let arguments = (ins APIntAttr:$value);
  let results = (outs APIntType:$result);

  let parser = "return parse$cppClass(parser, result);";
  let printer = "print$cppClass(p, *this);";

  let extraClassDeclaration = [{
    APSInt getValue();
  }];
}
