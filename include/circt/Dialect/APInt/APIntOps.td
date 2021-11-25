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

// Base class for binary operators.
class BinOp<string mnemonic, list<OpTrait> traits = []> :
      APIntOp<mnemonic, traits # [NoSideEffect]> {
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

// TODO the type cannot be inferred if operand types are not the same as the result type
//      Yet, for lossless arithmetic the bit sizes are typically different.
//      See i.e. https://github.com/Minres/CoreDSL/wiki/Expressions#addition

// Binary operator with uniform input/result types.
class UTBinOp<string mnemonic, list<OpTrait> traits = []> :
      BinOp<mnemonic,
               traits # [SameTypeOperands, SameOperandsAndResultType]> {
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


// TODO unary ops are probably desired too

// Arithmetic and Logical Operations.
def AddOp : TIBinOp<"add", [Commutative]>;
def MulOp : TIBinOp<"mul", [Commutative]>;
