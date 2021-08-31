//===- ArithmeticOps.td - LLHD arithmetic operations -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the additional arithmetic MLIR ops for LLHD.
//
//===----------------------------------------------------------------------===//

def LLHD_EqOp : LLHD_Op<"eq", [
    NoSideEffect,
    Commutative,
    AllTypesMatch<["lhs", "rhs"]>
  ]> {
  let summary = "Logical Equality";
  let description = [{
    This operation compares two values and returns 1 if they are the same and 0
    otherwise. It is capable of comparing all types in the LLHD dialect. It
    performs element-wise equality for arrays and tuples.

    Examples:

    ```mlir
    %const1 = constant 1 : i32
    %const2 = constant 2 : i32
    %eq = llhd.eq %const1, %const2 : i32  // %eq = 0

    %const3 = constant 0 : i1
    %tup1 = llhd.tuple %const1, %const3 : tuple<i32, i1>
    %tup2 = llhd.tuple %const2, %const3 : tuple<i32, i1>
    %tupeq = llhd.eq %tup1, %tup2 : i32  // %tupeq = 0
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "operands attr-dict `:` type($lhs)";

  let hasFolder = 1;
}

def LLHD_NeqOp : LLHD_Op<"neq", [
    NoSideEffect,
    Commutative,
    AllTypesMatch<["lhs", "rhs"]>
  ]> {
  let summary = "Logical Inequality";
  let description = [{
    This operation compares two values and returns 1 if they are not the same
    and 0 if they are the same. It is capable of comparing all types in the LLHD
    dialect. It performs element-wise equality for arrays and tuples.

    Examples:

    ```mlir
    %const1 = constant 1 : i32
    %const2 = constant 2 : i32
    %neq = llhd.neq %const1, %const2 : i32  // %neq = 1

    %const3 = constant 0 : i1
    %tup1 = llhd.tuple %const1, %const3 : tuple<i32, i1>
    %tup2 = llhd.tuple %const2, %const3 : tuple<i32, i1>
    %tupneq = llhd.neq %tup1, %tup2 : i32  // %tupeq = 1
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "operands attr-dict `:` type($lhs)";

  let hasFolder = 1;
}
