//===- ValueOps.td - LLHD value operations -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for LLHD value creation.
//
//===----------------------------------------------------------------------===//

def LLHD_ConstOp : LLHD_Op<"const", [ConstantLike, NoSideEffect]> {
  let summary = "Introduce a new constant.";
  let description = [{
    The `llhd.const` instruction introduces a new constant value as an
    SSA-operator.
    Legal types are integers and time. Note: Signals
    are not legal to define using `llhd.const`, use the `llhd.sig`
    instruction for that.

    Syntax:

    ```
    const-op ::= ssa-id `=`
      `llhd.const` attribute-value attr-dict `:` result-type
    ```

    Example:

    ```mlir
    %0 = llhd.const 1 : i64
    %1 = llhd.const #llhd.time<1ns, 2d, 3d> : !llhd.time
    ```
  }];

  let arguments = (ins AnyAttr: $value);
  let results = (outs AnyTypeOf<[AnySignlessInteger, LLHD_TimeType]>: $out);

  let hasFolder = 1;
}

def LLHD_ArrayOp : LLHD_Op<"array", [
    NoSideEffect,
    TypesMatchWith<
      "types and number of 'values' have to match the length and type of the "
      "'result' array",
      "result", "values",
      "std::vector<Type>($_self.cast<llhd::ArrayType>().getLength(), "
      "$_self.cast<llhd::ArrayType>().getElementType())">
  ]> {
  let summary = "Create an array from a list of values.";
  let description = [{
    The `llhd.array` operation allows to create an array from a list of
    SSA-values.

    Example:

    ```mlir
    %c1 = llhd.const 1 : i32
    %c2 = llhd.const 2 : i32
    %c3 = llhd.const 3 : i32
    %array = llhd.array %c1, %c2, %c3 : !llhd.array<3xi32>
    %elem = llhd.extract_element %array, 0 : !llhd.array<3xi32> -> i32
    ```
  }];

  let arguments = (ins Variadic<AnyType>: $values);
  let results = (outs LLHD_ArrayType: $result);

  let assemblyFormat = "$values attr-dict `:` type($result)";
}

def LLHD_ArrayUniformOp : LLHD_Op<"array_uniform", [
    NoSideEffect,
    TypesMatchWith<
      "type of 'init' has to match the element type of the 'result' array",
      "result", "init",
      "$_self.cast<llhd::ArrayType>().getElementType()">
  ]> {
  let summary = "Create a uniform array from an initialization value.";
  let description = [{
    The `llhd.array_uniform` operation allows to create an LLHD array from a
    single SSA-value used as initialization for all elements.

    Example:

    ```mlir
    %init = llhd.const 1 : i32
    %array = llhd.array_uniform %init : !llhd.array<3xi32>
    ```
  }];

  let arguments = (ins AnyType: $init);
  let results = (outs LLHD_ArrayType: $result);

  let assemblyFormat = "$init attr-dict `:` type($result)";
}

def LLHD_TupleOp : LLHD_Op<"tuple", [
    NoSideEffect,
    TypesMatchWith<
      "types of 'values' have to match the type of the corresponding 'result' "
      "tuple element",
      "result", "values",
      "$_self.cast<TupleType>().getTypes()">
  ]> {
  let summary = "Create a tuple from a list of values.";
  let description = [{
    The `llhd.tuple` operation creates a tuple from a list of SSA-values.

    Example:

    ```mlir
    %c1 = llhd.const 1 : i32
    %c2 = llhd.const 2 : i2
    %sig = llhd.sig "sig_name" %c1 : i32
    %array = llhd.array_uniform %c1 : !llhd.array<2xi32>
    %tuple = llhd.tuple %c1, %c2, %array, %sig :
      tuple<i32, i2, !llhd.array<2xi32>, !llhd.sig<i32>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>: $values);
  let results = (outs AnyTuple: $result);

  let assemblyFormat = "$values attr-dict `:` type($result)";
}
