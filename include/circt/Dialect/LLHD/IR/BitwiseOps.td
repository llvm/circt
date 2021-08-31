//===- BitwiseOps.td - LLHD bitwise operations -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the bitwise MLIR ops for LLHD.
//
//===----------------------------------------------------------------------===//

def LLHD_ShlOp : LLHD_Op<"shl", [
    NoSideEffect,
    AllTypesMatch<["base", "result"]>,
    SameTypeArbitraryWidth<
      "'hidden' and 'base' need to have the same type apart from the width",
      "$hidden", "$base">
  ]> {
  let summary = "Shifts a value to the left by a given amount.";
  let description = [{
    The type of the base value and the hidden value must be the same, but
    may differ in the number of bits or elements. The result always has the
    same type (including width) of the base value.
    The instruction is transparent to signals and pointers. For example,
    passing a signal as argument will shift the underlying value and return
    a signal to the shifted value.
    Allowed (underlying) types are signless integers, nine-valued-logic values
    and arrays. The shift amount has to be a signless integer. A shift amount
    bigger than the number of bits or elements of the hidden value is undefined.
    The hidden value is uncovered by non-zero shift amounts. E.g. consider
    the four bit values `base = 0xf`, `hidden = 0xc` shifted by an amount of
    three result in `0xe`.

    Syntax:

    ```
    shl-op ::= ssa-id `=`
      `llhd.shl` ssa-base `,` ssa-hidden `,` ssa-amount attr-dict `:`
      `(` base-type `,` hidden-type `,` amount-type `)` `->` result-type
    ```

    Examples:

    ```mlir
    %0 = llhd.shl %base, %hidden, %amount : (i4, i2, i2) -> i4
    ```
  }];

  // TODO: adjust type T and Th to include pointers
  let arguments = (ins
    AnyTypeOf<[HWIntegerType, ArrayType,
               LLHD_SigType<[HWIntegerType, ArrayType]>]>:$base,
    AnyTypeOf<[HWIntegerType, ArrayType,
               LLHD_SigType<[HWIntegerType, ArrayType]>]>:$hidden,
    HWIntegerType:$amount);

  let results = (outs
    AnyTypeOf<[HWIntegerType, ArrayType,
               LLHD_SigType<[HWIntegerType, ArrayType]>]>:$result);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getBaseWidth()  { return getLLHDTypeWidth(base().getType()); }
    unsigned getHiddenWidth() { return getLLHDTypeWidth(hidden().getType()); }
  }];

  let hasFolder = 1;
}

def LLHD_ShrOp : LLHD_Op<"shr", [
    NoSideEffect,
    AllTypesMatch<["base", "result"]>,
    SameTypeArbitraryWidth<
      "'hidden' and 'base' need to have the same type apart from the width",
      "$hidden", "$base">
  ]> {
  let summary = "Shifts a value to the right by a given amount.";
  let description = [{
    The type of the base value and the hidden value must be the same, but
    may differ in the number of bits or elements. The result always has the
    same type (including width) of the base value.
    The instruction is transparent to signals and pointers. For example,
    passing a signal as argument will shift the underlying value and return
    a signal to the shifted value.
    Allowed (underlying) types are signless integers, nine-valued-logic values
    and arrays. The shift amount has to be a signless integer. A shift amount
    bigger than the number of bits or elements of the hidden value is undefined.
    The hidden value is uncovered by non-zero shift amounts. E.g. consider
    the four bit values `base = 0xf`, `hidden = 0xc` shifted by an amount of
    three result in `0x9`.

    Syntax:

    ```
    shr-op ::= ssa-id `=`
      `llhd.shr` ssa-base `,` ssa-hidden `,` ssa-amount attr-dict `:`
      `(` base-type `,` hidden-type `,` amount-type `)` `->` result-type
    ```

    Examples:

    ```mlir
    %0 = llhd.shr %base, %hidden, %amount : (i4, i2, i2) -> i4
    ```
  }];

  // TODO: adjust type T and Th to include pointers
  let arguments = (ins
    AnyTypeOf<[HWIntegerType, ArrayType,
               LLHD_SigType<[HWIntegerType, ArrayType]>]>:$base,
    AnyTypeOf<[HWIntegerType, ArrayType,
               LLHD_SigType<[HWIntegerType, ArrayType]>]>:$hidden,
    HWIntegerType:$amount);

  let results = (outs
    AnyTypeOf<[HWIntegerType, ArrayType,
               LLHD_SigType<[HWIntegerType, ArrayType]>]>:$result);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getBaseWidth()  { return getLLHDTypeWidth(base().getType()); }
    unsigned getHiddenWidth() { return getLLHDTypeWidth(hidden().getType()); }
  }];

  let hasFolder = 1;
}
