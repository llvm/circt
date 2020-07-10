//===- BitwiseOps.td - LLHD bitwise operations -------------*- tablegen -*-===//
//
// This describes the bitwise MLIR ops for LLHD.
//
//===----------------------------------------------------------------------===//

def LLHD_NotOp : LLHD_ArithmeticOrBitwiseOp<"not", []> {
  let summary = "Bitwise NOT";
  let description = [{
    Takes an integer of any width or a nine-valued-logic (IEEE 1164) value
    of any width as input. Flips each bit of a value. The result always has
    the exact same type.

    Syntax:

    ```
    not-op ::= ssa-id `=` `llhd.not` ssa-value attr-dict `:` type
    ```

    Examples:

    ```mlir
    %0 = llhd.const 0 : i32
    %1 = llhd.not %0 : i32
    ```

    Truth Table for integers:

    | `not` |  0  |  1  |
    |:-----:|:---:|:---:|
    |       |  1  |  0  |

    Truth Table for nine-valued logic:

    | `not` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
    |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |       |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
  }];

  let arguments = (ins AnySignlessInteger:$value);

  let hasFolder = 1;
}

def LLHD_AndOp : LLHD_ArithmeticOrBitwiseOp<"and", [Commutative]> {
  let summary = "Bitwise AND";
  let description = [{
    Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
    values of the same width as input. Calculates the bitwise AND. The
    result is always of the exact same type as the two inputs.

    Syntax:

    ```
    and-op ::= ssa-id `=` `llhd.and` ssa-lhs `,` ssa-rhs attr-dict `:` type
    ```

    Examples:

    ```mlir
    %0 = llhd.const 0 : i32
    %1 = llhd.and %0, %0 : i32
    ```

    Truth Table for integers:

    | `and` |  0  |  1  |
    |:-----:|:---:|:---:|
    |   0   |  0  |  0  |
    |   1   |  0  |  1  |

    Truth Table for nine-valued logic:

    | `and` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
    |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |   U   |  U  |  U  |  0  |  U  |  U  |  U  |  0  |  U  |  U  |
    |   X   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
    |   0   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
    |   1   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
    |   Z   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
    |   W   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
    |   L   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
    |   H   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
    |   -   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
  }];

  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);

  let hasFolder = 1;
}

def LLHD_OrOp : LLHD_ArithmeticOrBitwiseOp<"or", [Commutative]> {
  let summary = "Bitwise OR";
  let description = [{
    Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
    values of the same width as input. Calculates the bitwise OR. The
    result is always of the exact same type as the two inputs.

    Syntax:

    ```
    or-op ::= ssa-id `=` `llhd.or` ssa-lhs `,` ssa-rhs attr-dict `:` type
    ```

    Examples:

    ```mlir
    %0 = llhd.const 0 : i32
    %1 = llhd.or %0, %0 : i32
    ```

    Truth Table for integers:

    | `or`  |  0  |  1  |
    |:-----:|:---:|:---:|
    |   0   |  0  |  1  |
    |   1   |  1  |  1  |

    Truth Table for nine-valued logic:

    | `or`  |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
    |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |   U   |  U  |  U  |  U  |  1  |  U  |  U  |  U  |  1  |  U  |
    |   X   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
    |   0   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
    |   1   |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
    |   Z   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
    |   W   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
    |   L   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
    |   H   |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
    |   -   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
  }];

  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);

  let hasFolder = 1;
}

def LLHD_XorOp : LLHD_ArithmeticOrBitwiseOp<"xor", [Commutative]> {
  let summary = "Bitwise XOR";
  let description = [{
    Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
    values of the same width as input. Calculates the bitwise XOR. The
    result is always of the exact same type as the two inputs.

    Syntax:

    ```
    xor-op ::= ssa-id `=` `llhd.xor` ssa-lhs `,` ssa-rhs attr-dict `:` type
    ```

    Examples:

    ```mlir
    %0 = llhd.const 0 : i32
    %1 = llhd.xor %0, %0 : i32
    ```

    Truth Table for integers:

    | `xor` |  0  |  1  |
    |:-----:|:---:|:---:|
    |   0   |  0  |  1  |
    |   1   |  1  |  0  |

    Truth Table for nine-valued logic:

    | `xor` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
    |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |   U   |  U  |  U  |  U  |  U  |  U  |  U  |  U  |  U  |  U  |
    |   X   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
    |   0   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
    |   1   |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
    |   Z   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
    |   W   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
    |   L   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
    |   H   |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
    |   -   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
  }];

  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);

  let hasFolder = 1;
}

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
    AnyTypeOf<[AnySignlessInteger, AnyVector,
               LLHD_SigType<[AnySignlessInteger, AnyVector]>]>:$base,
    AnyTypeOf<[AnySignlessInteger, AnyVector,
               LLHD_SigType<[AnySignlessInteger, AnyVector]>]>:$hidden,
    AnySignlessInteger:$amount);

  let results = (outs
    AnyTypeOf<[AnySignlessInteger, AnyVector,
               LLHD_SigType<[AnySignlessInteger, AnyVector]>]>:$result);

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
    AnyTypeOf<[AnySignlessInteger, AnyVector,
               LLHD_SigType<[AnySignlessInteger, AnyVector]>]>:$base,
    AnyTypeOf<[AnySignlessInteger, AnyVector,
               LLHD_SigType<[AnySignlessInteger, AnyVector]>]>:$hidden,
    AnySignlessInteger:$amount);

  let results = (outs
    AnyTypeOf<[AnySignlessInteger, AnyVector,
               LLHD_SigType<[AnySignlessInteger, AnyVector]>]>:$result);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getBaseWidth()  { return getLLHDTypeWidth(base().getType()); }
    unsigned getHiddenWidth() { return getLLHDTypeWidth(hidden().getType()); }
  }];

  let hasFolder = 1;
}
