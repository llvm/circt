//===- ArithmeticOps.td - LLHD arithmetic operations -------*- tablegen -*-===//
//
// This describes the additional arithmetic MLIR ops for LLHD.
//
//===----------------------------------------------------------------------===//

def LLHD_NegOp : LLHD_ArithmeticOrBitwiseOp<"neg", []> {
  let summary = "Negate a value.";
  let description = [{
    The operand and result always have the same type. The type has to be a
    signless integer of any width. Although, only signless integers are
    allowed, this instruction applies two's complement negation of the
    integer, basically treating it as a signed integer.

    Syntax:

    ```
    neg-op ::= ssa-id `=` `llhd.neg` ssa-value attr-dict `:` type
    ```

    Example:

    ```mlir
    %0 = llhd.const 42 : i32
    %1 = llhd.neg %0 : i32
    ```
  }];

  let arguments = (ins AnySignlessInteger:$value);

  let hasFolder = 1;
}

def LLHD_SModOp : LLHD_ArithmeticOrBitwiseOp<"smod", []> {
  let summary = "Signed modulo.";
  let description = [{
    This instruction computes the signed modulo of two signless integers of
    any width, treating the leading bit as sign. The operand and result
    types always have to be the same.
    To calculate the signed remainder of two integers, use `remi_signed`
    from the standard dialect.

    Syntax:

    ```
    smod-op ::= ssa-id `=` `llhd.smod` ssa-lhs `,` ssa-rhs attr-dict `:` type
    ```

    Examples:

    ```mlir
    %0 = llhd.const 9 : i4
    %1 = llhd.const 4 : i4
    %2 = llhd.smod %0, %1 : i4
    ```
  }];

  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);

  let hasFolder = 1;
}

def LLHD_EqOp : LLHD_Op<"eq", [
    NoSideEffect,
    Commutative,
    AllTypesMatch<["lhs", "rhs"]>
  ]> {
  let summary = "Logical Equality";
  let description = [{
    This operation compares two values and returns 1 if they are the same and 0
    otherwise. It is capable of comparing all types in the LLHD dialect. It
    performs element-wise equality for vectors and tuples.

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
    dialect. It performs element-wise equality for vectors and tuples.

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
