//===- MooreOps.td - Moore dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREOPS
#define CIRCT_DIALECT_MOORE_MOOREOPS

include "circt/Dialect/Moore/MooreDialect.td"
include "circt/Dialect/Moore/MooreTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class MooreOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Structure
//===----------------------------------------------------------------------===//

def SVModuleOp : MooreOp<"module", [
  IsolatedFromAbove,
  Symbol,
  NoTerminator,
  SingleBlock
]> {
  let summary = "A module definition";
  let description = [{
    The `moore.module` operation represents a SystemVerilog module, including
    its name, port list, and the constituent parts that make up its body. The
    module's body is an SSACFG region, since declarations within SystemVerilog
    modules generally have to appear before their uses, and dedicated assignment
    operators are used to make connections after declarations.

    See IEEE 1800-2017 § 3.3 "Modules" and § 23.2 "Module definitions".
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $bodyRegion
  }];
}

def InstanceOp : MooreOp<"instance", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Create an instance of a module";
  let description = [{
    The `moore.instance` operation instantiates a `moore.module` operation.

    See IEEE 1800-2017 § 23.3 "Module instances".
  }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName);
  let assemblyFormat = [{
    $instanceName $moduleName attr-dict
  }];
}

def Initial: I32EnumAttrCase<"Initial", 0, "initial">;
def Final: I32EnumAttrCase<"Final", 1, "final">;
def Always: I32EnumAttrCase<"Always", 2, "always">;
def AlwaysComb: I32EnumAttrCase<"AlwaysComb", 3, "always_comb">;
def AlwaysLatch: I32EnumAttrCase<"AlwaysLatch", 4, "always_latch">;
def AlwaysFF: I32EnumAttrCase<"AlwaysFF", 5, "always_ff">;

def ProcedureKindAttr: I32EnumAttr<"ProcedureKind", "Procedure kind",
            [Initial, Final, Always, AlwaysComb, AlwaysLatch, AlwaysFF]>{
  let cppNamespace = "circt::moore";
}

def ProcedureOp : MooreOp<"procedure", [
  SingleBlock,
  NoTerminator,
  NoRegionArguments,
  RecursiveMemoryEffects,
  RecursivelySpeculatable
]> {
  let summary = "A procedure executed at different points in time";
  let description = [{
    The `moore.procedure` operation represents the SystemVerilog `initial`,
    `final`, `always`, `always_comb`, `always_latch`, and `always_ff`
    procedures.

    Execution times of the various procedures:

    - An `initial` procedure is executed once at the start of a design's
      lifetime, before any other procedures are executed.

    - A `final` procedure is executed once at the end of a design's lifetime,
      after all other procedures have stopped execution.

    - An `always` or `always_ff` procedure is repeatedly executed during a
      design's lifetime. Timing and event control inside the procedure can
      suspend its execution, for example to wait for a signal to change. If no
      such timing or event control is present, the procedure repeats infinitely
      at the current timestep, effectively deadlocking the design.

    - An `always_comb` or `always_latch` procedure is executed once at the start
      of a design's lifetime, after any `initial` procedures, and throughout the
      lifetime of the design whenever any of the variables read by the body of
      the procedure changes. Since the procedure is only executed when its
      change, and not repeatedly, the body generally does not contain any timing
      or event control. This behavior mitigates a shortcoming of `always`
      procedures, which commonly have an event control like `@*` that blocks
      and waits for a change of any input signals. This prevents the body from
      executing when the design is initialized and properly reacting to the
      initial values of signals. In contrast, `always_comb` and `always_latch`
      procedures have an implicit unconditional execution at design start-up.

    See IEEE 1800-2017 § 9.2 "Structured procedures".
  }];

  let regions = (region SizedRegion<1>:$bodyRegion);
  let arguments = (ins ProcedureKindAttr:$kind);
  let results = (outs);

  let assemblyFormat = [{
    $kind attr-dict-with-keyword $bodyRegion
  }];
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

def In: I32EnumAttrCase<"In", 0>;
def Out: I32EnumAttrCase<"Out", 1>;
def InOut: I32EnumAttrCase<"InOut", 2>;
def Ref: I32EnumAttrCase<"Ref", 3>;

def DirectionAttr: I32EnumAttr<"Direction", "Port direction",
                                  [In, Out, InOut, Ref]>{
  let cppNamespace = "circt::moore";
}

def PortOp : MooreOp<"port", [
  RecursiveMemoryEffects
]> {
  let summary = "Declare a port";
  let arguments = (ins StrAttr:$name, DirectionAttr:$direction);
  let results = (outs);
  let assemblyFormat = [{
    $direction $name attr-dict
  }];
}

def VariableOp : MooreOp<"variable", [
  MemoryEffects<[MemAlloc]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"initial value and variable types match",
    "result", "initial", "$_self">,
]> {
  let summary = "A variable declaration";
  let description = [{
    See IEEE 1800-2017 § 6.8 "Variable declarations".
  }];
  let arguments = (ins StrAttr:$name, Optional<UnpackedType>:$initial);
  let results = (outs Res<UnpackedType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) ($initial^)? attr-dict
    `:` type($result)
  }];
}

def NetOp : MooreOp<"net", [
  MemoryEffects<[MemAlloc]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"assigned value and variable types match",
    "result", "assignment", "$_self">,
]> {
  let summary = "A net declaration";
  let description = [{
    See IEEE 1800-2017 § 6.7 "Net declarations".
  }];
  let arguments = (ins
    StrAttr:$name,
    StrAttr:$kind,
    Optional<UnpackedType>:$assignment
  );
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $kind ($assignment^)? attr-dict
    `:` type($result)
  }];
}

def ReadLValueOp : MooreOp<"read_lvalue", [SameOperandsAndResultType]> {
  let summary = "Read the current value of a declaration";
  let description = [{
    Samples the current value of a declaration. This is a helper to capture the
    exact point at which declarations that can be targeted by assignments are
    read.
  }];
  let arguments = (ins Arg<UnpackedType, "", [MemRead]>:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Assignments
//===----------------------------------------------------------------------===//

class AssignOpBase<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits # [SameTypeOperands]> {
  let arguments = (ins AnyType:$dst, AnyType:$src);
  let assemblyFormat = [{
    $dst `,` $src attr-dict `:` type($src)
  }];
}

def ContinuousAssignOp : AssignOpBase<"assign", [HasParent<"SVModuleOp">]> {
  let summary = "Continuous assignment within a module";
  let description = [{
    A continuous assignment in module scope, such as `assign x = y;`, which
    continuously drives the value on the right-hand side onto the left-hand
    side.

    See IEEE 1800-2017 § 10.3 "Continuous assignments".
  }];
}

def BlockingAssignOp : AssignOpBase<"blocking_assign", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "Blocking procedural assignment";
  let description = [{
    A blocking procedural assignment in a sequential block, such as `x = y`. The
    effects of the assignment are visible to any subsequent operations in the
    block.

    See IEEE 1800-2017 § 10.4.1 "Blocking procedural assignments".
  }];
  let arguments = (ins Arg<AnyType, "", [MemWrite]>:$dst, AnyType:$src);
}

def NonBlockingAssignOp : AssignOpBase<"nonblocking_assign"> {
  let summary = "Nonblocking procedural assignment";
  let description = [{
    A nonblocking procedural assignment in a sequential block, such as `x <= y;`
    or `x <= @(posedge y) z` or `x <= #1ns y`. The assignment does not take
    effect immediately. Subsequent operations in the block do not see the
    effects of this assignment. Instead, the assignment is scheduled to happen
    in a subsequent time step as dictated by the delay or event control.

    See IEEE 1800-2017 § 10.4.2 "Nonblocking procedural assignments".
  }];
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

def None: I32EnumAttrCase<"None", 0, "none">;
def PosEdge: I32EnumAttrCase<"PosEdge", 1, "posedge">;
def NegEdge: I32EnumAttrCase<"NegEdge", 2, "negedge">;
def BothEdges: I32EnumAttrCase<"BothEdges", 3, "bothedges">;

def EdgeAtrr: I32EnumAttr<"Edge", "Edge kind",
                          [None, PosEdge, NegEdge, BothEdges]>{
  let cppNamespace = "circt::moore";
}

def EventControlOp : MooreOp<"event", [
  HasParent<"ProcedureOp">
]> {
  let summary = "Detecting posedge and negedge";
  let description = "See SV Spec 9.4.2.";
  let arguments = (ins EdgeAtrr:$edge, StrAttr:$name);
  let results = (outs);
  let assemblyFormat = [{
    $edge $name attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

def ConstantOp : MooreOp<"constant", [Pure]> {
  let summary = "A constant integer value";
  let arguments = (ins APIntAttr:$value);
  let results = (outs SimpleBitVectorType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "Type":$type, "const APInt &":$value)>,
    OpBuilder<(ins "Type":$type, "int64_t":$value)>,
  ];
}

def ConversionOp : MooreOp<"conversion", [Pure]> {
  let summary = "A type conversion";
  let description = [{
    An explicit or implicit type conversion. These are either generated
    automatically in order to make assignments compatible:

    ```
    int a;
    shortint b;
    a = b;  // generates an implicit cast from shortint to int
    ```

    Or explicitly by the user through a type, sign, or const cast expression:

    ```
    byte'(a)
    unsigned'(a)
    signed'(a)
    42'(a)
    ```

    See IEEE 1800-2017 § 6.24 "Casting".
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasFolder = 1;
}

def NegOp : MooreOp<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Arithmetic negation";
  let description = [{
    Negate a value to its two's complement form. If any bit in the input is Z or
    X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def NotOp : MooreOp<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "Bitwise unary negation";
  let description = [{
    Applies the boolean NOT operation to each bit in the input. Corresponds to
    the `~` operator, as well as the negation in the `~&`, `~|`, `^~`, and `~^`
    reduction operators.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    | Input | Result |
    |-------|--------|
    | 0     | 1      |
    | 1     | 0      |
    | X     | X      |
    | Z     | X      |
  }];
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

class ReduceOpBase<string mnemonic, string operatorName> : MooreOp<mnemonic, [
  Pure,
  TypesMatchWith<"result is single bit of input", "input", "result", [{
    $_self.cast<UnpackedType>()
      .getSimpleBitVector()
      .toSingleBit()
      .getType($_self.getContext())
  }]>
]> {
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let summary = !subst("$op", operatorName, "Reduction $op operator");
  let description = !subst("$op", operatorName, [{
    Reduces all bits in the input to a single result bit by iteratively applying
    the boolean $op operator. If the input has only a single bit, that bit is
    returned.

    See IEEE 1800-2017 § 11.4.9 "Reduction operators". See the corresponding
    `and`, `or`, and `xor` operations for the truth table.
  }]);
}

def ReduceAndOp : ReduceOpBase<"reduce_and", "AND">;
def ReduceOrOp : ReduceOpBase<"reduce_or", "OR">;
def ReduceXorOp : ReduceOpBase<"reduce_xor", "XOR">;

def BoolCastOp : MooreOp<"bool_cast", [
  Pure,
  TypesMatchWith<"result is single bit matching input domain",
    "input", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let summary = "Cast a value to a single bit boolean";
  let description = [{
    Convert a nonzero or true value into 1, a zero or false value into 0, and
    any value containing Z or X bits into a X. This conversion is useful in
    combination with the logical and, or, implication, equivalence, and negation
    operators.

    See IEEE 1800-2017 § 11.4.7 "Logical operators".
  }];
  let arguments = (ins UnpackedType:$input);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasFolder = 1;
}

class BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

def AddOp : BinaryOpBase<"add", [Commutative]> {
  let summary = "Addition";
  let description = [{
    Add the operands. If any bit in the two operands is Z or X, all bits in the
    result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def SubOp : BinaryOpBase<"sub"> {
  let summary = "Subtraction";
  let description = [{
    Subtract the right-hand side from the left-hand side operand. If any bit in
    the two operands is Z or X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def MulOp : BinaryOpBase<"mul", [Commutative]> {
  let summary = "Multiplication";
  let description = [{
    Multiply the operands. If any bit in the two operands is Z or X, all bits in
    the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def DivOp : BinaryOpBase<"div"> {
  let summary = "Division";
  let description = [{
    Divide the left-hand side by the right-hand side operand. Any fractional
    part is truncated toward zero. If the right-hand side is zero, all bits of
    the result are X. If any bit in the two operands is Z or X, all bits in the
    result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def ModOp : BinaryOpBase<"mod"> {
  let summary = "Remainder";
  let description = [{
    Compute the remainder of the left-hand side divided by the right-hand side
    operand. If the right-hand side is zero, all bits of the result are X. The
    sign of the result is the sign of the left-hand side. If any bit in the two
    operands is Z or X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Consider the following examples:

    | LHS | RHS | Result |
    |-----|-----|--------|
    |  11 |   3 |      2 |
    | -11 |   3 |     -2 |
    |  11 |  -3 |      2 |
    | -11 |  -3 |     -2 |
  }];
}

def AndOp : BinaryOpBase<"and", [Commutative]> {
  let summary = "Bitwise AND operation";
  let description = [{
    Applies the boolean AND operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `&` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 0 | 0 | 0 |
    | 1 | 0 | 1 | X | X |
    | X | 0 | X | X | X |
    | Z | 0 | X | X | X |
  }];
}

def OrOp : BinaryOpBase<"or", [Commutative]> {
  let summary = "Bitwise OR operation";
  let description = [{
    Applies the boolean OR operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `|` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 1 | X | X |
    | 1 | 1 | 1 | 1 | 1 |
    | X | X | 1 | X | X |
    | Z | X | 1 | X | X |
  }];
}

def XorOp : BinaryOpBase<"xor", [Commutative]> {
  let summary = "Bitwise XOR operation";
  let description = [{
    Applies the boolean XOR operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `^` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 1 | X | X |
    | 1 | 1 | 0 | X | X |
    | X | X | X | X | X |
    | Z | X | X | X | X |
  }];
}

class ShiftOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  TypesMatchWith<
    "value and result types must match", "value", "result", "$_self">
]> {
  let description = [{
    Shifts the `value` to the left or right by `amount` number of bits. The
    result has the same type as the input value. The amount is always treated as
    an unsigned number and has no effect on the signedness of the result. X or
    Z bits in the input value are simply shifted left or right the same way 0 or
    1 bits are. If the amount contains X or Z bits, all result bits are X.

    `shl` shifts bits to the left, filling in 0 for the vacated least
    significant bits. `shr` and `ashr` shift bits to the right; `shr` fills in
    0 for the vacated most significant bits, and `ashr` copies the input's sign
    bit into the vacated most significant bits. Note that in contrast to the SV
    spec, the `ashr` _always_ fills in the sign bit regardless of the signedness
    of the input.

    `shl` corresponds to the `<<` and `<<<` operators. `shr` corresponds to the
    `>>` operator, and the `>>>` operator applied to an unsigned value. `ashr`
    corresponds to the `>>>` operator applied to a signed value.

    See IEEE 1800-2017 § 11.4.10 "Shift operators".
  }];
  let arguments = (ins SimpleBitVectorType:$value, SimpleBitVectorType:$amount);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $value `,` $amount attr-dict `:` type($value) `,` type($amount)
  }];
}

def ShlOp : ShiftOpBase<"shl"> { let summary = "Logical left shift"; }
def ShrOp : ShiftOpBase<"shr"> { let summary = "Logical right shift"; }
def AShrOp : ShiftOpBase<"ashr"> { let summary = "Arithmetic right shift"; }

class LogicalEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  Commutative,
  SameTypeOperands,
  TypesMatchWith<"result is single bit matching input domain",
    "lhs", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0, 1, or X result. If all corresponding bits in the left- and
    right-hand side are equal, and all are 0 or 1 (not X or Z), the two operands
    are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
    equal, but all are 0 or 1, the two operands are considered not equal (`eq`
    returns 0, `ne` returns 1). If any bit in the two operands is Z or X,
    returns X. `eq` corresponds to the `==` operator and `ne` to the `!=`
    operator.

    See IEEE 1800-2017 § 11.4.5 "Equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def EqOp : LogicalEqOpBase<"eq"> { let summary = "Logical equality"; }
def NeOp : LogicalEqOpBase<"ne"> { let summary = "Logical inequality"; }

class CaseEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  Commutative,
  SameTypeOperands
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0 or 1 result. If all corresponding bits in the left- and
    right-hand side are equal (both 0, 1, X, or Z), the two operands are
    considered equal (`case_eq` returns 1, `case_ne` returns 0). If any bits are
    not equal, the two operands are considered not equal (`case_eq` returns 0,
    `case_ne` returns 1). `case_eq` corresponds to the `===` operator and
    `case_ne` to the `!==` operator.

    See IEEE 1800-2017 § 11.4.5 "Equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

def CaseEqOp : CaseEqOpBase<"case_eq"> { let summary = "Case equality"; }
def CaseNeOp : CaseEqOpBase<"case_ne"> { let summary = "Case inequality"; }

class WildcardEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  SameTypeOperands,
  TypesMatchWith<"result is single bit matching input domain",
    "lhs", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0, 1, or X result. If any bit in the left-hand side is Z or X,
    returns X. Performs the same comparison as the `eq` and `ne` operations, but
    all right-hand side bits that are X or Z are skipped. Therefore, X and Z in
    the right-hand side act as wildcards or "don't care" values. `wildcard_eq`
    corresponds to the `==?` operator and `wildcard_ne` to the `!=?` operator.

    See IEEE 1800-2017 § 11.4.6 "Wildcard equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def WildcardEqOp : WildcardEqOpBase<"wildcard_eq"> {
  let summary = "Wildcard equality";
}
def WildcardNeOp : WildcardEqOpBase<"wildcard_ne"> {
  let summary = "Wildcard inequality";
}

class RelationalOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  SameTypeOperands,
  TypesMatchWith<"result is single bit matching input domain",
    "lhs", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let description = [{
    Compares the left- and right-hand side operand and returns a single bit 0,
    1, or X result. If any bit in the two operands is Z or X, returns X.
    Otherwise, if all bits are 0 or 1, `lt`, `le`, `gt`, and `ge` return whether
    the left-hand side is less than, less than or equal to, greater than, or
    greater than or equal to the right-hand side, respectively. `lt` corresponds
    to the `<` operator, `le` to `<=`, `gt` to `>`, and `ge` to `>=`.

    See IEEE 1800-2017 § 11.4.4 "Relational operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def LtOp : RelationalOpBase<"lt"> { let summary = "Less than"; }
def LeOp : RelationalOpBase<"le"> { let summary = "Less than or equal"; }
def GtOp : RelationalOpBase<"gt"> { let summary = "Greater than"; }
def GeOp : RelationalOpBase<"ge"> { let summary = "Greater than or equal"; }

def ConcatOp : MooreOp<"concat", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "A concatenation of expressions";
  let description = [{
    This operation represents the SystemVerilog concatenation expression
    `{x, y, z}`. See IEEE 1800-2017 §11.4.12 "Concatenation operators".

    All operands must be simple bit vector types.

    The concatenation result is a simple bit vector type. The result is unsigned
    regardless of the sign of the operands (see concatenation-specific rules in
    IEEE 1800-2017 §11.8.1 "Rules for expression types"). The size of the result
    is the sum of the sizes of all operands. If any of the operands is
    four-valued, the result is four-valued; otherwise it is two-valued.
  }];
  let arguments = (ins Variadic<SimpleBitVectorType>:$values);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $values attr-dict `:` functional-type($values, $result)
  }];
}

#endif // CIRCT_DIALECT_MOORE_MOOREOPS
