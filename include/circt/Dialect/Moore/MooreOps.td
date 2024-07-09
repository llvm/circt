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
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

// Base class for the operations in this dialect.
class MooreOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

class ResultIsSingleBitMatchingInputDomain<string result, string input> :
  TypesMatchWith<"result is single bit matching input domain",
    input, result, [{
    IntType::get($_self.getContext(), 1,
      llvm::cast<UnpackedType>($_self).getDomain())
  }]>;

//===----------------------------------------------------------------------===//
// Structure
//===----------------------------------------------------------------------===//

def SVModuleOp : MooreOp<"module", [
  IsolatedFromAbove,
  RegionKindInterface,
  Symbol,
  SingleBlockImplicitTerminator<"OutputOp">,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "A module definition";
  let description = [{
    The `moore.module` operation represents a SystemVerilog module, including
    its name, port list, and the constituent parts that make up its body. The
    module's body is a graph region.

    See IEEE 1800-2017 § 3.3 "Modules" and § 23.2 "Module definitions".
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<ModuleType>:$module_type,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region SizedRegion<1>:$bodyRegion);
  let hasCustomAssemblyFormat = 1;

  let builders = [OpBuilder<
    (ins "StringRef":$name, "hw::ModuleType":$type)>];
  let extraClassDeclaration = [{
    /// Return the `moore.output` op terminator of this module.
    OutputOp getOutputOp();
    /// Return the list of values assigned to output ports.
    OperandRange getOutputs();
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { 
      return RegionKind::Graph;
    }
  }];
}

def OutputOp : MooreOp<"output", [
  Terminator, HasParent<"SVModuleOp">, Pure, ReturnLike
]> {
  let summary = "Assign module outputs";
  let description = [{
    The `moore.output` operation marks the end of a `moore.module` body region
    and specifies the values to present for the module's output ports.
  }];

  let arguments = (ins Variadic<AnyType>:$outputs);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` type($outputs))?
  }];
  let hasVerifier = 1;
}

def InstanceOp : MooreOp<"instance", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "Create an instance of a module";
  let description = [{
    The `moore.instance` operation instantiates a `moore.module` operation.

    See IEEE 1800-2017 § 23.3 "Module instances".
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    FlatSymbolRefAttr:$moduleName,
    Variadic<AnyType>:$inputs,
    StrArrayAttr:$inputNames,
    StrArrayAttr:$outputNames
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
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

def VariableOp : MooreOp<"variable", [
  DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"initial value and variable types match",
    "result", "initial", "cast<RefType>($_self).getNestedType()">
]> {
  let summary = "A variable declaration";
  let description = [{
    See IEEE 1800-2017 § 6.8 "Variable declarations".
  }];
  let arguments = (ins StrAttr:$name, Optional<UnpackedType>:$initial);
  let results = (outs Res<RefType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) ($initial^)? attr-dict
    `:` type($result)
  }];
  let hasCanonicalizeMethod = true;
}

def NetKindAttr : I32EnumAttr<"NetKind", "Net type kind", [
    I32EnumAttrCase<"Supply0", 0, "supply0">,
    I32EnumAttrCase<"Supply1", 1, "supply1">,
    I32EnumAttrCase<"Tri", 2, "tri">,
    I32EnumAttrCase<"TriAnd", 3, "triand">,
    I32EnumAttrCase<"TriOr", 4, "trior">,
    I32EnumAttrCase<"TriReg", 5, "trireg">,
    I32EnumAttrCase<"Tri0", 6, "tri0">,
    I32EnumAttrCase<"Tri1", 7, "tri1">,
    I32EnumAttrCase<"UWire", 8, "uwire">,
    I32EnumAttrCase<"Wire", 9, "wire">,
    I32EnumAttrCase<"WAnd", 10, "wand">,
    I32EnumAttrCase<"WOr", 11, "wor">,
    I32EnumAttrCase<"Interconnect", 12, "interconnect">,
    I32EnumAttrCase<"UserDefined", 13, "userdefined">,
    I32EnumAttrCase<"Unknown", 14, "unknown">,
  ]> {
  let cppNamespace = "::circt::moore";
}

def NetOp : MooreOp<"net", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"assigned value and variable types match",
    "result", "assignment", "cast<RefType>($_self).getNestedType()">,
]> {
  let summary = "A net declaration";
  let description = [{
    The `moore.net` operation is a net declaration. Net types defines different
    types of net connection in SV. There are twelve built-in net types defined
    in the official standard construct of the operation:
    `supply0`, `supply1`, `tri`, `triand`, `trior`, `trireg`, `tri0`, `tri1`,
    `uwire`, `wire`, `wand`, `wor`.
    Optional assignment argument allows net operation to be initialized with
    specific values as soon as it is created. Only one net declaration
    assignment can be made for a particular net. See IEEE 1800-2017 § 10.3.1
    "The net declaration assignment" for the differences between net declaration
    assignments and continuous assign statements. It has some features that are
    not supported: declaring an interconnect net and using user-defined types in
    the net operation.
    
    See IEEE 1800-2017 § 6.7 "Net declarations".
  }];
  let arguments = (ins
    StrAttr:$name,
    NetKindAttr:$kind,
    Optional<UnpackedType>:$assignment
  );
  let results = (outs Res<RefType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $kind ($assignment^)? attr-dict
    `:` type($result)
  }];
}

def AssignedVarOp : MooreOp<"assigned_variable", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  TypesMatchWith<"initial value and variable types match",
    "result", "initial", "cast<RefType>($_self).getNestedType()">
]> {
  let summary = "The copy of variable must have the initial value";
  let arguments = (ins StrAttr:$name, UnpackedType:$initial);
  let results = (outs Res<RefType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $initial attr-dict `:` type($result)
  }];
}

def ReadOp : MooreOp<"read", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
  TypesMatchWith<"input and result types match",
    "result", "input", "RefType::get(cast<UnpackedType>($_self))">
]> {
  let summary = "Read the current value of a declaration";
  let description = [{
    Samples the current value of a declaration. This is a helper to capture the
    exact point at which declarations that can be targeted by all possible 
    expressions are read. It's similar to llvm.load.
  }];
  let arguments = (ins Arg<RefType, "", [MemRead]>:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Assignments
//===----------------------------------------------------------------------===//

class AssignOpBase<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits # [TypesMatchWith<"src and dst types match",
    "src", "dst", "RefType::get(cast<UnpackedType>($_self))">]> {
  let arguments = (ins RefType:$dst, UnpackedType:$src);
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
  DeclareOpInterfaceMethods<PromotableMemOpInterface>
]> {
  let summary = "Blocking procedural assignment";
  let description = [{
    A blocking procedural assignment in a sequential block, such as `x = y`. The
    effects of the assignment are visible to any subsequent operations in the
    block.

    See IEEE 1800-2017 § 10.4.1 "Blocking procedural assignments".
  }];
  let arguments = (ins
    Arg<RefType, "", [MemWrite]>:$dst,
    UnpackedType:$src
  );
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
/// Transit from 0 to x/z/1, and from x/z to 1.
def PosEdge: I32EnumAttrCase<"PosEdge", 1, "posedge">;
/// Transit from 1 to x/z/0, and from x/z to 0.
def NegEdge: I32EnumAttrCase<"NegEdge", 2, "negedge">;
/// Include the negedge and posedge.
def BothEdges: I32EnumAttrCase<"BothEdges", 3, "edge">;

def EdgeAtrr: I32EnumAttr<"Edge", "Edge kind",
                          [None, PosEdge, NegEdge, BothEdges]>{
  let cppNamespace = "circt::moore";
}

def EventOp : MooreOp<"wait_event", [
  HasParent<"ProcedureOp">
]> {
  let summary = "Detecting posedge and negedge";
  let description = [{
    It is introduced by the symbol `@`, and it allows statement execution to
    be delayed until the occurrence of some simulation event occurring in a
    procedure executing concurrently with this procedure.

    For the implicit event control(`@(*)`), there are two situations that are
    not automatically added to event expression:
      1. Identifiers that only appear in wait or event expressions.
         ```
          always @(*) begin    // equivalent to @(b)
            @(n) kid = b;      // n is not added to @(*)
          end
         ```
      2. Identifiers that only appear as a hierarchical_variable_identifier
         in the variable_lvalue of the left-hand side of assignments.
         ```
          always @(*) begin
            a = b + c;         // a is not added to @(*)
          end
         ```

    Example:
    ```
      @(a, b, c)        // none
      @(posedge clk)    // positive edge
      @(negedge clk)    // negative edge
      @(edge clk)       // both edge
      @(*)              // implicit event control
    ```
    See IEEE 1800-2017 § 9.4.2 "Event control".
  }];
  let arguments = (ins EdgeAtrr:$edge, UnpackedType:$input);
  let results = (outs);
  let assemblyFormat = [{
    $edge $input attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

def ConstantOp : MooreOp<"constant", [Pure]> {
  let summary = "A constant integer value";
  let arguments = (ins APIntAttr:$value);
  let results = (outs IntType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "IntType":$type, "const APInt &":$value)>,
    OpBuilder<(ins "IntType":$type, "int64_t":$value)>,
  ];
}

def Parameter : I32EnumAttrCase<"Parameter", 0, "parameter">;
def LocalParameter : I32EnumAttrCase<"LocalParameter", 1, "localparam">;
def SpecParameter : I32EnumAttrCase<"SpecParameter", 2, "specparam">;

def NamedConstAttr : I32EnumAttr<"NamedConst", "elaboration-time constants", 
       [Parameter, LocalParameter, SpecParameter]>{
    let cppNamespace = "circt::moore";
}

def NamedConstantOp : MooreOp<"named_constant", [
  SameOperandsAndResultType,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]>{
  let summary = "An elaboration time constant expression";
  let description = [{
    Constants are named data objects that never change. SystemVerilog provides
    three elaboration-time constants: parameter, localparam, and specparam.

    See IEEE 1800-2017 § 6.20 "Constants".
  }];
  let arguments = (ins StrAttr:$name,
                       NamedConstAttr:$kind,
                       IntType:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $kind
    ``custom<ImplicitSSAName>($name)
    $value `:` type($result) attr-dict
  }];
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
  ResultIsSingleBitMatchingInputDomain<"result", "input">
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
  ResultIsSingleBitMatchingInputDomain<"result", "input">
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

class DivOpBase<string mnemonic> : BinaryOpBase<mnemonic> {
  let summary = "Division";
  let description = [{
    Divide the left-hand side by the right-hand side operand. Any fractional
    part is truncated toward zero. If the right-hand side is zero, all bits of
    the result are X. If any bit in the two operands is Z or X, all bits in the
    result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def DivUOp : DivOpBase<"divu">;
def DivSOp : DivOpBase<"divs">;

class ModOpBase<string mnemonic> : BinaryOpBase<mnemonic> {
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

def ModUOp : ModOpBase<"modu">;
def ModSOp : ModOpBase<"mods">;

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
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
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
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
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
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
]> {
  let description = [{
    Compares the left- and right-hand side operand and returns a single bit 0,
    1, or X result. If any bit in the two operands is Z or X, returns X.
    Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
    `uge/sge` return whether the left-hand side is less than, less than or equal
    to, greater than, or greater than or equal to the right-hand side,
    respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
    `ugt/sgt` to `>`, and `uge/sge` to `>=`.

    See IEEE 1800-2017 § 11.4.4 "Relational operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def UltOp : RelationalOpBase<"ult"> {
  let summary = "Unsigned less than comparison";
}
def UleOp : RelationalOpBase<"ule"> {
  let summary = "Unsigned less than or equal comparison";
}
def UgtOp : RelationalOpBase<"ugt"> {
  let summary = "Unsigned greater than comparison";
}
def UgeOp : RelationalOpBase<"uge"> {
  let summary = "Unsigned greater than or equal comparison";
}

def SltOp : RelationalOpBase<"slt"> {
  let summary = "Signed less than comparison";
}
def SleOp : RelationalOpBase<"sle"> {
  let summary = "Signed less than or equal comparison";
}
def SgtOp : RelationalOpBase<"sgt"> {
  let summary = "Signed greater than comparison";
}
def SgeOp : RelationalOpBase<"sge"> {
  let summary = "Signed greater than or equal comparison";
}

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
  let arguments = (ins Variadic<IntType>:$values);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $values attr-dict `:` `(` type($values) `)` `->` type($result)
  }];
}

def ConcatRefOp : MooreOp<"concat_ref", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "The copy of concat that explicitly works on the ref type.";
  let arguments = (ins Variadic<RefType>:$values);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $values attr-dict `:` `(` type($values) `)` `->` type($result)
  }];
}

def ReplicateOp : MooreOp<"replicate", [
  Pure
]> {
  let summary = "Multiple concatenation of expressions";
  let description = [{
    This operation indicates a joining together of that many copies of the 
    concatenation `{constant{w}}`. Which enclosed together within brace.
    The 'constant' must a non-negative, non-x, and non-z constant expression.
    The 'constant' may be a value of zero, but it only exists in parameterized
    code, and it will be ignored(type is changed to the void).

    Example:
    ```
      {0{w}}   // empty! ignore it.
      {4{w}}   // the same as {w, w, w, w}
    ```
    See IEEE 1800-2017 §11.4.12 "Concatenation operators".
  }];
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def ExtractOp : MooreOp<"extract"> {
  let summary = "Addressing operation";
  let description = [{
    This operation includes the vector bit/part-select, array, and memory 
    addressing.If the address is invalid--out of bounds or has x or z bit--
    then it will produce x for 4-state or 0 for 2-state.
    Bit-select results are unsigned, regardless of the operands.
    Part-select results are unsigned, regardless of the operands even if 
    the part-select specifies the entire vector.
    See IEEE 1800-2017 § 11.8.1 "Rules for expression types"

    Example:
    ```
    logic v [7:0];
    v[1];                      // the bit-select addressing
    v[3:0];                    // the part-select addressing
    v[3-:4];  v[0+:4];         // They are equivalent to v[3:0]
    ```
    See IEEE 1800-2017 § 11.5.1 "Vector bit-select and part-select addressing".

    Example:
    ```
    // an array of 256-by-256 8-bit elements
    logic [7:0] twod_array [0:255][0:255];
    logic [7:0] mem_name [0:1023];      // a memory of 1024 8-bit words
    ```
    See IEEE 1800-2017 § 11.5.2 "Array and memory addressing".
  }];
  let arguments = (ins UnpackedType:$input, UnpackedType:$lowBit);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:`
    type($input) `,` type($lowBit) `->` type($result)
  }];
}

def ExtractRefOp : MooreOp<"extract_ref"> {
  let summary = "The copy of extract that explicitly works on the ref type.";
  let arguments = (ins RefType:$input, UnpackedType:$lowBit);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:`
    type($input) `,` type($lowBit) `->` type($result)
  }];
}

def StructCreateOp : MooreOp<"struct_create", [SameOperandsAndResultType]> {
  let summary = "Struct Create operation";
  let description = [{
    A structure represents a collection of data types
    that can be referenced as a whole, or the individual data types
    that make up the structure can be referenced by name.
    By default, structures are unpacked, meaning that there is
    an implementation-dependent packing of the data types.
    Unpacked structures can contain any data type.
    See IEEE 1800-2017 § 7.2 "Structures"

    Example:
    ```
    struct { bit [7:0] opcode; bit [23:0] addr; }IR;
    IR.opcode = 1;   // set field in IR.
    // anonymous structure
    // defines variable IR
    typedef struct {
    bit [7:0] opcode;
    bit [23:0] addr;
    } instruction; // named structure type
    instruction IR; // define variable
    ```
    See IEEE 1800-2017 § 7.2. "Structures".
  }];
  let arguments = (ins UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
  $input attr-dict `:` type($input)
  }];
}

def StructExtractOp : MooreOp<"struct_extract"> {
  let summary = "Struct Extract operation";
  let description = [{
    Structures can be converted to bits preserving the bit pattern.
    In other words, they can be converted back to the same value
    without any loss of information. When unpacked data are converted
    to the packed representation, the order of the data in the packed
    representation is such that the first field in the structure
    occupies the MSBs. The effect is the same as a concatenation of
    the data items (struct fields or array elements) in order.
    The type of the elements in an unpacked structure or array
    shall be valid for a packed representation in order to be
    cast to any other type, whether packed or unpacked.
    See IEEE 1800-2017 § 6.24.1 "Cast operator"

    Example:
    ```
    typedef struct {
    int addr = 1 + constant;
    int crc;
    byte data [4] = '{4{1}};
    } packet1;

    packet1 p1; // initialization defined by the typedef.
                // p1.crc will use the default value for an int
    ```
    See IEEE 1800-2017 § 7.2.1 "Assigning to structures".
  }];
  let arguments = (ins StrAttr:$memberName, UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
  $input `,`   $memberName  attr-dict `:`
  type($input) `->`
  type($result)
  }];
}

def StructInjectOp : MooreOp<"struct_inject"> {
  let summary = "Struct Field operation";
  let description = [{
    A structure can be assigned as a whole and passed to
    or from a subroutine as a whole. Members of a structure
    data type can be assigned individual default member
    values by using an initial assignment with the declaration
    of each member. The assigned expression shall be
    a constant expression.
    See IEEE 1800-2017 § 7.2.2 "Assigning to structures"

    Example:
    ```
    typedef struct {
    int addr = 1 + constant;
    int crc;
    byte data [4] = '{4{1}};
    } packet1;

    packet1 p1; // initialization defined by the typedef.
                // p1.crc will use the default value for an int
    ```
    See IEEE 1800-2017 § 7.2. "Assigning to structures".
  }];
  let arguments = (ins UnpackedType:$LHS, StrAttr:$memberName,
  UnpackedType:$RHS);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $LHS `,`   $memberName `,` $RHS
    attr-dict `:` type($LHS) type($RHS) `->` type($result)
  }];
}

def UnionCreateOp : MooreOp<"union_create"> {
  let summary = "Union Create operation";
  let description = [{
    A union is a data type that represents a single piece
    of storage that can be accessed using one of
    the named member data types. Only one of the
    data types in the union can be used at a time.
    By default, a union is unpacked, meaning there
    is no required representation for how members
    of the union are stored. Dynamic types and chandle
    types can only be used in tagged unions.
    See IEEE 1800-2017 § 7.3 "Unions"

    Example:
    ```
    typedef union { int i; shortreal f; } num; // named union type
    num n;
    n.f = 0.0; // set n in floating point format
    typedef struct {
    bit isfloat;
    union { int i; shortreal f; } n;           // anonymous union type
    } tagged_st;                               // named structure
    ```
    See IEEE 1800-2017 § 7.3 "Unions"
  }];
  let arguments = (ins StrAttr:$unionName, UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
  $unionName `,` $input attr-dict `:`
  type($input) `->` type($result)
  }];
}

def UnionExtractOp : MooreOp<"union_extract"> {
  let summary = "Union Extract operation";
  let description = [{
    With packed unions, writing one member and reading another is
    independent of the byte ordering of the machine,
    unlike an unpacked union of unpacked structures,
    which are C-compatible and have members in ascending address order.
    See IEEE 1800-2017 § 7.3.1 "Packed unions"

    Example:
    ```
    typedef union packed { // default unsigned
    s_atmcell acell;
    bit [423:0] bit_slice;
    bit [52:0][7:0] byte_slice;
    } u_atmcell;
    u_atmcell u1;
    byte b; bit [3:0] nib;
    b = u1.bit_slice[415:408]; // same as b = u1.byte_slice[51];
    nib = u1.bit_slice [423:420];
    ```
    See IEEE 1800-2017 § 7.3.1 "Packed unions"
  }];
  let arguments = (ins StrAttr:$memberName, UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
  $input `,`   $memberName  attr-dict `:`
  type($input) `->` type($result)
  }];
}

def ConditionalOp : MooreOp<"conditional",[
  RecursiveMemoryEffects,
  NoRegionArguments, 
  SingleBlockImplicitTerminator<"moore::YieldOp">
]> {
  let summary = "Conditional operation";
  let description = [{
    If cond_predicate is true, the operator returns the value of the first 
    expression without evaluating the second expression; if false, it returns 
    the value of the second expression without evaluating the first expression.
    If cond_predicate evaluates to an ambiguous value (x or z), then both the
    first expression and the second expression shall be evaluated, and compared
    for logical equivalence. If that comparison is true (1), the operator shall
    return either the first or second expression. Otherwise the operator returns
    a result based on the data types of the expressions.

    When both the first and second expressions are of integral types, if the
    cond_predicate evaluates to an ambiguous value and the expressions are not
    logically equivalent, their results shall be combined bit by bit using the
    table below to calculate the final result. The first and second expressions
    are extended to the same width.

    |?: | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | X | X | X |
    | 1 | X | 1 | X | X |
    | X | X | X | X | X |
    | Z | X | X | X | X |

    See IEEE 1800-2017 § 11.4.11 "Conditional operator".
  }];
  let arguments = (ins AnySingleBitType:$condition);
  let results = (outs UnpackedType:$result);
  let regions = (region SizedRegion<1>:$trueRegion,
                        SizedRegion<1>:$falseRegion);
  let assemblyFormat = [{
    $condition attr-dict `:` type($condition) `->` type($result)
    $trueRegion $falseRegion
  }];
}

def YieldOp : MooreOp<"yield", [
  Pure, 
  Terminator, 
  HasParent<"ConditionalOp">
]> {
  let summary = "conditional yield and termination operation";
  let description = [{
    "moore.yield" yields an SSA value from the Moore dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "moore.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "moore.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];
  let arguments = (ins UnpackedType:$result);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict $result `:` type($result)
  }];
  let hasVerifier = 1;
}
#endif // CIRCT_DIALECT_MOORE_MOOREOPS
