//===- MooreOps.td - Moore dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREOPS
#define CIRCT_DIALECT_MOORE_MOOREOPS

include "circt/Dialect/Moore/MooreAttributes.td"
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

  let arguments = (ins ProcedureKindAttr:$kind);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $kind attr-dict-with-keyword $body
  }];
}

def ReturnOp : MooreOp<"return", [
  Pure, Terminator, HasParent<"ProcedureOp">
]> {
  let summary = "Return from a procedure";
  let assemblyFormat = [{ attr-dict }];
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

def VariableOp : MooreOp<"variable", [
  DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>,
  DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"initial value and variable types match",
    "result", "initial", "cast<RefType>($_self).getNestedType()">
]> {
  let summary = "A variable declaration";
  let description = [{
    See IEEE 1800-2017 § 6.8 "Variable declarations".
  }];
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    Optional<UnpackedType>:$initial
  );
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
    OptionalAttr<StrAttr>:$name,
    NetKindAttr:$kind,
    Optional<UnpackedType>:$assignment
  );
  let results = (outs Res<RefType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $kind ($assignment^)? attr-dict
    `:` type($result)
  }];
  let hasCanonicalizeMethod = true;
}

def AssignedVariableOp : MooreOp<"assigned_variable", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  SameOperandsAndResultType
]> {
  let summary = "A variable with a unique continuously assigned value";
  let arguments = (ins OptionalAttr<StrAttr>:$name, UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $input attr-dict `:` type($input)
  }];
  let hasCanonicalizeMethod = true;
}

def ReadOp : MooreOp<"read", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
  TypesMatchWith<"input and result types match",
    "input", "result", "cast<RefType>($_self).getNestedType()">
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
    $input attr-dict `:` type($input)
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

// Any change on the input.
def AnyChange: I32EnumAttrCase<"AnyChange", 0, "any">;
// A transition from 0 to X/Z/1, or from X/Z to 1.
def PosEdge: I32EnumAttrCase<"PosEdge", 1, "posedge">;
// A transition from 1 to X/Z/0, or from X/Z to 0.
def NegEdge: I32EnumAttrCase<"NegEdge", 2, "negedge">;
// The combination of `PosEdge` and `NegEdge`.
def BothEdges: I32EnumAttrCase<"BothEdges", 3, "edge">;

def EdgeAttr: I32EnumAttr<"Edge", "Edge kind",
                          [AnyChange, PosEdge, NegEdge, BothEdges]> {
  let cppNamespace = "circt::moore";
}

def WaitEventOp : MooreOp<"wait_event", [
  RecursiveMemoryEffects,
  NoRegionArguments,
  SingleBlock,
  NoTerminator
]> {
  let summary = "Suspend execution until an event occurs";
  let description = [{
    The `moore.wait_event` op suspends execution of the current process until
    its body signals that an event has been the detected. Conceptually, the body
    of this op is executed whenever any potentially relevant signal has changed.
    If one of the contained `moore.detect_event` ops detect an event, execution
    resumes after the `moore.wait_event` operation. If no event is detected, the
    current process remains suspended.

    Example corresponding to the SystemVerilog `@(posedge x, negedge y iff z)`:
    ```
    moore.wait_event {
      %0 = moore.read %x : <i1>
      %1 = moore.read %y : <i1>
      %2 = moore.read %z : <i1>
      moore.detect_event posedge %0 : i1
      moore.detect_event negedge %1 if %2 : i1
    }
    ```

    The body may also contain any operations necessary to evaluate the event
    conditions. For example, the SV `@(posedge ~x iff i == 42)`:
    ```
    moore.wait_event {
      %0 = moore.read %x : <i1>
      %1 = moore.not %0 : i1
      %2 = moore.read %i : <i19>
      %3 = moore.constant 42 : i19
      %4 = moore.eq %2, %3 : i19
      moore.detect_event posedge %0 if %4 : i1
    }
    ```

    See IEEE 1800-2017 § 9.4.2 "Event control".
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{ attr-dict-with-keyword $body }];
}

def DetectEventOp : MooreOp<"detect_event", [
  HasParent<"WaitEventOp">
]> {
  let summary = "Check if an event occured within a `wait_event` op";
  let description = [{
    The `moore.detect_event` op is used inside the body of a `moore.wait_event`
    to check if an interesting value change has occurred on its operand. The
    `moore.detect_event` op implicitly stores the previous value of its operand
    and compares it against the current value to detect an interesting edge:

    - `posedge` checks for a low-to-high transition
    - `negedge` checks for a high-to-low transition
    - `edge` checks for either a `posedge` or a `negedge`
    - `any` checks for any value change (including e.g. X to Z)

    The edges are detected as follows:

    - `0` to `1 X Z`: `posedge`
    - `1` to `0 X Z`: `negedge`
    - `X Z` to `1`: `posedge`
    - `X Z` to `0`: `negedge`

    | From  | To 0    | To 1    | To X    | To Z    |
    |-------|---------|---------|---------|---------|
    | 0     | -       | posedge | posedge | posedge |
    | 1     | negedge | -       | negedge | negedge |
    | X     | negedge | posedge | -       | -       |
    | Z     | negedge | posedge | -       | -       |

    See IEEE 1800-2017 § 9.4.2 "Event control".
  }];
  let arguments = (ins
    EdgeAttr:$edge,
    UnpackedType:$input,
    Optional<BitType>:$condition
  );
  let assemblyFormat = [{
    $edge $input (`if` $condition^)? attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

def ConstantOp : MooreOp<"constant", [Pure, ConstantLike]> {
  let summary = "A constant integer value";
  let arguments = (ins FVIntegerAttr:$value);
  let results = (outs IntType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "IntType":$type, "const FVInt &":$value)>,
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

def StringConstantOp : MooreOp<"string_constant", [Pure]> {
  let summary = "Produce a constant string value";
  let description = [{
    Produces a constant value of string type.

    Example:
    ```mlir
    %0 = moore.string "hello world"
    ```
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
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

class PowOpBase<string mnemonic> : BinaryOpBase<mnemonic> {
  let summary = "Power";
  let description = [{
    Raise the left-hand side to the power of the right-hand side. `powu` treats
    its operands as unsigned numbers, while `pows` treats them as signed
    numbers.

    Evaluation rules for `a ** b`:

    |       | a < -1 | a = -1         | a = 0 | a = 1 | a > 1  |
    |-------|--------|----------------|-------|-------|--------|
    | b > 0 | a ** b | b odd ? -1 : 1 |     0 |     1 | a ** b |
    | b = 0 |      1 |              1 |     1 |     1 |      1 |
    | b < 0 |      0 | b odd ? -1 : 1 |     X |     1 |      0 |

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def PowUOp : PowOpBase<"powu">;
def PowSOp : PowOpBase<"pows">;

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

    `casez_eq` treats Z bits in either operand as wildcards and skips them
    during the comparison. `casexz_eq` treats X and Z bits as wildcards. These
    are different from the `wildcard_eq` operation, which only considers X/Z in
    the right-hand operand as wildcards.

    Case statements use this operation to perform case comparisons:
    - `case` statements use `case_eq`
    - `casez` statements use `casez_eq`
    - `casex` statements use `casexz_eq`

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
def CaseZEqOp : CaseEqOpBase<"casez_eq"> {
  let summary = "Case equality with Z as wildcard";
}
def CaseXZEqOp : CaseEqOpBase<"casexz_eq"> {
  let summary = "Case equality with X and Z as wildcard";
}

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

//===----------------------------------------------------------------------===//
// Bit/Element Extraction
//===----------------------------------------------------------------------===//

def ExtractOp : MooreOp<"extract", [Pure]> {
  let summary = "Extract a range or single bits from a value";
  let description = [{
    It's used to select from a value with a constant low bit.
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
  let arguments = (ins UnpackedType:$input, I32Attr:$lowBit);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
  }];
}

def DynExtractOp : MooreOp<"dyn_extract", [Pure]> {
  let description = [{
    It's similar with extract, but it's used to select from a value
    with a dynamic low bit.
  }];
  let arguments = (ins UnpackedType:$input, UnpackedType:$lowBit);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:`
    type($input) `,` type($lowBit) `->` type($result)
  }];
}

def ExtractRefOp : MooreOp<"extract_ref", [Pure]> {
  let description = [{
  The copy of extract that explicitly works on the ref type.
  }];
  let arguments = (ins RefType:$input, I32Attr:$lowBit);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
  }];
}

def DynExtractRefOp : MooreOp<"dyn_extract_ref", [Pure]> {
  let description = [{
  The copy of dyn_extract that explicitly works on the ref type.
  }];
  let arguments = (ins RefType:$input, UnpackedType:$lowBit);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:`
    type($input) `,` type($lowBit) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Array Manipulation
//===----------------------------------------------------------------------===//

def ArrayCreateOp : MooreOp<"array_create", [Pure, SameTypeOperands]> {
  let summary = "Create an array value from individual elements";
  let arguments = (ins Variadic<UnpackedType>:$elements);
  let results = (outs AnyStaticArrayType:$result);
  let assemblyFormat = [{
    $elements attr-dict `:` type($elements) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Struct Manipulation
//===----------------------------------------------------------------------===//

def StructCreateOp : MooreOp<"struct_create", [Pure]> {
  let summary = "Create a struct value from individual fields";
  let arguments = (ins Variadic<UnpackedType>:$fields);
  let results = (outs AnyStructType:$result);
  let assemblyFormat = [{
    $fields attr-dict `:` type($fields) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def StructExtractOp : MooreOp<"struct_extract", [Pure]> {
  let summary = "Obtain the value of a struct field";
  let arguments = (ins StrAttr:$fieldName, AnyStructType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `,` $fieldName attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def StructExtractRefOp : MooreOp<"struct_extract_ref", [
  Pure,
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>
]> {
  let summary = "Create a reference to a struct field";
  let arguments = (ins StrAttr:$fieldName, AnyStructRefType:$input);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `,` $fieldName attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def StructInjectOp : MooreOp<"struct_inject", [
  Pure,
  AllTypesMatch<["input", "result"]>
]> {
  let summary = "Update the value of a struct field";
  let description = [{
    Takes an existing struct value, sets one of its fields to a new value, and
    returns the resulting struct value.
  }];
  let arguments = (ins
    AnyStructType:$input,
    StrAttr:$fieldName,
    UnpackedType:$newValue
  );
  let results = (outs AnyStructType:$result);
  let assemblyFormat = [{
    $input `,` $fieldName `,` $newValue attr-dict
    `:` type($input) `,` type($newValue)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = true;
}

//===----------------------------------------------------------------------===//
// Union Manipulation
//===----------------------------------------------------------------------===//

def UnionCreateOp : MooreOp<"union_create", [Pure]> {
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
  let arguments = (ins UnpackedType:$input, StrAttr:$fieldName);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
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
  let arguments = (ins StrAttr:$fieldName, UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `,`   $fieldName  attr-dict `:`
    type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def UnionExtractRefOp : MooreOp<"union_extract_ref"> {
  let summary = "Union Extract operation";
  let arguments = (ins StrAttr:$fieldName, RefType:$input);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `,`   $fieldName  attr-dict `:`
    type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def ConditionalOp : MooreOp<"conditional",[
  RecursiveMemoryEffects,
  NoRegionArguments,
]> {
  let summary = "Conditional operation";
  let description = [{
    If the condition is true, this op evaluates the first region and returns its
    result without evaluating the second region. If the the condition is false,
    this op evaluates the second region and returns its result without
    evaluating the first region.

    If the condition is unknown (X or Z), _both_ regions are evaluated. If both
    results are equal as per `case_eq`, one of the results is returned. If the
    results are not equal, this op returns a value based on the data types of
    the results.

    In case the results of the first and second region are of an integral type,
    they are merged by applying the following bit-wise truth table:

    |?: | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | X | X | X |
    | 1 | X | 1 | X | X |
    | X | X | X | X | X |
    | Z | X | X | X | X |

    Non-integral data types define other rules which are not yet implemented.
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
  let assemblyFormat = [{
    attr-dict $result `:` type($result)
  }];
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_MOORE_MOOREOPS
