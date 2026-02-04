//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_LLHDOPS_TD
#define CIRCT_DIALECT_LLHD_LLHDOPS_TD

include "circt/Dialect/LLHD/LLHDDialect.td"
include "circt/Dialect/LLHD/LLHDTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Base class for all LLHD operations.
class LLHDOp<string mnemonic, list<Trait> traits = []>
    : Op<LLHDDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

class SigPtrIndexBitWidthConstraint<string index, string input>
  : TypesMatchWith<"Index width should be exactly clog2 (size of array)",
                   input, index,
                   [{
                      IntegerType::get($_self.getContext(),
                        llvm::Log2_64_Ceil(llhd::getLLHDTypeWidth($_self)))
                   }]>;

class SigArrayElementTypeConstraint<string result, string input>
  : TypesMatchWith<"result must be a signal of the array element type",
                   input, result,
                   "RefType::get(llhd::getLLHDElementType($_self))">;

class SameSigPtrArrayElementTypeConstraint<string result, string input>
  : PredOpTrait<"arrays element type must match",
                CPred<"llhd::getLLHDElementType($" # result # ".getType()) == "
                      # "llhd::getLLHDElementType($" # input # ".getType())">>;

class SmallerOrEqualResultTypeWidthConstraint<string result, string input>
  : PredOpTrait<"width of " # result # " type has to be smaller than or " #
                "equal to the " # input # " type",
                CPred<"llhd::getLLHDTypeWidth($" # result # ".getType()) <= " #
                      "llhd::getLLHDTypeWidth($" # input  # ".getType())">>;

def ProceduralRegion : NativeOpTrait<"ProceduralRegion"> {
  let cppNamespace = "::circt::llhd";
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def ConstantTimeOp : LLHDOp<"constant_time",
                                  [ConstantLike, Pure]> {
  let summary = "Introduce a new time constant.";
  let description = [{
    The `llhd.constant_time` instruction introduces a new constant time value as
    an SSA-operator.

    Example:

    ```mlir
    %1 = llhd.constant_time #llhd.time<1ns, 2d, 3d>
    ```
  }];

  let assemblyFormat = "$value attr-dict";

  let builders = [
    /// Build a ConstantTimeOp from an ArrayRef with three values
    /// (real-time, delta, epsilon) and a string for the real-time unit.
    OpBuilder<(ins "uint64_t":$time,
                   "const StringRef &":$timeUnit,
                   "unsigned":$delta,
                   "unsigned":$epsilon)>
  ];

  let arguments = (ins LLHDTimeAttr: $value);
  let results = (outs LLHDTimeType: $result);

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Time Operations
//===----------------------------------------------------------------------===//

def CurrentTimeOp : LLHDOp<"current_time", [MemoryEffects<[MemRead]>]> {
  let summary = "Get the current simulation time";
  let description = [{
    Materializes the current simulation time as an SSA value. This is equivalent
    to the `$time`, `$stime`, and `$realtime` system tasks in SystemVerilog, and
    the `now` keyword in VHDL.

    This operation has a memory read side effect to avoid motion and CSE across
    `llhd.wait` operations, and other operations that may suspend execution.
  }];
  let results = (outs LLHDTimeType:$result);
  let assemblyFormat = "attr-dict";
}

def TimeToIntOp : LLHDOp<"time_to_int", [Pure]> {
  let summary = "Convert a time to an integer number of femtoseconds";
  let description = [{
    If the time value converted to femtoseconds does not fit into an `i64`, the
    result is undefined.
  }];
  let arguments = (ins LLHDTimeType:$input);
  let results = (outs I64:$result);
  let assemblyFormat = "$input attr-dict";
}

def IntToTimeOp : LLHDOp<"int_to_time", [Pure]> {
  let summary = "Convert an integer number of femtoseconds to a time";
  let arguments = (ins I64:$input);
  let results = (outs LLHDTimeType:$result);
  let assemblyFormat = "$input attr-dict";
}

//===----------------------------------------------------------------------===//
// Signal Operations
//===----------------------------------------------------------------------===//

def SignalOp : LLHDOp<"sig", [
  DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  TypeMatchesRefNestedType<"init", "result">,
]> {
  let summary = "Create a signal.";
  let description = [{
    The `llhd.sig` instruction introduces a new signal in the IR. The input
    operand determines the initial value carried by the signal, while the
    result type will always be a signal carrying the type of the init operand.
    A signal defines a unique name within the entity it resides in.

    Example:

    ```mlir
    %c123_i64 = hw.constant 123 : i64
    %foo = llhd.sig %c123_i64 : i64
    %0 = llhd.sig name "foo" %c123_i64 : i64
    ```

    This example creates a new signal named "foo", carrying an `i64` type with
    initial value of 123.
  }];
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    AnyType:$init
  );
  let results = (outs Res<RefType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $init attr-dict
    `:` type($init)
  }];
}

def ProbeOp : LLHDOp<"prb", [
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
  DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypeMatchesRefNestedType<"result", "signal">,
  RefNestedTypeMatchesType<"signal", "result">,
]> {
  let summary = "Probe a signal.";
  let description = [{
    This operation probes a signal and returns the value it
    currently carries as a new SSA operand. The result type is always
    the type carried by the signal. In SSACFG regions, the operation has a read
    side effect on the signal operand. In graph regions, the operation is
    memory-effect free.

    Example:

    ```mlir
    %true = hw.constant true
    %sig_i1 = llhd.sig %true : i1
    %prbd = llhd.prb %sig_i1 : i1
    ```
  }];

  let arguments = (ins RefType:$signal);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$signal attr-dict `:` type($result)";
}

def DriveOp : LLHDOp<"drv", [
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
  DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
  TypeMatchesRefNestedType<"value", "signal">,
]> {
  let summary = "Drive a value into a signal.";
  let description = [{
    The `llhd.drv` operation drives a new value onto a signal. A time
    operand also has to be passed, which specifies the frequency at which
    the drive will be performed. An optional enable value can be passed as
    last argument. In this case the drive will only be performed if the
    value is 1. In case no enable signal is passed the drive will always be
    performed. This operation does not define any new SSA operands.

    Example:

    ```mlir
    %true = hw.constant true
    %false = hw.constant false
    %time = llhd.constant_time <1ns, 0d, 0e>
    %sig = llhd.sig %true : i1

    llhd.drv %sig, %false after %time : i1
    llhd.drv %sig, %false after %time if %true : i1
    ```
  }];

  let arguments = (ins Arg<RefType, "the signal to drive to",
                           [MemWrite]>: $signal,
                       AnyType: $value,
                       LLHDTimeType: $time,
                       Optional<I1>: $enable);

  let assemblyFormat = [{
    $signal `,` $value `after` $time ( `if` $enable^ )? attr-dict `:`
    type($value)
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def OutputOp : LLHDOp<"output", [
  TypeMatchesRefNestedType<"value", "result">,
]> {
  let summary = "Introduce a new signal and drive a value onto it.";
  let description = [{
    The `llhd.output` operation introduces a new signal and continuously
    drives a the given value onto it after a given time-delay. The same
    value is used to initialize the signal in the same way as the 'init'
    value in `llhd.sig`. An optional name can be given to the created signal.
    This shows up, e.g., in the simulation trace.

    Example:

    ```mlir
    %value = hw.constant true
    %time = llhd.constant_time <1ns, 0d, 0e>
    %sig = llhd.output "sigName" %value after %time : i1

    // is equivalent to

    %value = hw.constant true
    %time = llhd.constant_time <1ns, 0d, 0e>
    %sig = llhd.sig "sigName" %value : i1
    llhd.drv %sig, %value after %time : i1
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>: $name,
                       AnyType: $value,
                       LLHDTimeType: $time);

  let results = (outs RefType: $result);

  let assemblyFormat = [{
    ( $name^ )? $value `after` $time attr-dict `:` type($value)
  }];
}

def DelayOp : LLHDOp<"delay", [Pure, SameOperandsAndResultType]> {
  let summary = "specifies value propagation delay";
  let description = [{
    This operation propagates all value changes of the input to the output after
    the specified time delay.
    Reference values are not supported (e.g., pointers, inout, etc.)
    since the store-like operation used for those types should encode a delayed
    store.
  }];

  let arguments = (ins HWNonInOutType:$input, LLHDTimeAttr:$delay);
  let results = (outs HWNonInOutType:$result);

  let assemblyFormat = "$input `by` $delay attr-dict `:` type($result)";
}

def GlobalSignalOp : LLHDOp<"global_signal", [
  IsolatedFromAbove,
  NoRegionArguments,
  SingleBlock,
  Symbol,
]> {
  let summary = "A global signal declaration";
  let description = [{
    Define a global signal identified by a symbol name. The corresponding
    `llhd.get_global_signal` operation can be used to get a `!llhd.ref` to it.
    Global signals behave just like normal signals.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<AnyType>:$type
  );
  let regions = (region MaxSizedRegion<1>:$initRegion);
  let assemblyFormat = [{
    $sym_name attr-dict `:` $type (`init` $initRegion^)?
  }];
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    Block *getInitBlock();
  }];
}

def GetGlobalSignalOp : LLHDOp<"get_global_signal", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure,
]> {
  let summary = "Get a reference to a global signal";
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $global_name attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "llhd::GlobalSignalOp":$global), [{
      build($_builder, $_state,
        RefType::get(global.getType()), global.getSymName());
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Integer Signal Projections
//===----------------------------------------------------------------------===//

def SigExtractOp : LLHDOp<"sig.extract", [
  Pure,
  SmallerOrEqualResultTypeWidthConstraint<"result", "input">,
  SigPtrIndexBitWidthConstraint<"lowBit", "input">,
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
  DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
]> {
  let summary = "Extract a range of bits from an integer signal";
  let description = [{
    The `llhd.sig.extract` operation allows to access a range of bits
    of the `$input` operand, starting at the index given by the `$lowBit`
    operand. The result length is defined by the result type.
  }];

  let arguments = (ins RefTypeOf<HWIntegerType>:$input,
                       HWIntegerType:$lowBit);
  let results = (outs RefTypeOf<HWIntegerType>: $result);

  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
  }];

  let extraClassDeclaration = [{
    unsigned getResultWidth()  { return getLLHDTypeWidth(getResult().getType()); }
    unsigned getInputWidth() { return getLLHDTypeWidth(getInput().getType()); }
  }];
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// Array Signal Projections
//===----------------------------------------------------------------------===//

def SigArraySliceOp : LLHDOp<"sig.array_slice", [
  Pure,
  SmallerOrEqualResultTypeWidthConstraint<"result", "input">,
  SameSigPtrArrayElementTypeConstraint<"result", "input">,
  SigPtrIndexBitWidthConstraint<"lowIndex", "input">,
]> {
  let summary = "Get a range of consecutive values from a signal of an array";
  let description = [{
    The `llhd.sig.array_slice` operation allows to access a sub-range of the
    `$input` operand, starting at the index given by the `$lowIndex`
    operand. The resulting slice length is defined by the result type.
    Returns a signal aliasing the elements of the slice.

    Width of 'lowIndex' is defined to be the precise number of bits required to
    index the 'input' array. More precisely: for an input array of size M,
    the width of 'lowIndex' is ceil(log2(M)). Lower and upper bound indexes
    which are larger than the size of the 'input' array results in undefined
    behavior.

    Example:

    ```mlir
    %3 = llhd.sig.array_slice %input at %lowIndex :
      (!llhd.ref<!hw.array<4xi8>>) -> !llhd.ref<!hw.array<2xi8>>
    ```
  }];

  let arguments = (ins RefTypeOf<ArrayType>: $input,
                       HWIntegerType: $lowIndex);
  let results = (outs RefTypeOf<ArrayType>: $result);

  let assemblyFormat = [{
    $input `at` $lowIndex attr-dict `:` type($input) `->` type($result)
  }];

  let extraClassDeclaration = [{
    unsigned getResultWidth() {
      return getLLHDTypeWidth(getResult().getType());
    }
    unsigned getInputWidth() {
      return getLLHDTypeWidth(getInput().getType());
    }
    hw::ArrayType getInputArrayType() {
      return cast<hw::ArrayType>(
        cast<RefType>(getInput().getType()).getNestedType()
      );
    }
    hw::ArrayType getResultArrayType() {
      return cast<hw::ArrayType>(
        cast<RefType>(getResult().getType()).getNestedType()
      );
    }
  }];
  let hasFolder = true;
  let hasCanonicalizeMethod = true;
}

def SigArrayGetOp : LLHDOp<"sig.array_get", [
  Pure,
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
  DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
  SigPtrIndexBitWidthConstraint<"index", "input">,
  SigArrayElementTypeConstraint<"result", "input">,
]> {
  let summary = "Extract an element from a signal of an array.";
  let description = [{
    The `llhd.sig.array_get` operation allows to access the element of the
    `$input` operand at position `$index`. A new subsignal aliasing the element
    will be returned.

    Example:

    ```mlir
    // Returns a !llhd.ref<i8>
    %0 = llhd.sig.array_get %arr[%index] : !llhd.ref<!hw.array<4xi8>>
    ```
  }];

  let arguments = (ins RefTypeOf<ArrayType>:$input, HWIntegerType:$index);
  let results = (outs RefType: $result);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input)
  }];

  let extraClassDeclaration = [{
    hw::ArrayType getArrayType() {
      return cast<hw::ArrayType>(
        cast<RefType>(getInput().getType()).getNestedType()
      );
    }
  }];
}

//===----------------------------------------------------------------------===//
// Struct Signal Projections
//===----------------------------------------------------------------------===//

def SigStructExtractOp : LLHDOp<"sig.struct_extract", [
  Pure,
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
  DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  InferTypeOpInterface,
]> {
  let summary = "Extract a field from a signal of a struct.";
  let description = [{
    The `llhd.sig.struct_extract` operation allows access to the field of the
    `$input` operand given by its name via the `$field` attribute.
    A new subsignal aliasing the field will be returned.

    Example:

    ```mlir
    // Returns a !llhd.ref<i8>
    %0 = llhd.sig.struct_extract %struct["foo"]
      : !llhd.ref<!hw.struct<foo: i8, bar: i16>>
    ```
  }];

  let arguments = (ins RefTypeOf<AnyTypeOf<[StructType, UnionType]>>:$input, StrAttr:$field);
  let results = (outs RefTypeOf<AnyType>:$result);

  let assemblyFormat = [{
    $input `[` $field `]` attr-dict `:` type($input)
  }];

  let extraClassDeclaration = [{
    hw::StructType getStructType() {
      return cast<hw::StructType>(
        cast<RefType>(getInput().getType()).getNestedType()
      );
    }
  }];
}

//===----------------------------------------------------------------------===//
// Processes
//===----------------------------------------------------------------------===//

def ProcessOp : LLHDOp<"process", [
  HasParent<"hw::HWModuleOp">,
  NoRegionArguments,
  ProceduralRegion,
  RecursiveMemoryEffects,
]> {
  let summary = "A process that runs concurrently during simulation";
  let description = [{
    An `llhd.process` op encapsulates a region of IR that executes concurrently
    during simulation. Execution can be suspended using the `llhd.wait`
    terminator, which also includes a list of values that will cause the process
    execution to resume whenever they change. The `llhd.halt` terminator can be
    used to suspend execution forever. The process may have results, in which
    case any `llhd.wait` or `llhd.halt` terminators must provide a list of
    values to yield for the process results whenever execution is suspended. The
    process holds these result values until it is resumed and new result values
    are yielded.

    Example:
    ```mlir
    hw.module @top() {
      %0, %1 = llhd.process -> i42, i9001 {
        llhd.wait yield (%2, %3 : i42, i9001), ^bb1
      ^bb1:
        llhd.halt %4, %5 : i42, i9001
      }
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>: $body);
  let assemblyFormat = [{
    (`->` type($results)^)?
    attr-dict-with-keyword $body
  }];
  let hasCanonicalizeMethod = 1;
}

def FinalOp : LLHDOp<"final", [
  HasParent<"hw::HWModuleOp">,
  NoRegionArguments,
  ProceduralRegion,
  RecursiveMemoryEffects,
]> {
  let summary = "A process that runs at the end of simulation";
  let description = [{
    An `llhd.final` op encapsulates a region of IR that is to be executed after
    the last time step of a simulation has completed. This can be used to
    implement various forms of state cleanup and tear-down. Some verifications
    ops may also want to check that certain final conditions hold at the end of
    a simulation run.

    The `llhd.wait` terminator is not allowed in `llhd.final` processes since
    there is no later time slot for the execution to resume. Control flow must
    eventually end in an `llhd.halt` terminator.

    Execution order between multiple `llhd.final` ops is undefined.

    Example:
    ```mlir
    hw.module @Foo() {
      llhd.final {
        func.call @printSimulationStatistics() : () -> ()
        llhd.halt
      }
    }
    ```
  }];
  let regions = (region MinSizedRegion<1>: $body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def CombinationalOp : LLHDOp<"combinational", [
  HasParent<"hw::HWModuleOp">,
  NoRegionArguments,
  ProceduralRegion,
  RecursiveMemoryEffects,
]> {
  let summary = "A process that runs when any of its operand values change";
  let description = [{
    An `llhd.combinational` op encapsulates a region of IR that executes once at
    the beginning of the simulation, and subsequently whenever any of the values
    used in its body change. Control flow must eventually end in an `llhd.yield`
    terminator. The process may have results, in which case the `llhd.yield`
    terminators must provide a list of values to yield for the process results.
    Whenever any of the values used in the body change, the process reexecutes
    in order to compute updated results.

    This op is commonly used to embed a control-flow description of some
    combinational logic inside the surrounding module's graph region.

    Example:
    ```mlir
    hw.module @Foo() {
      %0, %1 = llhd.combinational -> i42, i9001 {
        cf.cond_br %2, ^bb1(%3, %4 : i42, i9001), ^bb1(%5, %6 : i42, i9001)
      ^bb1(%7: i42, %8: i9001):
        llhd.yield %7, %8 : i42, i9001
      }
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    (`->` type($results)^)?
    attr-dict-with-keyword $body
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Terminators
//===----------------------------------------------------------------------===//

def WaitOp : LLHDOp<"wait", [
  AttrSizedOperandSegments,
  HasParent<"ProcessOp">,
  Terminator,
]> {
  let summary = "Suspend execution of a process";
  let description = [{
    The `llhd.wait` terminator suspends execution of the parent process until
    any of the `observed` values change or a fixed `delay` has passed. Execution
    resumes at the `dest` block with the `destOperands` arguments. The
    `yieldOperands` are yielded as the result values of the parent process.

    Example:
    ```mlir
    llhd.wait ^bb1(%0, %1 : i42, i9001)
    llhd.wait yield (%0, %1 : i42, i9001), ^bb1
    llhd.wait delay %time, ^bb1
    llhd.wait (%0, %1 : i42, i9001), ^bb1
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$yieldOperands,
    Optional<LLHDTimeType>:$delay,
    Variadic<HWValueType>:$observed,
    Variadic<AnyType>:$destOperands
  );
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    (`yield` ` ` `(` $yieldOperands^ `:` type($yieldOperands) `)` `,`)?
    (`delay` $delay^ `,`)?
    (`(`$observed^ `:` qualified(type($observed))`)` `,`)?
    $dest (`(` $destOperands^ `:` qualified(type($destOperands)) `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

def HaltOp : LLHDOp<"halt", [
  ParentOneOf<["ProcessOp", "FinalOp"]>,
  Terminator,
]> {
  let summary = "Terminate execution of a process";
  let description = [{
    The `llhd.halt` terminator suspends execution of the parent process forever,
    effectively terminating it. The `yieldOperands` are yielded as the result
    values of the parent process.

    Example:
    ```mlir
    llhd.halt
    llhd.halt %0, %1 : i42, i9001
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$yieldOperands
  );
  let assemblyFormat = [{
    ($yieldOperands^ `:` type($yieldOperands))?
    attr-dict
  }];
  let hasVerifier = 1;
}

def YieldOp : LLHDOp<"yield", [
  ParentOneOf<["CombinationalOp", "GlobalSignalOp"]>,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Yield results back from a combinational process";
  let description = [{
    The `llhd.yield` terminator terminates control flow in the parent process
    and yields the `yieldOperands` as the result values of the process.

    Example:
    ```mlir
    llhd.combinational {
      llhd.yield
    }
    %2:2 = llhd.combinational -> i42, i9001 {
      llhd.yield %0, %1 : i42, i9001
    }
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$yieldOperands);
  let assemblyFormat = [{
    ($yieldOperands^ `:` type($yieldOperands))?
    attr-dict
  }];
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_LLHD_LLHDOPS_TD
