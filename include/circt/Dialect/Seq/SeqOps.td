//===- Seq.td - Seq dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Seq dialect. It contains the one op and
// pass. Once we add more than one, we should break it out like the other
// dialects.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Sim/SimTypes.td"
include "circt/Dialect/Seq/SeqAttributes.td"
include "circt/Dialect/Seq/SeqOpInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Base class for the operation in this dialect.
class SeqOp<string mnemonic, list<Trait> traits = []> :
    Op<SeqDialect, mnemonic, traits>;

def CompRegOp : SeqOp<"compreg",
    [Pure, Clocked, Resettable,
     AllTypesMatch<["input", "data"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>,
     AttrSizedOperandSegments]> {
       // AllTypesMatch doesn't work with Optional types yet.

  let summary = "Register a value, storing it for one cycle";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins
    AnyType:$input,
    ClockType:$clk,
    OptionalAttr<StrAttr>:$name,
    Optional<I1>:$reset,
    Optional<AnyType>:$resetValue,
    Optional<ImmutableType>:$initialValue,
    OptionalAttr<InnerSymAttr>:$inner_sym
  );
  let results = (outs AnyType:$data);
  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) $input `,` $clk
    (`reset` $reset^ `,` $resetValue)?
    (`initial` $initialValue^)? attr-dict `:` type($data)
    custom<OptionalTypeMatch>(ref(type($data)), ref($resetValue), type($resetValue))
    custom<OptionalImmutableTypeMatch>(ref(type($data)), ref($initialValue), type($initialValue))
  }];
  let hasVerifier = 1;

  let builders = [
    /// Create a register with no name nor inner_sym.
    OpBuilder<(ins "Value":$input, "Value":$clk), [{
      return build($_builder, $_state, input.getType(), input, clk,
                   /*name*/ StringAttr(),
                   /*reset*/ Value(), /*resetValue*/ Value(),
                    /*initialValue*/ Value(),
                   hw::InnerSymAttr());
    }]>,
    /// Create a register with an inner_sym matching the register's name.
    OpBuilder<(ins "Value":$input, "Value":$clk, "StringAttrOrRef":$name), [{
      auto nameAttr = name.get($_builder.getContext());
      return build($_builder, $_state, input.getType(), input, clk, nameAttr,
                   /*reset*/ Value(), /*resetValue*/ Value(),
                   /*initialValue*/ Value(),
                   hw::InnerSymAttr::get(nameAttr));
    }]>,
    /// Create a register with a reset and optional power-on value.
    OpBuilder<(ins "Value":$input, "Value":$clk, "Value":$reset, "Value":$rstValue,
                   CArg<"Value", "{}">:$initialValue), [{
      return build($_builder, $_state, input.getType(), input, clk, StringAttr(),
                   reset, rstValue, initialValue, hw::InnerSymAttr());
    }]>,
    /// Create a register with a reset, with an inner_sym matching the
    /// register's name, and optional power-on value.
    OpBuilder<(ins "Value":$input, "Value":$clk, "Value":$reset, "Value":$rstValue,
                   "StringAttrOrRef":$name, CArg<"Value", "{}">:$initialValue), [{
      auto nameAttr = name.get($_builder.getContext());
      return build($_builder, $_state, input.getType(), input, clk, nameAttr,
                   reset, rstValue, initialValue, hw::InnerSymAttr::get(nameAttr));
    }]>,
  ];
}

def CompRegClockEnabledOp : SeqOp<"compreg.ce",
    [Pure, Clocked, Resettable,
     AllTypesMatch<["input", "data"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>,
     AttrSizedOperandSegments]> {
       // AllTypesMatch doesn't work with Optional types yet.

  let summary = "When enabled, register a value";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins
    AnyType:$input,
    ClockType:$clk,
    I1:$clockEnable,
    OptionalAttr<StrAttr>:$name,
    Optional<I1>:$reset,
    Optional<AnyType>:$resetValue,
    Optional<ImmutableType>:$initialValue,
    OptionalAttr<InnerSymAttr>:$inner_sym
  );
  let results = (outs AnyType:$data);
  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) $input `,` $clk `,` $clockEnable
    (`reset` $reset^ `,` $resetValue)?
    (`initial` $initialValue^)? attr-dict `:` type($data)
    custom<OptionalTypeMatch>(ref(type($data)), ref($resetValue), type($resetValue))
    custom<OptionalImmutableTypeMatch>(ref(type($data)), ref($initialValue), type($initialValue))
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$clk, "Value":$ce,
                   "StringRef":$name), [{
      auto nameAttr = StringAttr::get($_builder.getContext(), name);
      return build($_builder, $_state, input.getType(), input, clk, ce,
                   nameAttr, /*reset*/ Value(), /*resetValue*/ Value(),
                   /*powerOnValue*/ Value(),
                   hw::InnerSymAttr::get(nameAttr));
    }]>,
    OpBuilder<(ins "Value":$input, "Value":$clk, "Value":$ce,
                   "Value":$reset, "Value":$rstValue, "StringRef":$name,
                   CArg<"Value", "{}">:$initialValue),
    [{
      auto nameAttr = StringAttr::get($_builder.getContext(), name);
      return build($_builder, $_state, input.getType(), input, clk, ce,
                   nameAttr, reset, rstValue, initialValue,
                   hw::InnerSymAttr::get(nameAttr));
    }]>,
  ];
}

def ShiftRegOp : SeqOp<"shiftreg", [
  Pure,
  Clocked,
  Resettable,
  AllTypesMatch<["input", "data"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>,
  AttrSizedOperandSegments
]> {
  let summary = "Shift register";
  let description = [{
    The `seq.shiftreg` op represents a shift register. It takes the input
    value and shifts it every cycle when `clockEnable` is asserted.
    The `reset` and `resetValue` operands are optional and if present, every
    entry in the shift register will be initialized to `resetValue` upon
    assertion of the reset signal. Exact reset behavior (sync/async) is
    implementation defined.
  }];

  let arguments = (ins
    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$numElements,
    AnyType:$input,
    ClockType:$clk,
    I1:$clockEnable,
    OptionalAttr<StrAttr>:$name,
    Optional<I1>:$reset,
    Optional<AnyType>:$resetValue,
    Optional<AnyType>:$powerOnValue,
    OptionalAttr<InnerSymAttr>:$inner_sym
  );
  let results = (outs AnyType:$data);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `[` $numElements `]`
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) $input `,` $clk `,` $clockEnable
    (`reset` $reset^ `,` $resetValue)?
    (`powerOn` $powerOnValue^)? attr-dict `:` type($data)
    custom<OptionalTypeMatch>(ref(type($data)), ref($resetValue), type($resetValue))
    custom<OptionalTypeMatch>(ref(type($data)), ref($powerOnValue), type($powerOnValue))
  }];
}

//===----------------------------------------------------------------------===//
// FIRRTL-flavored register
//===----------------------------------------------------------------------===//

def FirRegOp : SeqOp<"firreg",
    [Pure, Clocked, Resettable,
     AllTypesMatch<["next", "data"/*, "resetValue"*/]>,
     SameVariadicOperandSize, MemoryEffects<[MemWrite, MemRead, MemAlloc]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>]> {
      // AllTypesMatch doesn't work with Optional types yet.

  let summary = "Register with preset and sync or async reset";
  let description = [{
    `firreg` represents registers originating from FIRRTL after the lowering
    of the IR to HW.  The register is used as an intermediary in the process
    of lowering to SystemVerilog to facilitate optimisation at the HW level,
    compactly representing a register with a single operation instead of
    composing it from register definitions, always blocks and if statements.

    The `data` output of the register accesses the value it stores.  On the
    rising edge of the `clk` input, the register takes a new value provided
    by the `next` signal.  Optionally, the register can also be provided with
    a synchronous or an asynchronous `reset` signal and `resetValue`, as shown
    in the example below.

    ```
    %name = seq.firreg %next clock %clk [ sym @sym ]
        [ reset (sync|async) %reset, %value ]
        [ preset value ] : type
    ```

    Implicitly, all registers are pre-set to a randomized value.

    A register implementing a counter starting at 0 from reset can be defined
    as follows:

    ```
    %zero = hw.constant 0 : i32
    %reg = seq.firreg %next clock %clk reset sync %reset, %zero : i32
    %one = hw.constant 1 : i32
    %next = comb.add %reg, %one : i32
    ```
  }];

  let arguments = (ins
    AnyType:$next,
    ClockType:$clk,
    StrAttr:$name,
    OptionalAttr<InnerSymAttr>:$inner_sym,
    OptionalAttr<APIntAttr>:$preset,
    Optional<I1>:$reset, Optional<AnyType>:$resetValue,
    UnitAttr:$isAsync
  );
  let results = (outs AnyType:$data);

  let hasCanonicalizeMethod = true;
  let hasFolder = true;
  let builders = [
    OpBuilder<(ins "Value":$next, "Value":$clk,
                   "StringAttr":$name,
                   CArg<"hw::InnerSymAttr", "{}">:$innerSym,
                   CArg<"Attribute","{}">:$preset)>,
    OpBuilder<(ins "Value":$next, "Value":$clk,
                   "StringAttr":$name,
                   "Value":$reset, "Value":$resetValue,
                   CArg<"hw::InnerSymAttr", "{}">:$innerSym,
                   CArg<"bool", "false">:$isAsync)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Check whether the register has a reset value.
    bool hasReset() { return !!getReset(); }
    /// Check whether the register has a preset value.
    bool hasPresetValue() { return !!getPresetAttr(); }
  }];
}

def FIFOOp : SeqOp<"fifo", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    TypesMatchWith<"Input type should match output type",
      "input", "output", [{ $_self }]>,
    AttrSizedResultSegments
  ]> {
  let summary = "A high-level FIFO operation";
  let description = [{
    This operation represents a high-level abstraction of a FIFO. Access to the
    FIFO is structural, and thus may be composed with other core RTL dialect
    operations.
    The fifo operation is configurable with the following parameters:
    1. Depth (cycles)
    2. Read latency (cycles) is the number of cycles it takes for a read to
       return data after the read enable signal is asserted.
    3. Almost full/empty thresholds (optional). If not provided, these will
       be asserted when the FIFO is full/empty.

    Like `seq.hlmem` there are no guarantees that all possible fifo configuration
    are able to be lowered. Available lowering passes will pattern match on the
    requested fifo configuration and attempt to provide a legal lowering.
  }];

  let arguments = (ins
    AnyType:$input, I1:$rdEn, I1:$wrEn, ClockType:$clk, I1:$rst,
    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$depth,
    DefaultValuedOptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>, "0">
      :$rdLatency,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>:$almostFullThreshold,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>:$almostEmptyThreshold
  );

  let results = (outs
    AnyType:$output, I1:$full, I1:$empty, Optional<I1>:$almostFull,
    Optional<I1>:$almostEmpty);

  let assemblyFormat = [{
    `depth` $depth
    (`rd_latency` $rdLatency^)?
    custom<FIFOAFThreshold>($almostFullThreshold, type($almostFull))
    custom<FIFOAEThreshold>($almostEmptyThreshold, type($almostEmpty))
    `in` $input `rdEn` $rdEn `wrEn` $wrEn `clk` $clk `rst` $rst attr-dict `:` type($input)
  }];
  let hasVerifier = 1;
}

def HLMemOp : SeqOp<"hlmem", [
     Clocked,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]> ]> {

  let summary = "Instantiate a high-level memory.";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins ClockType:$clk, I1:$rst, SymbolNameAttr:$name);
  let results = (outs HLMemType:$handle);

  let extraClassDeclaration = [{
    HLMemType getMemType() {
      return llvm::cast<HLMemType>(getHandle().getType());
    }
  }];

  let builders = [
    OpBuilder<(ins "Value":$clk, "Value":$rst, "StringRef":$name,
                   "llvm::ArrayRef<int64_t>":$shape, "Type":$elementType)>
  ];

  let assemblyFormat = "$name $clk `,` $rst attr-dict `:` type($handle)";
}

class HLMemTypeValueConstraint<string hlmemvalue, string value>
  : TypesMatchWith<"type of 'value' matches element type of 'hlmemvalue'",
                     hlmemvalue, value,
                     "llvm::cast<HLMemType>($_self).getElementType()">;

class HLMemTypeIndexingConstraint<string hlmemvalue, string addresses>
  : PredOpTrait<"address bit widths should be identical to the clog2 of the shape of the HLMemType",
      CPred<"isValidIndexValues($" # hlmemvalue # ", $" # addresses # ")">>;

def ReadPortOp : SeqOp<"read", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    HLMemTypeIndexingConstraint<"memory", "addresses">,
    AttrSizedOperandSegments
  ]> {
  let summary = "Structural read access to a seq.hlmem, with an optional read enable signal.";
  let arguments = (ins
    HLMemType:$memory,
    Variadic<HWIntegerType>:$addresses,
    Optional<I1>:$rdEn,
    I64Attr:$latency
  );
  let results = (outs AnyType:$readData);
  let hasCustomAssemblyFormat = 1;
  let builders = [OpBuilder<(
    ins "Value":$memory, "ValueRange":$addresses, "Value":$rdEn, "unsigned":$latency)>
  ];
}

def WritePortOp : SeqOp<"write", [
    HLMemTypeValueConstraint<"memory", "inData">,
    HLMemTypeIndexingConstraint<"memory", "addresses">
  ]> {
  let summary = "Structural write access to a seq.hlmem";
  let arguments = (ins
    HLMemType:$memory,
    Variadic<HWIntegerType>:$addresses,
    AnyType:$inData,
    I1:$wrEn,
    I64Attr:$latency
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Clock Gate
//===----------------------------------------------------------------------===//

def ClockGateOp : SeqOp<"clock_gate", [
  Pure,
  DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>,
  AllTypesMatch<["input", "output"]>
]> {
  let summary = "Safely gates a clock with an enable signal";
  let description = [{
    The `seq.clock_gate` enables and disables a clock safely, without glitches,
    based on a boolean enable value. If the enable operand is 1, the output
    clock produced by the clock gate is identical to the input clock. If the
    enable operand is 0, the output clock is a constant zero.

    The `enable` operand is sampled at the rising edge of the input clock; any
    changes on the enable before or after that edge are ignored and do not
    affect the output clock.

    The `test_enable` operand is optional and if present is OR'd together with
    the `enable` operand to determine whether the output clock is gated or not.

    The op can be referred to using an inner symbol. Upon translation, the
    symbol will target the instance to the external module it lowers to.

    ```
    %gatedClock = seq.clock_gate %clock, %enable
    %gatedClock = seq.clock_gate %clock, %enable, %test_enable
    ```
  }];

  let arguments = (ins
    ClockType:$input,
    I1:$enable,
    Optional<I1>:$test_enable,
    OptionalAttr<InnerSymAttr>:$inner_sym
  );

  let results = (outs ClockType:$output);

  let builders = [
    OpBuilder<(ins "Value":$clock, "Value":$enable), [{
      build($_builder, $_state, clock, enable, Value(), hw::InnerSymAttr());
    }]>,
  ];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $input `,` $enable (`,` $test_enable^)? (`sym` $inner_sym^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Clock Multiplexer
//===----------------------------------------------------------------------===//

def ClockMuxOp : SeqOp<"clock_mux", [Pure]> {
  let summary = "Safely selects a clock based on a condition";
  let description = [{
    The `seq.clock_mux` op selects a clock from two options. If `cond` is
    true, the first clock operand is selected to drive downstream logic.
    Otherwise, the second clock is used.

    ```
    %clock = seq.clock_mux %cond, %trueClock, %falseClock
    ```
  }];

  let arguments = (ins I1:$cond, ClockType:$trueClock, ClockType:$falseClock);
  let results = (outs ClockType:$result);

  let hasFolder = 1;

  let assemblyFormat = [{
    $cond `,` $trueClock `,` $falseClock attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Clock Dividers
//===----------------------------------------------------------------------===//

def ClockDividerOp : SeqOp<"clock_div", [Pure]> {
  let summary = "Produces a clock divided by a power of two";
  let description = [{
    The output clock is phase-aligned to the input clock.

    ```
    %div_clock = seq.clock_div %clock by 1
    ```
  }];

  let arguments = (ins ClockType:$input, I64Attr:$pow2);
  let results = (outs ClockType:$output);

  let assemblyFormat = [{
    $input `by` $pow2 attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Clock Inverter and Buffer
//===----------------------------------------------------------------------===//

def ClockInverterOp : SeqOp<"clock_inv", [Pure]> {
  let summary = "Inverts the clock signal";
  let description = [{
    Note that the compiler can optimize inverters away, preventing their
    use as part of explicit clock buffers.

    ```
    %inv_clock = seq.clock_inv %clock
    ```
  }];

  let arguments = (ins ClockType:$input);
  let results = (outs ClockType:$output);

  let hasFolder = 1;

  let assemblyFormat = [{
    $input attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// FIRRTL-flavored memory
//===----------------------------------------------------------------------===//

def FirMemOp : SeqOp<"firmem", [
  MemoryEffects<[MemAlloc]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>
]> {
  let summary = "A FIRRTL-flavored memory";
  let description = [{
    The `seq.firmem` op represents memories lowered from the FIRRTL dialect. It
    is used to capture some of the peculiarities of what FIRRTL expects from
    memories, while still representing them at the HW dialect level.

    A `seq.firmem` declares the memory and captures the memory-level parameters
    such as width and depth or how read/write collisions are resolved. The read,
    write, and read-write ports are expressed as separate operations that take
    the declared memory as an operand.
  }];

  let arguments = (ins
    I32Attr:$readLatency,
    I32Attr:$writeLatency,
    RUWAttr:$ruw,
    WUWAttr:$wuw,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<InnerSymAttr>:$inner_sym,
    OptionalAttr<FirMemInitAttr>:$init,
    OptionalAttr<StrAttr>:$prefix,
    OptionalAttr<AnyAttr>:$output_file
  );
  let results = (outs FirMemType:$memory);

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name)
    $readLatency `,` $writeLatency `,` $ruw `,` $wuw
    attr-dict `:` type($memory)
  }];

  let hasCanonicalizeMethod = true;
}

class AddressMatchesFirMem<string memoryValue, string addressValue> :
  TypesMatchWith<"address type should match clog2 of memory depth",
    memoryValue, addressValue, [{
      IntegerType::get(
        $_self.getContext(),
        std::max(1U, llvm::Log2_64_Ceil(cast<FirMemType>($_self).getDepth())))
    }]>;

class DataMatchesFirMem<string memoryValue, string dataValue> :
  TypesMatchWith<"data type should match memory width",
    memoryValue, dataValue, [{
      IntegerType::get(
        $_self.getContext(),
        std::max(1U, cast<FirMemType>($_self).getWidth()))
    }]>;

def FirMemReadOp : SeqOp<"firmem.read_port", [
  Clocked,
  MemoryEffects<[MemRead]>,
  AddressMatchesFirMem<"memory", "address">,
  DataMatchesFirMem<"memory", "data">
]> {
  let summary = "A memory read port";
  let description = [{
    The `seq.firmem.read_port` op represents a read port on a `seq.firmem`
    memory. It takes the memory as an operand, together with the address to
    be read, the clock on which the read is synchronized, and an optional
    enable. Omitting the enable operand has the same effect as passing a
    constant `true` to it.
  }];

  let arguments = (ins
    FirMemType:$memory,
    AnySignlessInteger:$address,
    ClockType:$clk,
    Optional<I1>:$enable
  );
  let results = (outs AnySignlessInteger:$data);
  let assemblyFormat = [{
    $memory `[` $address `]` `,` `clock` $clk
    (`enable` $enable^)?
    attr-dict `:` type($memory)
  }];
  let hasCanonicalizeMethod = 1;
}

def FirMemWriteOp : SeqOp<"firmem.write_port", [
  Clocked,
  MemoryEffects<[MemWrite]>,
  AddressMatchesFirMem<"memory", "address">,
  DataMatchesFirMem<"memory", "data">,
  AttrSizedOperandSegments
]> {
  let summary = "A memory write port";
  let description = [{
    The `seq.firmem.write_port` op represents a write port on a `seq.firmem`
    memory. It takes the memory as an operand, together with the address and
    data to be written, the clock on which the write is synchronized, an
    optional enable, and and optional write mask. Omitting the enable operand
    has the same effect as passing a constant `true` to it. Omitting the write
    mask operand has the same effect as passing an all-ones value to it. A write
    mask operand can only be present if the `seq.firmem` specifies a mask width;
    otherwise it must be omitted.
  }];

  let arguments = (ins
    FirMemType:$memory,
    AnySignlessInteger:$address,
    ClockType:$clk,
    Optional<I1>:$enable,
    AnySignlessInteger:$data,
    Optional<AnySignlessInteger>:$mask
  );
  let assemblyFormat = [{
    $memory `[` $address `]` `=` $data `,` `clock` $clk
    (`enable` $enable^)? (`mask` $mask^)?
    attr-dict `:` type($memory) (`,` type($mask)^)?
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def FirMemReadWriteOp : SeqOp<"firmem.read_write_port", [
  Clocked,
  MemoryEffects<[MemRead, MemWrite]>,
  AddressMatchesFirMem<"memory", "address">,
  DataMatchesFirMem<"memory", "writeData">,
  DataMatchesFirMem<"memory", "readData">,
  AttrSizedOperandSegments
]> {
  let summary = "A memory read-write port";
  let description = [{
    The `seq.firmem.read_write_port` op represents a read-write port on a
    `seq.firmem` memory. It takes the memory as an operand, together with the
    address and data to be written, a mode operand indicating whether the port
    should perform a read (`mode=0`) or a write (`mode=1`), the clock on which
    the read and write is synchronized, an optional enable, and and optional
    write mask. Omitting the enable operand has the same effect as passing a
    constant `true` to it. Omitting the write mask operand has the same effect
    as passing an all-ones value to it. A write mask operand can only be present
    if the `seq.firmem` specifies a mask width; otherwise it must be omitted.
  }];

  let arguments = (ins
    FirMemType:$memory,
    AnySignlessInteger:$address,
    ClockType:$clk,
    Optional<I1>:$enable,
    AnySignlessInteger:$writeData,
    I1:$mode,
    Optional<AnySignlessInteger>:$mask
  );
  let results = (outs AnySignlessInteger:$readData);
  let assemblyFormat = [{
    $memory `[` $address `]` `=` $writeData `if` $mode `,` `clock` $clk
    (`enable` $enable^)? (`mask` $mask^)?
    attr-dict `:` type($memory) (`,` type($mask)^)?
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Tied-off clock
//===----------------------------------------------------------------------===//

def ConstClockOp : SeqOp<"const_clock", [Pure, ConstantLike]> {
  let summary = "Produce constant clock value";
  let description = [{
    The constant operation produces a constant clock value.
    ```
      %clock = seq.const_clock low
    ```
  }];

  let arguments = (ins ClockConstAttr:$value);
  let results = (outs ClockType:$result);

  let assemblyFormat = "$value attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Clock cast
//===----------------------------------------------------------------------===//

def ToClockOp : SeqOp<"to_clock", [Pure]> {
  let summary = "Cast from a wire type to a clock type";

  let arguments = (ins I1:$input);
  let results = (outs ClockType:$output);

  let assemblyFormat = "$input attr-dict";

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def FromClockOp : SeqOp<"from_clock", [Pure]> {
  let summary = "Cast from a clock type to a wire type";

  let arguments = (ins ClockType:$input);
  let results = (outs I1:$output);

  let assemblyFormat = "$input attr-dict";

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def InitialOp : SeqOp<"initial", [SingleBlock,
      SingleBlockImplicitTerminator<"YieldOp">,
      RecursivelySpeculatable,
      RecursiveMemoryEffects, IsolatedFromAbove]> {
  let summary = "Operation that produces values for initialization";
  let description = [{
    `seq.initial` op creates values wrapped types with !seq.immutable.
    See the Seq dialect rationale for a longer description.
  }];

  let arguments = (ins Variadic<ImmutableType>: $inputs);
  let results = (outs Variadic<ImmutableType>); // seq.immutable values
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"TypeRange", "{}">:$resultTypes, CArg<"std::function<void()>", "{}">:$ctor)>
  ];

  let assemblyFormat = [{
    `(` $inputs `)` $body attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def YieldOp : SeqOp<"yield",
                   [Pure, Terminator, HasParent<"InitialOp">]> {
  let summary = "Yield values";

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins), "build($_builder, $_state, mlir::ValueRange());">
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def FromImmutableOp : SeqOp<"from_immutable", [Pure]> {
  let summary = "Cast from an immutable type to a wire type";

  let arguments = (ins ImmutableType:$input);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
}