//===- FIRRTLIntrinsics.td - FIRRTL intrinsic ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for FIRRTL intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_TD

include "FIRRTLAttributes.td"
include "FIRRTLDialect.td"
include "FIRRTLTypes.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Generic intrinsic operation for parsing into before lowering.
//===----------------------------------------------------------------------===//

def GenericIntrinsicOp : FIRRTLOp<"int.generic",
    [HasCustomSSAName]> {
  let summary = "Generic intrinsic operation for FIRRTL intrinsics.";

  let arguments = (
    ins StrAttr:$intrinsic,
    Variadic<PassiveType>:$operands,
    DefaultValuedAttr<ParamDeclArrayAttr, "{}">:$parameters
  );

  let results = (outs Optional<PassiveType>:$result);
  let assemblyFormat = "$intrinsic custom<ParameterList>($parameters) ($operands^)? attr-dict-with-keyword `:` functional-type($operands, $result)";
}

//===----------------------------------------------------------------------===//
// Plusarg Intrinsics
//===----------------------------------------------------------------------===//

def PlusArgsTestIntrinsicOp : FIRRTLOp<"int.plusargs.test",
    [HasCustomSSAName, Pure]> {
  let summary = "SystemVerilog `$test$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs NonConstUInt1Type:$found);
  let assemblyFormat = "$formatString attr-dict";
}

def PlusArgsValueIntrinsicOp : FIRRTLOp<"int.plusargs.value",
    [HasCustomSSAName, Pure]> {
  let summary = "SystemVerilog `$value$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs NonConstUInt1Type:$found, AnyType:$result);
  let assemblyFormat = "$formatString attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// FPGA-specific intrinsics
//===----------------------------------------------------------------------===//

def FPGAProbeIntrinsicOp : FIRRTLOp<"int.fpga_probe", []> {
  let summary = "Mark a value to be observed through FPGA debugging facilities";

  let description = [{
    The `firrtl.int.fpga_probe` intrinsic marks a value in
    the IR to be made observable through FPGA debugging facilities. Most FPGAs
    offer a form of signal observation or logic analyzer to debug a design. This
    operation allows the IR to indicate which signals should be made observable
    for debugging. Later FPGA-specific passes may then pick this information up
    and materialize the necessary logic analyzers or tool scripts.
  }];

  let arguments = (ins AnyType:$input, NonConstClockType:$clock);
  let results = (outs);
  let assemblyFormat = "$clock `,` $input attr-dict `:` type($input)";

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Clock Intrinsics
//===----------------------------------------------------------------------===//

def ClockGateIntrinsicOp : FIRRTLOp<"int.clock_gate", [Pure]> {
  let summary = "Safely gates a clock with an enable signal";
  let description = [{
    The `int.clock_gate` enables and disables a clock safely, without glitches,
    based on a boolean enable value. If the enable input is 1, the output clock
    produced by the clock gate is identical to the input clock. If the enable
    input is 0, the output clock is a constant zero.

    The enable input is sampled at the rising edge of the input clock; any
    changes on the enable before or after that edge are ignored and do not
    affect the output clock.
  }];

  let arguments = (ins NonConstClockType:$input,
                       NonConstUInt1Type:$enable,
                       Optional<NonConstUInt1Type>:$test_enable);
  let results = (outs NonConstClockType:$output);
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $input `,` $enable (`,` $test_enable^)? attr-dict
  }];
}

def ClockInverterIntrinsicOp : FIRRTLOp<"int.clock_inv", [Pure]> {
  let summary = "Inverts the clock signal";

  let description = [{
    The `firrtl.int.clock.inv` intrinsic takes a clock signal and inverts it.
    It can be used to build registers and other operations which are triggered
    by a negative clock edge relative to a reference signal. The compiler is
    free to optimize inverters (particularly double inverters).

    See the corresponding `seq.clock_inv` operation.
  }];

  let arguments = (ins NonConstClockType:$input);
  let results = (outs NonConstClockType:$output);
  let assemblyFormat = "$input attr-dict";
}

def ClockDividerIntrinsicOp : FIRRTLOp<"int.clock_div", [Pure]> {
  let summary = "Produces a clock divided by a power of two";
  let description = [{
    The `firrtl.int.clock_div` takes a clock signal and divides it by a
    power-of-two ratio. The output clock is phase-aligned to the input clock.

    ```
    %div_clock = seq.clock_div %clock by 1
    ```
  }];

  let arguments = (ins NonConstClockType:$input, I64Attr:$pow2);
  let results = (outs NonConstClockType:$output);
  let assemblyFormat = "$input `by` $pow2 attr-dict";
}

//===----------------------------------------------------------------------===//
// Verification Intrinsics
//===----------------------------------------------------------------------===//

def UnclockedAssumeIntrinsicOp : FIRRTLOp<"int.unclocked_assume", []> {
  let summary = "Special Assume Verification Statement to assume predicate ";

  let description = [{
    The `firrtl.int.unclocked_assume` intrinsic is a special assume statement
    lowered into a SV concurrent assertion within always block that has the assumed
    predicate in a sensitivity list.
  }];
  let arguments = (ins
    UInt1Type:$predicate,
    UInt1Type:$enable,
    StrAttr:$message,
    Variadic<AnyType>:$substitutions,
    StrAttr:$name
  );

  let assemblyFormat = [{
    $predicate `,` $enable `,` $message (`(` $substitutions^ `)`)? `:`
    type($predicate) `,` type($enable) (`,` qualified(type($substitutions))^)?
    custom<VerifAttrs>(attr-dict)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Other intrinsics
//===----------------------------------------------------------------------===//

def IsXIntrinsicOp : FIRRTLOp<"int.isX",
    [HasCustomSSAName, Pure]> {
  let summary = "Test for 'x";
  let description = [{
    The `int.isX` expression checks that the operand is not a verilog literal
    'x.  FIRRTL doesn't have a notion of 'x per-se, but x can come in to the
    system from external modules and from SV constructs.  Verification
    constructs need to explicitly test for 'x.
    }];

  let arguments = (ins FIRRTLBaseType:$arg);
  let results = (outs NonConstUInt1Type:$result);
  let hasFolder = 1;
  let assemblyFormat = "$arg attr-dict `:` type($arg)";
}

def HasBeenResetIntrinsicOp : FIRRTLOp<"int.has_been_reset", [Pure]> {
  let summary = "Check that a proper reset has been seen.";
  let description = [{
    The result of `firrtl.int.has_been_reset` reads as 0 immediately after simulation
    startup and after each power-cycle in a power-aware simulation. The result
    remains 0 before and during reset and only switches to 1 after the reset is
    deasserted again.

    See the corresponding `verif.has_been_reset` operation.
  }];
  let arguments = (ins NonConstClockType:$clock, AnyResetType:$reset);
  let results = (outs NonConstUInt1Type:$result);
  let hasFolder = 1;
  let assemblyFormat = "$clock `,` $reset attr-dict `:` type($reset)";
}


def DPICallIntrinsicOp : FIRRTLOp<"int.dpi.call",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<CombDataflow>]> {
  let summary = "Import and call DPI function";
  let description = [{
    The `int.dpi.call` intrinsic calls an external function.
    See Sim dialect DPI call op.
    }];

  let arguments = (ins StrAttr:$functionName,
                       OptionalAttr<StrArrayAttr>:$inputNames,
                       OptionalAttr<StrAttr>:$outputName,
                       Optional<NonConstClockType>:$clock,
                       Optional<NonConstUInt1Type>:$enable,
                       Variadic<PassiveType>:$inputs);
  let results = (outs Optional<PassiveType>:$result);
  let assemblyFormat = [{
    $functionName `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
    attr-dict `:` functional-type($inputs, results)
  }];
  let hasVerifier = 1;
}

def ViewIntrinsicOp : FIRRTLOp<"view", []> {
  let summary = "A SystemVerilog Interface only usable for waveform debugging";
  let description = [{
    This will become a SystemVerilog Interface that is driven by its arguments.
    This is _not_ intended to be used for anything other than assistance when
    debugging in a waveform.  This is _not_ a true SystemVerilog Interface, it
    is only lowered to one.
  }];
  let arguments = (ins StrAttr:$name, AugmentedBundleType:$augmentedType, Variadic<FIRRTLBaseType>:$inputs);
  let assemblyFormat = "$name `,` $augmentedType (`,` $inputs^)? attr-dict `:` type($inputs)";
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_TD
