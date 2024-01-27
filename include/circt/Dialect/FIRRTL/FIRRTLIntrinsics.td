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

def ClockInverterIntrinsicOp : FIRRTLOp<"int.clock_inv", []> {
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

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_TD
