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
}
#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_TD
