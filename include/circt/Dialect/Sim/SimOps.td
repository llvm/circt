//===- SimOps.td - `sim` dialect ops -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for `sim`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPS_TD
#define CIRCT_DIALECT_SIM_SIMOPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "circt/Dialect/Sim/SimDialect.td"
include "circt/Dialect/Seq/SeqTypes.td"

class SimOp<string mnemonic, list<Trait> traits = []> :
    Op<SimDialect, mnemonic, traits>;

def PlusArgsTestOp : SimOp<"plusargs.test", [Pure]> {
  let summary = "SystemVerilog `$test$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found);
  let assemblyFormat = "$formatString attr-dict";
}

def PlusArgsValueOp : SimOp<"plusargs.value", [Pure]> {
  let summary = "SystemVerilog `$value$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found, AnyType:$result);
  let assemblyFormat = "$formatString attr-dict `:` type($result)";
}

def FinishOp : SimOp<"finish"> {
  let summary = "Simulation finish condition";

  let arguments = (ins ClockType:$clk, I1:$cond);
  let results = (outs);

  let assemblyFormat = "$clk `,` $cond attr-dict";
}

def FatalOp : SimOp<"fatal"> {
  let summary = "Simulation failure condition";

  let arguments = (ins ClockType:$clk, I1:$cond);
  let results = (outs);

  let assemblyFormat = "$clk `,` $cond attr-dict";
}

#endif // CIRCT_DIALECT_SIM_SIMOPS_TD
