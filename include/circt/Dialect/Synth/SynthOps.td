//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
#define CIRCT_DIALECT_SYNTH_SYNTHOPS_TD

include "circt/Dialect/Synth/Synth.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class SynthOp<string mnemonic, list<Trait> traits = []> :
    Op<Synth_Dialect, mnemonic, traits>;

def MajorityInverterOp : SynthOp<"mig.maj_inv",
                                 [SameOperandsAndResultType, Pure]> {
  let summary = "Majority-Inverter operation";
  let description = [{
    The `synth.mig.maj_inv` operation represents a Majority-Inverter in the
    Synth dialect. This is used to represent majority inverter graph in
    synthesis. This operation computes the majority function of its inputs,
    where operands can be inverted respectively.

    The majority function returns 1 when more than half of the inputs are 1,
    and 0 otherwise. For three inputs, it's equivalent to:
    (a & b) | (a & c) | (b & c).

    Example:
    ```mlir
      %r1 = synth.mig.maj_inv %a, %b, %c : i1
      %r2 = synth.mig.maj_inv not %a, %b, not %c : i1
      %r3 = synth.mig.maj_inv %a, %b, %c, %d, %e : i3
    ```

    The number of inputs must be odd to avoid ties.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs,
                       DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);
  let hasVerifier = true;

  let assemblyFormat = [{
    custom<VariadicInvertibleOperands>($inputs, type($result), $inverted,
                                       attr-dict)
  }];
  let cppNamespace = "::circt::synth::mig";
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
