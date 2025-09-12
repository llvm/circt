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
include "mlir/Interfaces/ControlFlowInterfaces.td"
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

def AndInverterOp : SynthOp<"aig.and_inv", [SameOperandsAndResultType, Pure]> {
  let summary = "AIG dialect AND operation";
  let description = [{
    The `synth.aig.and_inv` operation represents an And-Inverter in the AIG dialect.
    Unlike `comb.and`, operands can be inverted respectively.

    Example:
    ```mlir
      %r1 = synth.aig.and_inv %a, %b: i3
      %r2 = synth.aig.and_inv not %a, %b, not %c : i3
      %r3 = synth.aig.and_inv not %a : i3
    ```

    Traditionally, an And-Node in AIG has two operands. However, `synth.aig.and_inv`
    extends this concept by allowing variadic operands and non-i1 integer types.
    Although the final stage of the synthesis pipeline requires lowering
    everything to i1-binary operands, it's more efficient to progressively lower
    the variadic multibit operations.

    Variadic operands have demonstrated their utility in low-level optimizations
    within the `comb` dialect. Furthermore, in synthesis, it's common practice
    to re-balance the logic path. Variadic operands enable the compiler to
    select more efficient solutions without the need to traverse binary trees
    multiple times.

    The ability to represent multibit operations during synthesis is crucial for
    scalability. This approach enables a form of vectorization, allowing for
    batch processing of logic synthesis when multibit operations are constructed
    in a similar manner.
  }];
  // TODO: Restrict to HWIntegerType.
  let arguments = (ins Variadic<AnyType>:$inputs, DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    custom<VariadicInvertibleOperands>($inputs, type($result), $inverted, attr-dict)
  }];

  let builders = [OpBuilder<(ins "Value":$input, CArg<"bool", "false">:$invert),
                            [{
      SmallVector<bool> inverted {invert};
      return build($_builder, $_state, {input}, inverted);
    }]>,
                  OpBuilder<(ins "Value":$lhs, "Value":$rhs,
                                CArg<"bool", "false">:$invertLhs,
                                CArg<"bool", "false">:$invertRhs),
                            [{
      SmallVector<bool> inverted {invertLhs, invertRhs};
      return build($_builder, $_state, {lhs, rhs}, inverted);
    }]>];

  let extraClassDeclaration = [{
    // Evaluate the operation with the given input values.
    APInt evaluate(ArrayRef<APInt> inputs);

    // Check if the input is inverted.
    bool isInverted(size_t idx) {
      return getInverted()[idx];
    }
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let cppNamespace = "::circt::synth::aig";
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
