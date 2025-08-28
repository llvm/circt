//===- AIGOps.td - AIG Op Definitions -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for AIG.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_OPS_TD
#define CIRCT_DIALECT_AIG_OPS_TD

include "circt/Dialect/AIG/AIG.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AIGOp<string mnemonic, list<Trait> traits = []> :
  Op<AIG_Dialect, mnemonic, traits>;

def AndInverterOp : AIGOp<"and_inv", [SameOperandsAndResultType, Pure]> {
  let summary = "AIG dialect AND operation";
  let description = [{
    The `aig.and_inv` operation represents an And-Inverter in the AIG dialect.
    Unlike `comb.and`, operands can be inverted respectively.

    Example:
    ```mlir
      %r1 = aig.and_inv %a, %b: i3
      %r2 = aig.and_inv not %a, %b, not %c : i3
      %r3 = aig.and_inv not %a : i3
    ```

    Traditionally, an And-Node in AIG has two operands. However, `aig.and_inv`
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

  let builders = [
    OpBuilder<(ins "Value":$input, CArg<"bool", "false">:$invert), [{
      SmallVector<bool> inverted {invert};
      return build($_builder, $_state, {input}, inverted);
    }]>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, CArg<"bool", "false">:$invertLhs,
                   CArg<"bool", "false">:$invertRhs), [{
      SmallVector<bool> inverted {invertLhs, invertRhs};
      return build($_builder, $_state, {lhs, rhs}, inverted);
    }]>
  ];

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
}

#endif // CIRCT_DIALECT_AIG_OPS_TD
