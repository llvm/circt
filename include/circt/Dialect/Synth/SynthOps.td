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

class InvertibleOperandsOp<string mnemonic, list<Trait> traits = []>
  : SynthOp<mnemonic,[
      // Share the BooleanLogicOpInterface contract and invertible operand
      // syntax across Synth ops such as `synth.aig.and_inv`.
      SameOperandsAndResultType, Pure,
      DeclareOpInterfaceMethods<BooleanLogicOpInterface, [
        "areInputsPermutationInvariant", "evaluateBooleanLogicWithoutInversion",
        "supportsNumInputs", "createBooleanLogicOp", "getLogicDepthCost",
        "getLogicAreaCost"
      ]> ] # traits> {
  // TODO: Restrict to HWIntegerType.
  let arguments = (ins Variadic<AnyType>:$inputs,
                       DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    custom<VariadicInvertibleOperands>($inputs, type($result), $inverted, attr-dict)
  }];
}

def AndInverterOp : InvertibleOperandsOp<"aig.and_inv"> {
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

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let cppNamespace = "::circt::synth::aig";
}

def XorInverterOp : InvertibleOperandsOp<"xor_inv"> {
  let summary = "XOR gate with invertible variadic inputs";
  let description = [{
    The `synth.xor_inv` operation represents a parity node whose operands may be
    individually inverted before the XOR is evaluated.

    Example:
    ```mlir
      %r0 = synth.xor_inv %a, %b : i1
      %r1 = synth.xor_inv not %a, %b, %c : i8
      %r2 = synth.xor_inv not %a : i1
    ```
  }];

  let builders = [
    OpBuilder<(ins "Value":$input, CArg<"bool", "false">:$invert),
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
    }]>,
    OpBuilder<(ins "ArrayRef<std::pair<Value, bool>>":$operands),
                            [{
      SmallVector<Value> values;
      SmallVector<bool> inverts;
      for (auto &[val, inv] : operands) {
        values.push_back(val);
        inverts.push_back(inv);
      }
      return build($_builder, $_state, values, inverts);
    }]>
    ];
    
    let hasFolder = 1;
    let hasCanonicalizeMethod = 1;

}

def DotOp : SynthOp<"dot", [
    SameOperandsAndResultType, Pure,
    DeclareOpInterfaceMethods<BooleanLogicOpInterface, [
      "areInputsPermutationInvariant", "evaluateBooleanLogicWithoutInversion",
      "supportsNumInputs", "createBooleanLogicOp",
      "getLogicAreaCost"
    ]>
  ]> {
  let summary = "Ordered 3-input DOT gate with invertible inputs";
  let description = [{
    The `synth.dot` operation represents the ordered three-input Boolean
    function `x xor (z or (x and y))`, applied after optional per-input
    inversion.

    Unlike common AND, OR, XOR, and majority gates, `synth.dot` is not
    permutation-invariant: its operands play distinct roles in the Boolean
    expression. Dot gates are known to be a promising intermediate
    representation for area optimization, as discussed in "Three-Input Gates
    for Logic Synthesis" (Dewmini Sudara Marakkalage et al., TCAD 2021).

    Example:
    ```mlir
      %r0 = synth.dot %x, %y, %z : i1
      %r1 = synth.dot not %x, %y, not %z : i8
    ```
  }];

  let arguments = (ins AnyType:$x, AnyType:$y, AnyType:$z,
                       DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$x, "Value":$y, "Value":$z,
                                CArg<"bool", "false">:$invertX,
                                CArg<"bool", "false">:$invertY,
                                CArg<"bool", "false">:$invertZ), [{
      SmallVector<bool> inverted{invertX, invertY, invertZ};
      return build($_builder, $_state, x.getType(), x, y, z,
                   $_builder.getDenseBoolArrayAttr(inverted));
    }]>,
    OpBuilder<(ins "ValueRange":$inputs, "ArrayRef<bool>":$inverted), [{
      assert(inputs.size() == 3 && "dot expects exactly three operands");
      assert(inverted.size() == 3 && "dot expects exactly three inversion flags");
      return build($_builder, $_state, inputs[0].getType(), inputs[0], inputs[1],
                   inputs[2], $_builder.getDenseBoolArrayAttr(inverted));
    }]>,
    OpBuilder<(ins "Type":$resultType, "ValueRange":$inputs,
                                "ArrayRef<bool>":$inverted), [{
      return build($_builder, $_state, inputs, inverted);
    }]>];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ChoiceOp : SynthOp<"choice", [SameOperandsAndResultType, Pure]> {
  let summary = "Synth choice operation";
  let description = [{
    The `synth.choice` operation represents a set of structurally equivalent
    choices, corresponding to the choice nodes introduced in "Reducing
    Structural Bias in Technology Mapping" (ICCAD 2005, S. Chatterjee et al.).

    A compiler may assume all operands are equivalent and replace the operation
    with any one of them at any point in the pipeline.

    Example:
    ```mlir
      %r1 = synth.choice %a, %b : i1
      %r2 = synth.choice %a, %b, %c : i16
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = "$inputs attr-dict `:` type($result)";
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
