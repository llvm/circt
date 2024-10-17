//===- AIGOps.td - AIG Op Definitions -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIG Ops are defined in tablegen.
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
    Unlike comb.and, operands can be inverted respectively.

    Example:
    ```mlir
      %r1 = aig.and_inv %a, %b: i3
      %r2 = aig.and_inv not %a, %b, not %c : i3
      %r3 = aig.and_inv not %a : i3
    ```
  }];
  // TODO: Restrict to HWIntegerType.
  let arguments = (ins Variadic<AnyType>:$inputs, DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);

  // NOTE: Custom assembly format is needed to pretty print the `inverted`
  // attribute.
  let hasCustomAssemblyFormat = 1;

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

def CutOp : AIGOp<"cut", [IsolatedFromAbove, SingleBlock]> {
  let summary = "AIG dialect Cut operation";
  let description = [{
    The `aig.cut` operation represents a cut in the And-Inverter-Graph.
    This operation is variadic and can take multiple inputs and outputs,
    which corresponds to the input and output edges in AIG conceptually.

    ```mlir
    %0, %1 = aig.cut %a, %b, %c, %d : (i1, i1, i1, i1) -> (i1, i1) {
    ^bb0(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1):
      %0 = aig.and_inv not %arg0, %arg1 : i1
      %1 = aig.and_inv %arg1, %arg3 : i1
      aig.output %0, %1 : i1
    }
    ```

  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $results) $body
  }];

  let builders = [
    OpBuilder<(ins
          CArg<"TypeRange", "{}">:$resultTypes,
          CArg<"ValueRange", "{}">:$inputs,
          CArg<"std::function<void()>", "{}">:$ctor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def OutputOp : AIGOp<"output", [Terminator,
                                ReturnLike, ParentOneOf<["CutOp"]>]> {
  let summary = "AIG dialect Output operation";
  let description = [{
    The `aig.output` operation represents out edges of a cut.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
}

#endif // CIRCT_DIALECT_AIG_OPS_TD
