//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPINTERFACES_TD
#define CIRCT_DIALECT_SYNTH_SYNTHOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def BooleanLogicOpInterface : OpInterface<"BooleanLogicOpInterface"> {
  let cppNamespace = "::circt::synth";

  let description = [{
    Common interface for boolean-logic DAG nodes whose semantics are defined by
    a list of input values together with per-input inversion flags.

    Modern logic-network representations often use the same "value plus
    complemented edge" model across several node kinds, for example AIG-style
    nodes as well as closely related 3-input or otherwise non-binary DAG
    nodes. Passes such as structural hashing, bit lowering, and simulation
    should be able to operate on those common semantics without hardcoding a
    specific concrete op.
  }];

  let methods = [
    InterfaceMethod<[{
        Get the logical input values.
      }],
      "::mlir::ValueRange", "getInputs", (ins), "",
      [{
        return $_op.getInputs();
      }]
    >,
    InterfaceMethod<[{
        Get the logical input at the specified index and whether it is
        inverted before the operation is evaluated.
      }],
      "::std::pair<::mlir::Value, bool>", "getInputPair", (ins "unsigned":$index), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return {concreteOp->getOperand(index), concreteOp.isInverted(index)};
      }]
    >,

    InterfaceMethod<[{
        Get the logical input value at the specified index.
      }],
      "::mlir::Value", "getInput", (ins "unsigned":$index), "",
      [{
        return $_op.getInputs()[index];
      }]
    >,
    InterfaceMethod<[{
        Return whether the specified logical input is inverted before the
        operation is evaluated.
      }],
      "bool", "isInverted", (ins "unsigned":$index), "",
      [{
        return $_op.getInverted()[index];
      }]
    >,
    InterfaceMethod<[{
        Get the inversion flags for all logical inputs.
      }],
      "::llvm::ArrayRef<bool>", "getInverted", (ins)>,
    InterfaceMethod<[{
        Set the inversion flags for all logical inputs.
      }],
      "void", "setInverted", (ins "::llvm::ArrayRef<bool>":$inverted)
    >,
    InterfaceMethod<[{
        Get the primary logical result of this operation.
      }],
      "::mlir::Value", "getResult", (ins), "",
      [{
        return $_op->getResult(0);
      }]
    >,
    InterfaceMethod<[{
        Return whether the operation semantics are invariant under
        permutation of logical input pairs. Reordering a logical input means
        moving both the input value and its inversion flag together.
      }],
      "bool", "areInputsPermutationInvariant", (ins), "",
      [{
        return false;
      }]
    >,
    InterfaceMethod<[{
        Evaluate the boolean logic operation for the provided raw operand
        inputs. The implementations are responsible for applying any per-input
        inversion described by the operation itself.
      }],
      "::llvm::APInt", "evaluateBooleanLogic",
      (ins "::llvm::function_ref<const ::llvm::APInt &(unsigned)>":$getInputValue)
    >,
    InterfaceMethod<[{
        Compute known bits for the result from the logical inputs.
        The implementations are responsible for applying any per-input inversion
        described by the operation itself.
      }],
      "::llvm::KnownBits", "computeKnownBits",
      (ins "::llvm::function_ref<const ::llvm::KnownBits &(unsigned)>":$getInputKnownBits)
    >,
    InterfaceMethod<[{
        Clone or fold this operation with the provided inputs while preserving
        the existing inversion flags.
      }],
      "::mlir::Value", "cloneWithSameInversion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::ValueRange":$newInputs), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        assert(!newInputs.empty());
        auto resultType = newInputs.front().getType();
        return builder.createOrFold<ConcreteOp>(concreteOp.getLoc(), resultType,
                                                newInputs,
                                                concreteOp.getInverted());
      }]
    >
  ];
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPINTERFACES_TD
