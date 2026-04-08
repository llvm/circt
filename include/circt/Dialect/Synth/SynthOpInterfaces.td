//===- SynthOpInterfaces.td - Synth operation interfaces ------*- tablegen -*-===//
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
    Common interface for bitwise boolean-logic operations.
  }];

  let methods = [
    InterfaceMethod<[{
        Get the number of logical inputs to this operation.
      }],
      "unsigned", "getNumLogicInputs", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp->getNumOperands();
      }]
    >,
    InterfaceMethod<[{
        Get the logical input at the specified index.
      }],
      "::mlir::Value", "getLogicInput", (ins "unsigned":$index), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp->getOperand(index);
      }]
    >,
    InterfaceMethod<[{
        Return whether the specified logical input is inverted before the
        operation is evaluated.
      }],
      "bool", "isLogicInputInverted", (ins "unsigned":$index), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp.isInverted(index);
      }]
    >,
    InterfaceMethod<[{
        Evaluate the boolean logic operation for the provided inputs.
      }],
      "::llvm::APInt", "evaluateBooleanLogic",
      (ins "::llvm::ArrayRef<::llvm::APInt>":$inputs), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp.evaluate(inputs);
      }]
    >,
    InterfaceMethod<[{
        Return the single-bit truth table for this operation. Bit `i` of the
        returned APInt stores the output for assignment `i`, where operand `0`
        corresponds to the least-significant assignment bit.
      }],
      "::llvm::APInt", "getSingleBitTruthTable", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        unsigned numInputs = concreteOp.getNumLogicInputs();
        size_t numRows = size_t{1} << numInputs;
        assert(numRows <= std::numeric_limits<unsigned>::max() &&
               "truth table too wide to materialize");

        ::llvm::APInt truthTable(static_cast<unsigned>(numRows), 0);
        ::llvm::SmallVector<::llvm::APInt> inputs;
        inputs.reserve(numInputs);
        for (size_t assignment = 0; assignment < numRows; ++assignment) {
          inputs.clear();
          for (unsigned input = 0; input < numInputs; ++input)
            inputs.emplace_back(1, (assignment >> input) & 1);

          if (concreteOp.evaluateBooleanLogic(inputs)[0])
            truthTable.setBit(assignment);
        }
        return truthTable;
      }]
    >
  ];
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPINTERFACES_TD
