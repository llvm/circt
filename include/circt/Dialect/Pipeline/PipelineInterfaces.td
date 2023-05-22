//===- PipelineInterfaces.td - Pipeline Interfaces -----------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Pipeline dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def StageLike : OpInterface<"StageLike"> {
  let cppNamespace = "circt::pipeline";
  let description = 
  [{
    An interface for stage-like operations, which contains a notion of:
    1. data in- and outputs
    2. control in- and outputs
  }];

  let methods = [
    InterfaceMethod<
      [{"Returns the `enable` input of this stage-like operation."}],
      "Value",
      "getEnable"
    >,
    InterfaceMethod<
        [{"Returns the `valid` output of this stage-like operation."}],
        "Value",
        "getValid"
    >,

    InterfaceMethod<
      [{"Returns the inputs to this stage-like operation."}],
      "ValueRange",
      "getInputs"
    >,
    InterfaceMethod<
      [{"Returns the outputs of this stage-like operation."}],
      "ValueRange",
      "getOutputs"
    >,

    InterfaceMethod<
      [{"Return the input types of this stage-like operation."}],
      "TypeRange",
      "getInputTypes",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getInputs().getTypes();
      }]>,

    InterfaceMethod<
      [{"Return the output types of this stage-like operation."}],
      "TypeRange",
      "getOutputTypes",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOutputs().getTypes();
      }]>
  ];
}
