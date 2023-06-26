//===- PipelineInterfaces.td - Operation Interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the Pipeline dialect operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIPELINE_PIPELINEINTERFACES_TD
#define CIRCT_DIALECT_PIPELINE_PIPELINEINTERFACES_TD

include "mlir/IR/OpBase.td"

def PipelineLike : OpInterface<"PipelineLike"> {
  let cppNamespace = "circt::pipeline";
  let description = "Trait for pipeline-like operations";

  let methods = [
    InterfaceMethod<"Returns true if the pipeline is latency insensitive",
      "bool",
      "isLatencyInsensitive", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        bool allInputsAreChannels = llvm::all_of($_op.getInputs(), [](Value operand) {
            return operand.getType().isa<esi::ChannelType>();
        });
        bool allOutputsAreChannels = llvm::all_of(
            $_op.getResultTypes(), [](Type type) { return type.isa<esi::ChannelType>(); });
        return allInputsAreChannels && allOutputsAreChannels;
      }]
      >,
   InterfaceMethod<"Returns true if the pipeline is latency sensitive",
      "bool",
      "isLatencySensitive", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return !isLatencyInsensitive();
      }]
      >,
    InterfaceMethod<"Returns the pipeline inputs",
      "ValueRange",
      "getInputs", (ins)>,
    InterfaceMethod<"Returns the inner pipeline inputs",
      "ValueRange",
      "getInnerInputs", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Block* firstStage = $_op.getEntryStage();        
        auto inputs = $_op.getInputs();
        if(inputs.empty())
          return ValueRange();
        
        return firstStage->getArguments().slice(
          inputs.getBeginOperandIndex(),
          inputs.size());
      }]
    >,
    InterfaceMethod<"Returns the pipeline external inputs",
      "ValueRange",
      "getExtInputs", (ins)>,
    InterfaceMethod<"Returns the inner external inputs",
      "ValueRange",
      "getInnerExtInputs", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Block* firstStage = $_op.getEntryStage();
        auto extInputs = $_op.getExtInputs();
        if(extInputs.empty())
          return ValueRange();

        return firstStage->getArguments().slice(
          extInputs.getBeginOperandIndex(),
          extInputs.size());
      }]
     >,
    InterfaceMethod<"Returns the first stage in the pipeline",
      "Block*",
      "getEntryStage",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return &$_op->getRegions().front().front();
      }]
      >
  ];
}

#endif // CIRCT_DIALECT_PIPELINE_PIPELINEINTERFACES_TD
