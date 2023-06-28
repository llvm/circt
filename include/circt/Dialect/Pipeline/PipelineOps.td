//===- PipelineOps.td - Pipeline dialect operations --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIPELINE_PIPELINEOPS_TD
#define CIRCT_DIALECT_PIPELINE_PIPELINEOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

include "circt/Dialect/Pipeline/PipelineInterfaces.td"
include "circt/Dialect/Pipeline/PipelineDialect.td"

def UnscheduledPipelineOp : Op<Pipeline_Dialect, "unscheduled", [
    IsolatedFromAbove,
    Pure,
    SingleBlockImplicitTerminator<"ReturnOp">,
    RegionKindInterface,
    HasOnlyGraphRegion,
    PipelineLike,
    AttrSizedOperandSegments
  ]> {
  
  let summary = "unscheduled pipeline operation";
  let description = [{
    The "pipeline.unscheduled" operation represents a pipeline that has not yet
    been scheduled. It contains a single block representing a graph region of
    operations to-be-scheduled into a pipeline.
    Mainly serves as a container and entrypoint for scheduling.

    The interface of a `pipeline.unscheduled` is similar to that of a
    `pipeline.scheduled`. Please refer to this op for further documentation
    about the interface signals.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs, Variadic<AnyType>:$extInputs, I1:$go, I1:$clock, I1:$reset, Optional<I1>:$stall
  );
  let results = (outs Variadic<AnyType>:$results, I1:$done);
  let regions = (region SizedRegion<1>: $body);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $inputs `)` (`stall` $stall^)? (`ext` `(` $extInputs^ `:` type($extInputs) `)`)? 
      `clock` $clock `reset` $reset `go` $go attr-dict `:`functional-type($inputs, $results) $body
  }];

  let extraClassDeclaration = [{
    /// Returns the body of a Pipeline component.
    Block *getBodyBlock() {
      Region* region = &getOperation()->getRegion(0);
      assert(region->hasOneBlock() && "The body should have one Block.");
      return &region->front();
    }
  }];
}

def ScheduledPipelineOp : Op<Pipeline_Dialect, "scheduled", [
    IsolatedFromAbove,
    Pure,
    RegionKindInterface,
    PipelineLike,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Scheduled pipeline operation";
  let description = [{
    The `pipeline.scheduled` operation represents a scheduled pipeline.
    The pipeline contains a single block representing a graph region.

    A `pipeline.scheduled` operation can exist in multiple phases, mainly
    pertaining to when registers have been materialized (made explicit).
    For an in-depth explanation, please refer to the Pipeline dialect rationale.

    A `pipeline.scheduled` supports a `stall` input. This signal is intended to
    connect to all stages within the pipeline, and is used to stall the entirety
    of the pipeline. It is lowering defined how stages choose to use this signal,
    although in the common case, a `stall` signal would typically connect to
    the clock-enable input of the stage-separating registers.

    The `go` input is used to start the pipeline. This value is fed through
    the stages as the stage valid signal.
    Note: the op is currently only designed for pipelines with II=1. For
    pipelines with II>1, a user must themselves maintain state about when
    the pipeline is ready to accept new inputs. We plan to add support for
    head-of-pipeline backpressure in the future.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs, Variadic<AnyType>:$extInputs,  I1:$clock, I1:$reset, I1:$go, Optional<I1>:$stall
  );
  let results = (outs Variadic<AnyType>:$results, I1:$done);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "ValueRange":$inputs, "ValueRange":$extInputs,
      "Value":$clock, "Value":$reset, "Value":$go, CArg<"Value", "{}">:$stall)>
  ];

  let assemblyFormat = [{
    `(` $inputs `)` (`stall` $stall^)? (`ext` `(` $extInputs^ `:` type($extInputs) `)`)? 
      `clock` $clock `reset` $reset `go` $go attr-dict `:`functional-type($inputs, $results) $body
  }];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::SSACFG;
    }

    // Returns all of the stages in this pipeline.
    llvm::iplist<mlir::Block>& getStages() {
      return getRegion().getBlocks();
    }

    // Returns the valid signal of the given pipeline stage. This is always
    // the last block argument of a stage.
    Value getStageValidSignal(size_t stageIdx) {
      Block* stage = getStage(stageIdx);
      return stage->getArguments().back();
    }

    void getAsmBlockArgumentNames(mlir::Region &region,
                        mlir::OpAsmSetValueNameFn setNameFn);

    // Returns all of the stages in this pipeline. The stages are ordered
    // with respect to their position in the pipeline as determined by the
    // control flow of `pipeline.stage` operations.
    // Stages are ordered from first (entry) to last (exit).
    llvm::SmallVector<Block*> getOrderedStages();

    // Returns a map of stages to their index in the pipeline - this is
    // with respect to 'getOrderedStages'.
    llvm::DenseMap<Block*, unsigned> getStageMap();

    // Gets the n'th stage of this pipeline
    Block* getStage(unsigned n) {
      auto& blocks = getRegion().getBlocks();
      assert(n < blocks.size() && "Stage index out of bounds");
      return &*std::next(blocks.begin(), n);
    }

    // Returns the entry stage of this pipeline.
    Block* getEntryStage() {
      return getStage(0);
    }

    // Returns the last stage in the pipeline.
    Block* getLastStage();

    // Adds a new stage to this pipeline. It is the users responsibility to
    // modify other stages to point to this new stage.
    Block* addStage();

    // Returns true if the pipeline has entered the register materialized phase.
    // This enables certain invariants such as "all values used within a stage
    // must be defined within a stage".
    bool isMaterialized();

    // Returns the arguments for a stage. The stage valid signal is _not_ part
    // of the returned values.
    ValueRange getStageArguments(Block* stage) {
      if(stage != getEntryStage())
        return stage->getArguments().drop_back();

      // The entry stage block mirrors the input argument order of the pipeline.
      // Grab the arguments corresponding to the inputs.
      return stage->getArguments().slice(getInputs().getBeginOperandIndex(), getInputs().size());
    }

    // Returns the valid signal for the given stage. The valid signal is always
    // the last signal in the stage argument list.
    Value getStageValidSignal(Block* stage) {
      return stage->getArguments().back();
    }

    size_t getNumStageArguments(Block* stage) {
      if(stage != getEntryStage())
        return stage->getNumArguments() - /*stage valid*/1;
      return getInputs().size();
    }
  }];
}


def StageOp : Op<Pipeline_Dialect, "stage", [
    AttrSizedOperandSegments,
    HasParent<"ScheduledPipelineOp">,
    Pure,
    Terminator
  ]> {
  let summary = "Pipeline stage terminator.";
  let description = [{
    The `pipeline.stage` operation represents a stage terminator. It is used
    to communicate:
    1. which stage (block) to transition to next
    2. which registers to build at this stage boundary
    3. which values to pass through to the next stage without registering
  }];

  let arguments = (ins Variadic<AnyType>:$registers, Variadic<AnyType>:$passthroughs);
  let successors = (successor AnySuccessor:$nextStage);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $nextStage (`regs` `(` $registers^ `:` type($registers) `)`)?
      (`pass` `(` $passthroughs^ `:` type($passthroughs) `)` )? attr-dict
  }];

  let extraClassDeclaration = [{
    // Set the destination stage.
    void setNextStage(Block *block) {
      setSuccessor(block);
    }
  }];
}


def ReturnOp : Op<Pipeline_Dialect, "return", [
    Terminator,
    ParentOneOf<["UnscheduledPipelineOp", "ScheduledPipelineOp"]>
  ]> {
  let summary = "Pipeline dialect return.";
  let description = [{
    The "return" operation represents a terminator of a `pipeline.pipeline`.
  }];

  let hasVerifier = 1;
  let arguments = (ins Variadic<AnyType>:$inputs);
  let builders = [OpBuilder<(ins), [{ return; }]>];
  let assemblyFormat = [{ ($inputs^)? attr-dict (`:` type($inputs)^)? }];
}

def LatencyOp : Op<Pipeline_Dialect, "latency", [
    SingleBlockImplicitTerminator<"LatencyReturnOp">,
    RegionKindInterface,
    HasOnlyGraphRegion,
    ParentOneOf<["UnscheduledPipelineOp", "ScheduledPipelineOp"]>
  ]> {
  
  let summary = "Pipeline dialect latency operation.";
  let description = [{
    The `pipeline.latency` operation represents an operation for wrapping
    multi-cycle operations. The operation declares a single block
    wherein any operation may be placed within. The operation is not
    `IsolatedFromAbove` meaning that the operation can reference values
    defined outside of the operation (subject to the materialization
    phase of the parent pipeline).
  }];

  let arguments = (ins ConfinedAttr<I32Attr, [IntMinValue<1>]>:$latency);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "unsigned":$latency)>
  ];
  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    attr-dict $latency `->` `(` type($results) `)` $body
  }];

  let extraClassDeclaration = [{
    // Returns the body block of the latency operation.
    Block* getBodyBlock() {
      return &getBody().front();
    }
  }];
}

def LatencyReturnOp : Op<Pipeline_Dialect, "latency.return", [
    Terminator,
    HasParent<"LatencyOp">
  ]> {
  let summary = "Pipeline latency return operation.";
  let description = [{
    The `pipeline.latency.return` operation represents a terminator of a
    `pipeline.latency` operation.
  }];

  let hasVerifier = 1;
  let arguments = (ins Variadic<AnyType>:$inputs);
  let builders = [OpBuilder<(ins), [{ return; }]>];
  let assemblyFormat = [{ ($inputs^)? attr-dict (`:` type($inputs)^)? }];
}


#endif // CIRCT_DIALECT_PIPELINE_PIPELINEOPS_TD
