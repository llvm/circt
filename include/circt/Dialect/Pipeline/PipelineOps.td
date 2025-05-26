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

include "circt/Dialect/Pipeline/PipelineDialect.td"
include "circt/Dialect/Seq/SeqTypes.td"


class PipelineBase<string mnemonic, list<Trait> traits = []> :
  Op<Pipeline_Dialect, mnemonic, !listconcat(traits, [
      RegionKindInterface,
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<OpAsmOpInterface, [
        "getAsmResultNames",
        "getAsmBlockArgumentNames"]>
      ])> {
  let results = (outs Variadic<AnyType>:$dataOutputs, I1:$done);
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins
      "TypeRange":$dataOutputs,
      "ValueRange":$inputs,
      "ArrayAttr":$inputNames, "ArrayAttr":$outputNames,
      "Value":$clock, "Value":$go,
      CArg<"Value", "{}">:$reset,
      CArg<"Value", "{}">:$stall,
      CArg<"StringAttr", "{}">:$name,
      CArg<"ArrayAttr", "{}">:$stallability
      )>
  ];

  /// Additional class declarations inside the pipeline op.
  code extraModuleClassDeclaration = ?;

  // The entry block of the pipeline contains a lot of block arguments - the
  // indices are kept away from the user and only "hardcoded" here, as well as
  // in the printer/parser.
  // Order:
  // 1. Inputs
  // 2. Stall (opt)
  // 3. Clock
  // 4. Reset (opt)
  // 5. Go
  let extraClassDeclaration = extraModuleClassDeclaration # [{
    // Returns the entry stage of this pipeline.
    Block* getEntryStage() {
      Region* region = &getOperation()->getRegion(0);
      return &region->front();
    }

    // Returns true if this pipeline has a stall signal.
    bool hasStall() {
      return static_cast<bool>(getStall());
    }

    // Returns true if this pipeline has a reset signal.
    bool hasReset() {
      return static_cast<bool>(getReset());
    }

    mlir::Block::BlockArgListType getInnerInputs() {
      return getEntryStage()->getArguments().take_front(getInputs().size());
    }

    // Returns a list of external inputs to the pipeline. These are defined
    // as any value which is defined outside the inner pipeline region.
    // This will walk the entire pipeline, so it is recommended to cache the
    // result if it is used multiple times.
    llvm::SmallVector<Value> getExtInputs() {
      return detail::getValuesDefinedOutsideRegion(getRegion());
    }

    // Gets the n'th stage of this pipeline
    Block* getStage(unsigned n) {
      auto& blocks = getRegion().getBlocks();
      assert(n < blocks.size() && "Stage index out of bounds");
      return &*std::next(blocks.begin(), n);
    }

    // Returns the enable signal of the given pipeline stage. This is always
    // the last block argument of a stage for anything but the entry stage.
    Value getStageEnableSignal(size_t stageIdx) {
      return getStageEnableSignal(getStage(stageIdx));
    }

    // Returns the enable signal for the given stage. The enable signal is always
    // the last signal in the stage argument list.
    Value getStageEnableSignal(Block* stage) {
      return stage->getArguments().back();
    }
  }];
}

def UnscheduledPipelineOp : PipelineBase<"unscheduled", [
    RegionKindInterface,
    HasOnlyGraphRegion,
    SingleBlockImplicitTerminator<"ReturnOp">
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
    OptionalAttr<StrAttr>:$name, Variadic<AnyType>:$inputs, Optional<I1>:$stall,
    ClockType:$clock, Optional<I1>:$reset, I1:$go, StrArrayAttr:$inputNames,
    StrArrayAttr:$outputNames
  );
  let regions = (region SizedRegion<1>: $body);
  let extraModuleClassDeclaration = "";
}

def ScheduledPipelineOp : PipelineBase<"scheduled"> {
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
    the stages as the current stage valid/next stage enable signal.
    Note: the op is currently only designed for pipelines with II=1. For
    pipelines with II>1, a user must themselves maintain state about when
    the pipeline is ready to accept new inputs. We plan to add support for
    head-of-pipeline backpressure in the future.

    Any value defined outside the pipeline is considered an external input. An
    external input will _not_ be registered.

    The pipeline may optionally be provided with an array of bits `stallability`
    which is used to determine which stages are stallable.
    - If not provided and the pipeline has a stall signal, all stages are stallable.
    - If provided, and the pipeline has a stall signal, the number of bits must
      match the number of stages in the pipeline. Each bit represents a stage,
      in the order of which the stages appear wrt. the `pipeline.stage` operations.
      A bit set to 1 indicates that the stage is stallable, and 0 indicates that
      the stage is not stallable.

    The exit (non-registered) stage of a pipeline cannot be non-stallable, and
    will always follow the stallability of the parent pipeline.

    For more information about non-stallable stages, and how these are lowered,
    please refer to the Pipeline dialect rationale.
  }];


  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    Variadic<AnyType>:$inputs,
    Optional<I1>:$stall,
    ClockType:$clock, Optional<I1>:$reset, I1:$go,
    StrArrayAttr:$inputNames, StrArrayAttr:$outputNames,
    OptionalAttr<BoolArrayAttr>:$stallability
  );
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;

  code extraModuleClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::SSACFG;
    }

    // Returns all of the stages in this pipeline.
    llvm::iplist<mlir::Block>& getStages() {
      return getRegion().getBlocks();
    }

    size_t getNumStages() {
      return getStages().size();
    }

    // Returns all of the stages in this pipeline. The stages are ordered
    // with respect to their position in the pipeline as determined by the
    // control flow of `pipeline.stage` operations.
    // Stages are ordered from first (entry) to last (exit).
    llvm::SmallVector<Block*> getOrderedStages();

    // Returns a map of stages to their index in the pipeline - this is
    // with respect to 'getOrderedStages'.
    llvm::DenseMap<Block*, unsigned> getStageMap();

    // Returns the last stage in the pipeline.
    Block* getLastStage();

    // Adds a new stage to this pipeline. It is the users responsibility to
    // modify other stages to point to this new stage.
    Block* addStage();

    // Returns true if the pipeline has entered the register materialized phase.
    // This enables certain invariants such as "all values used within a stage
    // must be defined within a stage".
    bool isMaterialized();

    // Returns the data arguments for a stage. The stage enable signal is _not_ part
    // of the returned values.
    mlir::Block::BlockArgListType getStageDataArgs(Block* stage) {
      if(stage == getEntryStage())
        return getInnerInputs();

      // Data arguments for all non-entry stages are all block arguments
      // except for the last block arguments (which is the stage enable signal).
      return stage->getArguments().drop_back();
    }

    // Returns the stage kind of the given stage index.
    StageKind getStageKind(size_t stageIdx);
  }];
}

def SourceOp : Op<Pipeline_Dialect, "src", [
  Pure,
  TypesMatchWith<"input and result types are equivalent", "input", "output", "$_self">,
  HasParent<"ScheduledPipelineOp">,
  ]> {
  let summary = "Pipeline source operation";
  let description = [{
    The `pipeline.src` operation represents a source operation in a scheduled,
    non-register materialized pipeline.
    It is used as a canonicalization barrier to prevent cross-block canonicalization
    of operations that are not allowed to be moved or mutated across pipeline
    stages (i.e. MLIR blocks).
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $input `:` type($input) attr-dict
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
    4. An optional hierarchy of boolean values to be used for clock gates for
       each register.
      - The implicit '!stalled' gate will always be the first signal in the
        hierarchy. Further signals are added to the hierarchy from left to
        right.


    Example:
    ```mlir
    pipeline.stage ^bb1 regs(%a : i32 gated by [%foo, %bar], %b : i1) pass(%c : i32)
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$registers,
    Variadic<AnyType>:$passthroughs,
    Variadic<I1>:$clockGates,
    I64ArrayAttr:$clockGatesPerRegister,
    OptionalAttr<StrArrayAttr>:$registerNames,
    OptionalAttr<StrArrayAttr>:$passthroughNames);
  let successors = (successor AnySuccessor:$nextStage);
  let results = (outs);
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    $nextStage
    custom<StageRegisters>($registers, type($registers), $clockGates, $clockGatesPerRegister, $registerNames)
    custom<Passthroughs>($passthroughs, type($passthroughs), $passthroughNames)
    attr-dict
  }];

  let extraClassDeclaration = [{
    // Set the destination stage.
    void setNextStage(Block *block) {
      setSuccessor(block);
    }

    // Returns the list of clock gates for the given register.
    ValueRange getClockGatesForReg(unsigned regIdx);

    // Returns the register name for a given register index.
    StringAttr getRegisterName(unsigned regIdx) {
      if(auto names = getOperation()->getAttrOfType<ArrayAttr>("registerNames")) {
        auto name = llvm::cast<StringAttr>(names[regIdx]);
        if(!name.strref().empty())
          return name;
      }

      return {};
    }

    // Returns the passthrough name for a given passthrough index.
    StringAttr getPassthroughName(unsigned passthroughIdx) {
      if(auto names = getOperation()->getAttrOfType<ArrayAttr>("passthroughNames")) {
        auto name = llvm::cast<StringAttr>(names[passthroughIdx]);
        if(!name.strref().empty())
          return name;
      }

      return {};
    }


  }];

  let builders = [
    OpBuilder<(ins "Block*":$dest, "ValueRange":$registers, "ValueRange":$passthroughs)>,
    OpBuilder<(ins "Block*":$dest, "ValueRange":$registers, "ValueRange":$passthroughs,
      // Clock gates per register, in the order of the registers.
      "llvm::ArrayRef<llvm::SmallVector<Value>>":$clockGates,
      CArg<"mlir::ArrayAttr", "{}">:$registerNames, CArg<"mlir::ArrayAttr", "{}">:$passthroughNames)>
  ];
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

  let assemblyFormat = [{
    $latency `->` `(` type($results) `)` $body attr-dict
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
