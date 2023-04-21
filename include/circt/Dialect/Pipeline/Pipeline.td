//===- Pipeline.td - Pipeline Definitions ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pipeline Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifdef PIPELINE_OPS
#else
#define PIPELINE_OPS


include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

def Pipeline_Dialect : Dialect {
  let name = "pipeline";
  let cppNamespace = "::circt::pipeline";
}

def PipelineOp : Op<Pipeline_Dialect, "pipeline", [
    IsolatedFromAbove,
    Pure,
    SingleBlockImplicitTerminator<"ReturnOp">,
    RegionKindInterface,
    HasOnlyGraphRegion 
  ]> {
  let summary = "pipeline operation";
  let description = [{
    The "pipeline.pipeline" operation represents a retimeable pipeline.
    The pipeline contains a single block representing a graph region. Pipeline
    stages are represented by `pipeline.rt.register` operations. Semantics
    of values crossing register boundaries are defined by lowering passes.
    The pipeline representation is centered around providing latency insensitive
    values (valid signals between stages). Such signals can either be removed
    (in the feedforward, statically scheduled case), used to control stalling
    (feedback, statically scheduled case) or in conjunction with handshake signals
    for dynamically scheduled pipelines.
    A pipelines' latency sensitivity is based on the I/O of the pipeline - if
    any in- or output port is an ESI channel, all ports are expected to be ESI
    channels, and the pipeline is considered latency sensitive.
    The internal representation of the pipeline is agnostic to the latency
    insensitivity of the I/O. This is by design - allowing us a single source
    of truth for lowering either latency sensitive or latency insensitive pipelines.

    A typical flow would go like this:

    An untimed datapath is defined:
    ```
    pipeline.pipeline(%in0 : i32, %in1 : i32) -> (i32) {
      ^bb0:(%arg0 : i32, %arg1: i32):
        %add0 = comb.add %arg0, %arg1 : i32
        %add1 = comb.add %add0, %arg0 : i32
        %add2 = comb.add %add1, %arg1 : i32
        pipeline.return %add2 : i32
    }
    ```
    The datapath is scheduled:
    ```
    pipeline.pipeline(%in0 : i32, %in1 : i32) -> (i32) {
      ^bb0:(%arg0 : i32, %arg1: i32, %go : i1):
        %add0 = comb.add %arg0, %arg1 : i32

        %s0_valid = pipeline.stage when %go
        %add1 = comb.add %add0, %arg0 : i32
  
        %s1_valid = pipeline.stage when %g1
        %add2 = comb.add %add1, %arg1 : i32

        pipeline.return %add2 valid %s1_valid : i32
    }
    ```

    Stage-crossing dependencies are made explicit through registers.
    ```
    pipeline.pipeline(%in0 : i32, %in1 : i32) -> (i32) {
      ^bb0:(%arg0 : i32, %arg1: i32):
        %add0 = comb.add %arg0, %arg1 : i32

        %s0_valid, %add0_r = pipeline.stage.register when %go regs (%add0: i32)
        %add1 = comb.add %add0_r, %arg0 : i32
  
        %s1_valid, %add1_r = pipeline.stage.register when %g1 regs (%add1: i32)
        %add2 = comb.add %add1_r, %arg1 : i32

        pipeline.return %add2 valid %s1_valid : i32
    }
    ```

    This representation can then be lowered to statically or dynamically scheduled
    pipelines.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs, I1:$clock, I1:$reset
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>: $body);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $inputs `)` `clock` $clock `reset` $reset attr-dict `:` functional-type($inputs, results) $body
  }];

  let extraClassDeclaration = [{
    // Returns true if this pipeline has a latency-insensitive interface.
    // Latency sensitivity is defined based on whether _any_ `$input` is an 
    // ESI channel.
    bool isLatencyInsensitive(); 
    bool isLatencySensitive() { return !isLatencyInsensitive(); }

    /// Returns the body of a Pipeline component.
    Block *getBodyBlock() {
      Region* region = &getOperation()->getRegion(0);
      assert(region->hasOneBlock() && "The body should have one Block.");
      return &region->front();
    }

  }];
}


def PipelineStageOp : Op<Pipeline_Dialect, "stage", [
    HasParent<"PipelineOp">
  ]> {
  let summary = "Pipeline pipeline stage.";
  let description = [{
    The `pipeline.stage` operation represents a stage separating register
    in a pipeline. The stage does not define any explicit registers, but solely
    defines a cut of a dataflow graph based on its lexical position in the
    pipeline. Pipeline registers are made explicit through the register
    materialization pass, wherein this op is replaced by
    `pipeline.stage.register` operations.
  }];

  let arguments = (ins I1:$when);
  let results = (outs I1:$valid);

  let assemblyFormat = [{
    `when` $when attr-dict
  }];

  let extraClassDeclaration = [{
    // Returns the index of this stage in the pipeline.
    unsigned index() {
      auto stageOps = getOperation()->getParentOfType<PipelineOp>().getOps<PipelineStageOp>();
      return std::distance(stageOps.begin(), llvm::find(stageOps, *this));
    }
  }];
}

def PipelineStageRegisterOp : Op<Pipeline_Dialect, "stage.register", [
    HasParent<"PipelineOp">,
    RangedTypesMatchWith<"result type matches operand", "regIns", "regOuts",
                         "llvm::make_range($_self.begin(), $_self.end())">
  ]> {
  let summary = "Pipeline pipeline stage.";
  let description = [{
    The `pipeline.stage` operation represents a stage separating register
    in a pipeline with materialized register values. `pipeline.stage` and
    `pipeline.stage.register` operations are not allowed to co-exist in the
    same pipeline body. This is because, once register values are materialized,
    all delays as well as knowledge about multicycle paths have been lowered
    away.
  }];

  let arguments = (ins Variadic<AnyType>:$regIns, I1:$when);
  let results = (outs Variadic<AnyType>:$regOuts, I1:$valid);

  let builders = [OpBuilder<(ins "mlir::Value":$when, "mlir::ValueRange":$regIns)>];

  let assemblyFormat = [{
    `when` $when (`regs` $regIns^)? attr-dict (`:` type($regIns)^)?
  }];

  let extraClassDeclaration = [{
    // Returns the index of this stage in the pipeline.
    unsigned index() {
      auto stageOps = getOperation()->getParentOfType<PipelineOp>().getOps<PipelineStageRegisterOp>();
      return std::distance(stageOps.begin(), llvm::find(stageOps, *this));
    }
  }];
}

def ReturnOp : Op<Pipeline_Dialect, "return", [Terminator]> {
  let summary = "Pipeline dialect return.";
  let description = [{
    The "return" operation represents a terminator of a `pipeline.pipeline`.
  }];

  let hasVerifier = 1;
  let arguments = (ins Variadic<AnyType>:$outputs, I1:$valid);
  let builders = [OpBuilder<(ins), [{ return; }]>];
  let assemblyFormat = [{ ($outputs^)? `valid` $valid attr-dict (`:` type($outputs)^)? }];
}

#endif // PIPELINE_OPS
