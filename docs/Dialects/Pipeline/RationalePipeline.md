# Pipeline Dialect Rationale

This document describes various design points of the `pipeline` dialect, why it is
the way it is, and current status. This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

## Pipeline Phases

A `pipeline.pipeline` operation can be used in a sequence of phases, each
of which incrementally transforms the pipeline from being unscheduled towards
being an RTL representation of a pipeline. Each phase is mutually exlusive,
meaning that the "phase-defining" operations
(`pipeline.ss, pipeline.ss.reg, pipeline.stage`) are not allowed to co-exist.

### Phase 1: Unscheduled

The highest-level phase that a pipeline may be in is the unscheduled phase.
In this case, the body of the pipeline simply consists of a feed-forward set of
operations representing a dataflow graph.

```mlir
%out = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
  ^bb0(%a0 : i32, %a1: i32, %g : i1):
    %add0 = comb.add %a0, %a1 : i32
    %add1 = comb.add %add0, %a0 : i32
    %add2 = comb.add %add1, %add0 : i32
    pipeline.return %add2 valid %s1_valid : i32
}
```

### Phase 2: Scheduled

Uisng e.g. the `pipeline-schedule-linear` pass, a pipeline may be scheduled wrt.
an operator library denoting the latency of each operation. The result of a scheduling
problem is the insertion of `pipeline.ss` operations into the pipeline body.
`pipeline.ss` are stage separating operations denoting the end of a stage and the
beginning of the next. The semantics are thus that **any SSA def-use edge that
crosses a stage boundary is a pipeline register**.  
Note that we also intend to add support for attaching multi-cycle latencies to
SSA values in the future, which will allow for more fine-grained control over
the registers in the pipeline.  
Given these semantics, this phase represents an abstraction for retiming a
pipeline, seeing as additional `pipeline.ss` operations may be inserted or 
moved around without changing the semantics of the computation, only the
latency characteristics of the pipeline.

```mlir
%out = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
^bb0(%a0 : i32, %a1: i32, %g : i1):
  %add0 = comb.add %a0, %a1 : i32

  %s0_valid = pipeline.ss enable %g
  %add1 = comb.add %add0, %a0 : i32 // %a0 is a block argument fed through a stage.

  %s1_valid = pipeline.ss enable %s0_valid
  %add2 = comb.add %add1, %add0 : i32 // %add0 crosses multiple stages.

  pipeline.return %add2 valid %s1_valid : i32
}
```

### Phase 3: Register materialized

Once the prior phase has been completed, pipeline registers must be materialized.  
This amounts to a dataflow analysis to check the phase 2 property of def-use edges
across pipeline stages, performed by the `pipeline-explicit-regs` pass.  
The result of this is the change of `pipeline.ss` to `pipeline.ss.reg` operations.

```mlir
%0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
^bb0(%a0: i32, %a1: i32, %g: i1):
  %1 = comb.add %a0, %a1 : i32

  %1_s0, %a0_s0, %valid = pipeline.ss.reg enable %g regs %1, %a0 : i32, i32
  %2 = comb.add %1_s0, %a0_s0 : i32

  %2_s1, %1_s1 %valid_3 = pipeline.ss.reg enable %valid regs %2, %1_s0 : i32, i32
  %3 = comb.add %2_s1, %1_s1 : i32 // %1 from the entry stage is chained through both stage 1 and 2.

  pipeline.return %3 valid %valid_3 : i32
}
```

### Phase 4: Staged

The final phase of a pipeline is the staged phase. In this stage, we break with the notion
of a dataflow graph, and instead make each pipeline stage explicit, performed by the
`pipeline-stagesep-to-stage` pass.
A `pipeline.stage` is an operation with an explicit enable signal, a set of
inputs and outputs.  The internal of a `pipeline.stage` is isolated from above, 
ensuring that stage internals can exclusively access values that have been fed
into the stage. The `pipeline.stage.return` operation determines which values 
are to be registered, and returned as the stage output.  

From here, a pipeline is ready to be lowered to a hardware representation.

```mlir
%0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
^bb0(%a0: i32, %a1: i32, %arg2: i1):
  %outputs:2, %valid = pipeline.stage ins %a0, %a1 enable %g : (i32, i32, i1) -> (i32, i32) {
  ^bb0(%arg3: i32, %arg4: i32, %arg6: i1):
    %2 = comb.add %arg3, %arg4 : i32
    pipeline.stage.return regs %2, %arg3 valid %arg6 : (i32, i32)
  }
  %outputs_2:2, %valid_3 = pipeline.stage ins %outputs#0, %outputs#1 enable %valid : (i32, i32) -> (i32, i32) {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
    %2 = comb.add %arg3, %arg4 : i32
    pipeline.stage.return regs %2, %arg3 valid %arg5 : (i32, i32)
  }
  %1 = comb.add %outputs_2#0, %outputs_2#1 : i32
  pipeline.return %1 valid %valid_3 : i32
}
```