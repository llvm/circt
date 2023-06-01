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
%out = pipeline.unscheduled(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
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
problem is the movement of operations to specific blocks.
Each block represents a pipeline stage, with `pipeline.stage` operations being
stage-terminating operations that determine the order of stages.

At this level, the semantics of the pipeline are that **any SSA def-use edge that
crosses a stage is a pipeline register**.  
Note that we also intend to add support for attaching multi-cycle latencies to
SSA values in the future, which will allow for more fine-grained control over
the registers in the pipeline.  
Given these relaxed semantics, this level of abstraction is suitable for pipeline
retiming. Operations may be moved from one stage to another, or new blocks may be
inserted between existing blocks, without changing the semantics of the pipeline.
The only requirement is that def-use edges wrt. the order of stages are preserved.

```mlir
%out = pipeline.scheduled(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
^bb0(%a0 : i32, %a1: i32, %go : i1):
  %add0 = comb.add %a0, %a1 : i32
  pipeline.stage ^bb1 enable %go

^bb1:
  %add1 = comb.add %add0, %a0 : i32 // %a0 is a block argument fed through a stage.
  pipeline.stage ^bb2 enable %go

^bb2:
  %add2 = comb.add %add1, %add0 : i32 // %add0 crosses multiple stages.
  pipeline.return %add2 enable %go : i32 // %go crosses multiple stages
}
```

### Phase 3: Register materialized

Once the prior phase has been completed, pipeline registers must be materialized.  
This amounts to a dataflow analysis to check the phase 2 property of def-use edges
across pipeline stages, performed by the `pipeline-explicit-regs` pass.  

The result of this pass is the addition of block arguments to each block representing
a pipeline stage, block arguments which represent stage inputs. It is the
`pipeline.stage` operation which determines which values are registered and which
are passed through directly. The block arguments to each stage should thus "just"
be seen as wires feeding into the stage. 
In case a value was marked as multicycle, a value is passed through the `pass` list
in the stage terminating `pipeline.stage` operation. This indicates that the value
is to be passed through to the target stage without being registered.  
At this level, an invariant of the pipeline is that **any SSA value used within
a stage must be defined within the stage**, wherein that definition may be either
a block argument or as a result of another operation in the stage.  
The order of block arguments to a stage is, that register inputs from the
predecessor stage come first, followed by pass-through values. A verifier will
check that the signature of a stage block matches the predecessor `pipeline.stage`
operation.

```mlir
%0 = pipeline.scheduled(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
^bb0(%a0: i32, %a1: i32, %go: i1):
  %1 = comb.add %a0, %a1 : i32
  pipeline.stage ^bb1 regs (%1, %a0, %go) pass () enable %go

^bb1(%1_s0 : i32, %a0_s0 : i32, %go_s0 : i1):
  %2 = comb.add %1_s0, %a0_s0 : i32
  pipeline.stage ^bb2 regs (%2, %1_s0, %go_s0) pass () enable %go_s0

^bb2(%2_s1 : i32, %1_s1 : i32, %go_s1 : i1):
  %3 = comb.add %2_s1, %1_s1 : i32 // %1 from the entry stage is chained through both stage 1 and 2.
  pipeline.return %3 valid %go_s1 : i32 // and likewise with %go
}
```
