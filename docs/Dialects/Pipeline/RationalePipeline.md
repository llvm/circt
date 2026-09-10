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

Using e.g. the `pipeline-schedule-linear` pass, a pipeline may be scheduled wrt.
an operator library denoting the latency of each operation. The result of a scheduling
problem is the movement of operations to specific blocks.
Each block represents a pipeline stage, with `pipeline.stage` operations being
stage-terminating operations that determine the order of stages.

At this level, the semantics of the pipeline are that **any SSA def-use edge that
crosses a stage is a pipeline register**. To prevent cross-block canonicalization
to occur post-scheduling, these def-use edges must be expressed using the
`pipeline.src` operation, which is used to refer a value from within the basic
block that uses it, but which is defined in _any_ prior, dominating stage (block).  
Given these relaxed semantics, this level of abstraction is suitable for pipeline
retiming. Operations may be moved from one stage to another, or new blocks may be
inserted between existing blocks, without changing the semantics of the pipeline.
The only requirement is that def-use edges wrt. the order of stages are preserved.
For expressing semantics about multi-cycle latencies of SSA, please refer to the
`multicycle operations` segment below.  


```mlir
%out = pipeline.scheduled(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
^bb0(%a0 : i32, %a1: i32, %go : i1):
  %add0 = comb.add %a0, %a1 : i32
  pipeline.stage ^bb1 enable %go

^bb1:
  // %add0, %a0 is referenced in this stage via. dominance, w/ pipeline.src
  // preventing cross-block canonicalization.
  %add0_bb1 = pipeline.src %add0 : i32
  %a0_bb1 = pipeline.src %a0 : i32
  %add1 = comb.add %add0_bb1, %a0_bb1 : i32
  pipeline.stage ^bb2 enable %go

^bb2:
  %add0_bb2 = pipeline.src %add0 : i32
  %add1_bb2 = pipeline.src %add1 : i32
  %add2 = comb.add %add1_bb2, %add0_bb2 : i32 // %add0 crosses multiple stages.
  pipeline.return %add2 enable %go : i32   // %go crosses multiple stages
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

## A note on constants
Constants (defined as all operations which the `OpTrait::ConstantLike` trait) are
special cases in the pipeline dialect. These are allowed to be used anywhere
within the pipeline, and are not subject to the SSA def-use edge restrictions
described above. By doing so, we allow for constant canonicalizers to run,
which may apply regardless of where a constant is used within the pipeline.  
The result of this is that constant-like operations will be moved to the
entry stage of the pipeline.  
In the `pipeline-to-hw` pass, in case the user selects to perform `outline`d
lowering, constants will be **copied** into the stages which reference them.

## Multicycle operations
Oftentimes, we may have operations which take multiple cycles to complete within
a pipeline. Support for this is provided by the `pipeline.latency` operation.
The main purpose of this operation is to provide a way to inform the
register materialization pass to pass values through stages _without_ registering them.

Currently, all return values have an identical latency. This is an
arbitrary restriction, and may be lifted in the future if needed.

As an example pipeline:
```mlir
^bb1:
...
%out = pipeline.latency 2 -> (i32) {
  %dl1 = seq.compreg %in : i32
  %dl2 = seq.compreg %dl1 : i32
  pipeline.latency.return %dl2 : i32
}
pipeline.stage ^bb2

^bb2:
// It is illegal to reference %out here
pipeline.stage ^bb3


^bb3:
// It is legal to reference %out here
pipeline.stage ^bb4

^bb4:
// It is legal to reference %out here. This will also imply a register
// between stage bb3 and bb4.
%out_bb4 = pipeline.src %out : i32
foo.bar %out_bb4 : i32
```

which will register materialize to:
```mlir
^bb1:
...
%out = pipeline.latency 2 -> (i32) {
  %dl1 = seq.compreg %in : i32
  %dl2 = seq.compreg %dl1 : i32
  pipeline.latency.return %dl2 : i32
}
pipeline.stage ^bb2 pass(%out : i32)    // %out is passed through

^bb2(%out_s2 : i32):
pipeline.stage ^bb3 pass(%out_s2 : i32) // %out is passed through


^bb3(%out_s3 : i32):
pipeline.stage ^bb4 regs(%out_s3 : i32) // %out is registered

^bb4(%out_s4 : i32):
foo.bar %out_s4 : i32
```

## Non-stallable Pipeline Stages

**Note:** the following is only valid for pipelines with a stall signal.

An option of the Pipeline abstraction presented in this dialect is the ability
to have _non-stallable stages_ (NS). NS stages are used wherever a pipeline
access resources that are not able to stop on a dime, and thus require a fixed
amount of cycles to complete. 

Non-stallable stages are marked as an attribute of the pipeline operations,
wherein a bitvector is provided (by the user) to indicate which stage(s) are
non-stallable.

To see how non-stallable stages are implemented, consider the following. For
every stage, we define two signals - `S_{N,en}` is the signal that indicates
that the stage currently has valid contents (i.e. not a bubble). `S_{N,valid}`
is the signal that is used as a clock-enable for the output registers of a
stage.

Stages can be grouped into three distinct types based on how their valid signal
is defined: stallable stages, non-stallable stages and runoff stages.

<img title="Stage control" src="includes/img/Pipeline/stage_control.png"/>

1. Stallable stages are any stages which appear **before** the first
   non-stallable stage in the pipeline.
2. Non-stallable stages are the stages explicitly marked as non-stallable by the
   user.
3. Runoff stages and stages that appear **after** (and by extension, **between**
   non-stallable stages). Runoff stages consider their own enablement wrt. the
   stall signal, as well as the enablement of the **last non-stallable
   register** (LNS) wrt. the runoff stage's position in the pipeline.

The purpose of the runoff stages is to ensure that they are able to pass through
as many pipeline cycles as there are upstream non-stallable stages, such that
the contents of the non-stallable stages is not discarded.  
An important implication of this is that pipelines with non-stallable stages
**must** be connected to some buffer mechanism that is able to hold as many
pipeline output value cycles as there are non-stallable stages in the pipeline.

As an example, the following 6 stage pipeline will have the following valid
signals: <img title="Stage control" src="includes/img/Pipeline/ns_ex1.png"/>


### Example 1:
In this example, we have two NS stages followed by three runoff stages:

<img title="Stage control" src="includes/img/Pipeline/ns_ex2.png"/>

In this example we see that, as expected, two cycles are output from the
pipeline after the stall signal goes high, corresponding to the two NS stages.

### Example 2:
In this example, we have two NS stages, then one runoff stage, then one NS
stage, and finally one runoff stage:

<img title="Stage control" src="includes/img/Pipeline/ns_ex3.png"/>

In this example we see that, as expected, three cycles are output from the
pipeline after the stall signal goes high, corresponding to the three NS stages.
