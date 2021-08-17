# FSM Dialect Rationale

This document describes various design points of the FSM dialect, why they are
the way they are, and current status.  This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

[Finite-state machine (FSM)](https://en.wikipedia.org/wiki/Finite-state_machine)
is an abstract machine that can be in exactly one of a finite number of states
at any given time.  The FSM can change from one state to another in response to
some inputs; the change from one state to another is called a transition.
Verification, Hardware IP block control, system state control, hardware design,
and software design have aspects that are succinctly described as some form of
FSM.  For integrated development purposes, late design-time choice, and
per-project choice, we want to encode system descriptions in an FSM form.  We
want a compiler to be able to manipulate, query, and generate code in multiple
domains, such as SW drivers, firmware, RTL hardware, and verification.

The FSM dialect in CIRCT is designed to provide a set of abstractions for FSM
with the following features:

1. Provide explicit and structural representations of states, transitions, and
internal variables of an FSM, allowing convenient analysis and transformation.
2. Provide a target-agnostic representation of FSM, allowing the state machine
to be instantiated and attached to other dialects from different domains.
3. By cooperating with two conversion passes, FSMToHW and FSMToStandard, allow
to lower the FSM abstraction into HW+Comb+SV and Standard+SCF+MemRef dialects
for the purposes of simulation, code generation, etc.

## Operations

**Two ways of instantiation**
A state machine is defined by an `fsm.machine` operation with a list of inputs
and outputs. The FSM dialect provides two ways to instantiate a state machine,
`fsm.hw_instance` operation and `fsm.instance`+`fsm.trigger` operations,  to
comply with the semantics of two different domains, HW and SW, respectively.

In HW IRs (such as HW+SV+Comb and FIRRTL), although an MLIR value is only
defined once in the IR, it is actually "driven" by its predecessors continuously
during the runtime and can "hold" different values at different moments.
However, in the world of SW IRs (such as Standard+SCF), we don't have such a
semantic -- SW IRs “run” sequentially.

To integrate the instance of an `fsm.machine` into both HW and SW IRs, we
introduce the two different ways of instantiation. A running example is shown
below:

```mlir
// State machine definition.
fsm.machine @foo(%in : 16) -> i16 {
  …
}

// HW instantiation.
hw.module @bar() {
  %in = …
  %out = fsm.hw_instance “foo_inst” @foo(%in) : i16 -> i16
}

// SW instantiation.
Func @qux() {
  %foo_inst = fsm.instance “foo_inst” @foo
  %in0 = …
  %out0 = fsm.trigger %foo_inst(%in0) : i16 -> i16
  …
  %in1 = …
  %out1 = fsm.trigger %foo_inst(%in1) : i16 -> i16
}
```

**Explicit state and transition representation**

Each state of an FSM is represented explicitly with an `fsm.state` operation.
Each `fsm.state` contains a list of `fsm.transition` operations representing the
transitions that can be triggered under the current state.  `fsm.state` also has
a symbol name that can be referred to by `fsm.transition`s as the next state.
The following MLIR code snippet shows a running example:

```mlir
fsm.machine @foo(%in : i16) -> i16 {
  fsm.state “IDLE” transitions {
    fsm.transition @BUSY …
    fsm.transition @IDLE …
  }
  …
}
```

In the contrast to the explicit representation, we could also represent FSM
states and transitions with MLIR *block*s and branchings, respectively.
Although this implicit representation may provide more flexibility, the explicit
one can generally make it easier to transform the IR and generate code to some
important targets, specifically, RTL designs in HW+Comb+SV dialects.

**Action regions**

TODO

**Internal variables**

To avoid *state explosion*, we introduce `fsm.variable` operation (similar to
the [extended state](https://en.wikipedia.org/wiki/UML_state_machine#Extended_states)
in UML state machine) to represent a variable that lives internally in an FSM
instance and can hold a value of any type.  The value of `fsm.variable` can be
updated with an `fsm.update` operation in any action regions:

```mlir
fsm.machine @foo(%in : i16) -> i16 {
  %bar = fsm.variable : i16
  fsm.state “IDLE” transitions {
    fsm.transition @BUSY action {
      fsm.update %bar, %in : i16
    }
  }
  …
}
```

**Guard regions**

Each `fsm.transition` has a `guard` region, which must have a terminator
returning a Boolean value indicating whether the transition is taken:

```mlir
fsm.machine @foo(%in : i16) -> i16 {
  %bar = fsm.variable : i16
  fsm.state “IDLE” transitions {
    fsm.transition @BUSY guard {
      %c0_i16 = constant 0 : i16
      %cond = cmpi eq, %bar, %c0_i16 : i16
      fsm.return %cond : i1
    } action {
      fsm.update %bar, %in : i16
    }
  }
  …
}
```

If a state has more than one transition, the guard of each transition is
evaluated in a top down order.  Therefore, the `guard` region should not contain
any side-effect operations, such as `fsm.update`.  Note that an empty `guard`
region is evaluated as true, which means the corresponding transition is always
taken.

**Where are “events”?**

TODO

## Lowerings

In the `FSMToHW` conversion pass, we adopt a binary state encoding style and an
(one-always block)[http://www.sunburst-design.com/papers/CummingsSNUG2019SV_FSM1.pdf]
SystemVerilog coding style to lower the state machine abstractions in FSM
dialect to RTL designs in HW+Comb+SV dialects. Therefore, an `fsm.machine` is
converted to a `hw.module` containing one `always @(posedge clk)` block, while
the `fsm.hw_instance` is naturally converted to a `hw.instance` instantiating
the generated `hw.module`.

In the contrast, the `FSMToStandard` pass converts `fsm.machine` to a `func`
representing the behavior of all combinational logics between two
`posedge clk`s, and each `fsm.trigger` is converted to a `call` to the generated
`func`.
