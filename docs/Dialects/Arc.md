# Arc Dialect

This dialect provides operations and types to represent state transfer functions in a circuit, enabling efficient scheduling of operations for simulation.

[TOC]


## Rationale

The main goal of the Arc dialect is to provide an intermediate representation of hardware designs that is optimized for simulation.
It transforms hardware descriptions from the HW, Seq, and Comb dialects into a form where all module hierarchies have been flattened, combinational logic is represented as callable "arcs" (state transfer functions), and sequential elements are modeled explicitly.

The Arc dialect is used by the *arcilator* simulation tool, which compiles Arc IR to a binary object via LLVM for fast simulation.


## Process and Coroutine Lowering

LLHD distinguishes two suspendable constructs.
An `llhd.process` defines procedural behavior inline in an `hw.module`; it runs once and may suspend execution at `llhd.wait` ops or terminate at `llhd.halt`.
An `llhd.coroutine` is a separately-defined suspendable subroutine, invoked at `llhd.call_coroutine` sites from inside a process or another coroutine; it terminates with `llhd.return`.

A process is, semantically, a coroutine defined inline in a module and invoked exactly once at its definition site.
Both bodies are SSACFG regions that are turned into a state machine driven by a *program counter* (PC), with values live across a suspension carried in *persistent state*.
Processes and coroutines therefore share a single lowering mechanism with only minor differences.

### Outlined Form

Both constructs are rewritten into a canonical outlined form: an `arc.coroutine.define` definition plus one or more call sites that re-enter it.
For a process, the call site is an `arc.coroutine.instance` placed in the enclosing `hw.module`.
For a coroutine, each `llhd.call_coroutine` becomes an `arc.coroutine.call` inside its parent coroutine's body.
After outlining, processes and coroutines are no longer distinguished.
Recursive coroutines are rejected during lowering.

### Program Counter

Every coroutine uses the same PC encoding:

| Name      | Value     | Meaning                                              |
|-----------|-----------|------------------------------------------------------|
| `START`   | `0`       | First entry; the body executes from its entry block. |
| resume    | `1..N`    | Resume at one of the body's suspension points.       |
| `RETURN`  | `MAX-1`   | The body returned normally; results are valid.       |
| `HALT`    | `MAX`     | The body halted; no further execution.               |

`START = 0` matches the zero-initialized layout of fresh persistent state and requires no special initialization at runtime.
Resume PCs are densely packed low integers, lowering to a single `switch` and keeping the per-coroutine PC width small.
`RETURN` and `HALT` are shared constants across all coroutines, so call sites dispatch on completion uniformly.

### Persistent State

The state carried across a suspension op corresponds to all the SSA values that are alive from that op into the resume block.
It is therefore not listed in the coroutine definition explicitly, but implied by its data and control flow structure.
The persistent state only becomes explicit when calling a coroutine, since the caller needs to decide how to re-enter a coroutine.

When lowering to a concrete implementation, the persisted state is a union of structs, with one variant per resume block capturing all the live values.
Multiple suspension ops targeting the same resume block share a variant.

Each resume block's first arguments must match the coroutine's function type.
These leading arguments are supplied fresh by the caller on each resumption and are therefore *not* part of the persistent state.
Any remaining block arguments hold the values passed as destination operands from the suspension op and *are* part of the persistent state.
The values captured into each variant are the SSA values that are live across the suspension ops into the resume block.

When a coroutine contains an `arc.coroutine.call`, the callee's state and PC are SSA values returned from the call.
If the call site is itself suspended -- i.e. the callee did not complete in a single eval -- those values are live across the parent's "I am inside a call" suspension point and are captured into the parent's variant like any other block argument.
State allocation is therefore compositional: the size of a coroutine's persistent state is the size of its own union plus, transitively, the size of each callee's persistent state at each call site.
Lowering proceeds bottom-up over the call graph so that callee state sizes are known by the time a parent is lowered.

### Instances and Wakeup

`arc.coroutine.instance` exists only inside `hw.module` bodies and represents the once-per-module entry into a top-level coroutine.
It guards entry into the coroutine with `if (now >= my_wakeup && resume_pc != HALT)`.
The referenced coroutine must produce an `i64` wakeup time as its last result, which is not returned as a result from the instance op.
The model's `next_wakeup` slot is reset to `UINT64_MAX` by `LowerState` at the top of every eval body.
Each `arc.coroutine.instance`, regardless of whether it dispatched, contributes its current stored wakeup to a min-reduction into that slot.
The driver reads the slot after eval to decide when next to call the model.


## Types

[include "Dialects/ArcTypes.md"]


## Attributes

[include "Dialects/ArcAttributes.md"]


## Enums

[include "Dialects/ArcEnums.md"]


## Interfaces

[include "Dialects/ArcInterfaces.md"]


## Operations

[include "Dialects/ArcOps.md"]


## Passes

[include "ArcPasses.md"]
