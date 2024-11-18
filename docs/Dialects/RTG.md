# Random Test Generation (RTG) Rationale

This dialect provides types, operations, and interfaces modeling randomization
constructs for random test generation. Furthermore, it contains passes to
perform and analyze the represented randomization.

[TOC]

## Rationale

This dialect aims to provide a unified representation for randomized tests, more
precisely the parts of the tests that encode the randomization constructs (e.g.,
picking a random resource from a set of resources).
This means, this dialect is only useful in combination with at least one other
dialect that represents the actual test constructs, i.e., the parts that get
randomized. After all the randomization constructs are fully elaborated, the
resulting test will only consist of those dialects.

Examples for tests that can be randomized, and thus candidates for companion
dialects, are instruction sequences of any ISA (Instruction Set Architecture),
transaction sequences over a latency insensitive (ready-valid) channel, input
sequences to an FSM (Finite State Machine), transaction sequences for
potentially any other protocol. Albeit, the initial motivation are ISA tests and
will thus be the best supported, at least initially.

While it should be valid to add constructs to this dialect that are special to
any of the above mentioned use-cases, the dialect should generally be designed
such that all of them can be supported.

## Interfacing with the RTG dialect

The RTG dialect is only useful in combination with at least one other dialect
representing the test. Such a dialect has to be aware of the RTG dialect and
has to implement various interfaces (depending on which parts of the RTG
infrastructure it intends to use).

A test can be declared with the `rtg.test` operation which takes an
`!rtg.target` type attribute to specify requirements for the test to be able to
be generated and executed. In addition to the target having to provide an
`!rtg.target` typed value that is a refinement of the required target type, the
test can specify additional requirements using the `rtg.require` operation.

Targets can be defined using the `rtg.target` operation. Certain operations
are only allowed inside the target operation but not the test operation to
guarantee that certain resources (specifically contexts) cannot be generated
on-the-fly.

The RTG framework will match all tests with all targets that fulfill their
requirements. The user can specify which targets and tests should be included
in the test generation process and how many tests should be generated per
target or in total.

Currently, there are three concepts in the RTG dialect the companion dialects
can map their constructs to:

* *Instructions*: instructions/operations intended to be performed in series
  by the test, such as ISA instructions, or protocol transactions
* *Resources*: resources the instructions operate on, such as registers or
  memories in ISA tests, or channels in hardware transaction tests
* *Contexts*: the context or environment the instuctions are performed in,
  e.g., on which CPU and in which mode for ISA test.

To express their mapping, the companion dialects must implement the the
interfaces as follows.

**Instructions**

They implement `InstructionOpInterface`on all operations that represent an
instruction, transaction, or similar. Those operations are intended to be
statements and not define SSA values.

**Resources**

Operations that create, declare, define, or produce a handle to a resource
instance must implement the `ResourceOpInterface` interface and define at least
one SSA value of a type that implements the `ResourceTypeInterface` interface.
(TODO: maybe we don't actually need the `ResourceTypeInterface`?)

The `RegisterOpInterface` can be implemented in addition to the
`ResourceOpInterface` if the resource is a register to become supported by the
register allocation and assembly emission pass.

**Contexts**

The `ContextResourceType` is used to represent contexts. Instructions can be
placed in specific context using the `rtg.on_context` operation (Note: we might
promote this to a type interface in the future).
Operations that define contexts (i.e., create new contexts) must implement the
`ContextResourceOpInterface` interface.
Operations implementing the `ContextResourceOpInterface` interface are only
allowed inside the `rtg.target` operation.

## Randomization

This dialect aims to provide utilities that allow users to generate tests
anywhere on the spectrum from directed tests to (almost) fully random tests
(refer to *fuzzing*).

* *Constraints*: allow the user to specify constraints to avoid generating
  illegal or useless tests, e.g., by
  * allowing to specify a sequence of instructions that always have to be picked
    in exactly that order and form (see `rtg.sequence` operation)
  * dependencies between resources or resource usages
  * etc.
* *Probabilities and Biases*: allow certain tests (or parts of a test) to be
  picked more likely than others (can be seen as 'soft-constraints') (see
  `!rtg.bag` type and associated operations)
* *Enumerations*: allow to enumerate all tests that can be produced by the
  current randomness constraints, possibly in a way that places the more likely
  tests to occur earlier in the enumeration

(TODO: expand here once these things are built out)

Main IR constructs to introduce randomness:

* *Sets*: a set of elements of the same type; the usual set operations apply as
  well as uniformly at random picking one element of the set
* *Bags/Biased Sets*: a generalization of *sets* that allows one element to
  occur multiple times and thus make it more likely to be picked (i.e., models
  non-uniform distributions)

## Use-case-specific constructs

This section provides an overview of operations/types/interfaces added with the
intend to be only used for one (or a few) specific use-cases/test-targets.

### ISA Tests

* *Labels*: Handling of labels is added to the RTG dialect because they are common
  across ISAs. Leaving them to the ISA specific companion dialects would likely
  lead to frequent code duplication (once for each ISA).
* *Register Allocation Pass*
* *Assembly Emission Pass*

## Frontends

Any dialect or entry point is allowed to generate valid RTG IR with a companion
dialect. The dialect already comes with an extensive CAPI, Python Bindings, and
a small Python library that simplify usage over the pure Python Bindings.

## Backends

The RTG dialect does not have a backend itself. It is fully lowered by its
dialect transformation passes that perform the randomization. The result will be
IR consisting purely of the companion dialect and thus it is up to this
companion dialect to define any backends.

## Operations

[include "Dialects/RTGOps.md]

## Types

[include "Dialects/RTGTypes.md]

## Passes

[include "Dialects/RTGPasses.md]
