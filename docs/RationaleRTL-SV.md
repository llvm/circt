RTL and SV Dialect Rationale
============================

This document describes various design points of the RTL and SV dialects, why
they are the way they are, and current status.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

Introduction
============

[SystemVerilog](https://en.wikipedia.org/wiki/SystemVerilog) is an industry
standard language for hardware design and verification, is known by a large
number of engineers who write it directly, and is
an important interchange format between EDA tools.  However, while it is
ubiquitous, SystemVerilog is not easy to generate or transform.  Furthermore, it
is non-trivial for compiler tools to generate high quality human readable
SystemVerilog.

The RTL and SV dialects attempt to address these problems with three major
contributions: 

 1) By providing the RTL dialect, which contains a common set of abstractions
    for combinational logic.  This dialect is designed to allow easy analysis
    and transformation, is allows other dialects to "mix in" with it to provide
    higher level functionality.
 2) By providing the SV dialect, which provides direct access to a wide variety
    of SystemVerilog constructs, including behavioral constructs, syntactic
    sugar constructs, and even idioms like `ifdef` blocks.
 3) By providing a high quality implementation and a number of useful compiler
    passes for analyzing and transforming these dialects, and a SystemVerilog
    emitter that generates pretty output.

The combination of these capabilities provides a useful suite of functionality
for compiler tools that want to generate high quality SystemVerilog.

The RTL Dialect
===============

The RTL dialect is designed as a mid-level compiler IR for combinational logic.
It is *not* designed to model SystemVerilog or any other hardware design
language directly.  Instead, it is designed to be easy to analyze and transform,
and be a flexible and extensible substrate that may be extended with higher
level dialects mixed into it.

The RTL dialect defines a set of common functionality, such as `rtl.module` for
representing hardware modules, and operations like `rtl.add` and `rtl.mux` for
logic.

Type System
-----------

TODO: Describe inout types.  Analogy to lvalues vs rvalues.  Array indices for
both forms.  Arrays, structs,
moving [UnpackedArray](https://github.com/llvm/circt/issues/389) to SV someday.

Operations
----------

TODO: Spotlight on module.  Allows arbitrary types for ports.

TODO: Why is add variadic?  Why consistent operand types instead of allowing
implicit extensions?

** No Replication or ZExt Operators **

System Verilog 11.4.12.1 describes a replication operator, which replicates an
operand a constant N times.  We decided that this was redundant and just sugar
for the `rtl.concat` operator, so we just use `rtl.concat` (with the same 
operand) instead.

We also chose to not have a zero extension operator, since it is strictly
duplicative with `concat(zero, value)`.  The presence of such an operator adds a
lot of complexity to canonicalization patterns and doesn't add any expressive
power.  The `rtl.sext` operator exists because it efficiently models large
sign extensions which are common, and would require many operands if modeled as
a concat operator (in contrast, a zero extension always requires exactly one
zero value).

** Zero Bit Integers **

Combinatorial operations like add and multiply work on values of signless
standard integer types, e.g. `i42`, but they do not allow zero bit inputs.  This
design point is motivated by a couple of reasons:

1) The semantics of some operations (e.g. `rtl.sext`) do not have an obvious
   definition with a zero bit input.

1) Zero bit operations are useless for operations that are definable, and their
   presence makes the compiler more complicated.

On the second point, consider an example like `rtl.mux` which could allow zero
bit inputs and therefore produce zero bit results.  Allowing that as a design
point would require us to special case this in our cost models, and we would
have that optimizes it away.

By rejecting zero bit operations, we choose to put the complexity into the
lowering passes that generate the RTL dialect (e.g. LowerToRTL from FIRRTL).

Note that this decision only affects the core operations in the RTL dialect
itself - it is perfectly reasonable to define your operations and mix them into
other RTL constructs.  Also, certain other operations do support zero bit
declarations in limited ways:

 - The `rtl.module` operation allows zero bit ports, since it supports an open
   type system.  They are dropped from Verilog emission.
 - Interface signals are allowed to be zero bits wide.  They are dropped from
   Verilog emission.

Cost Model
----------

As a very general mid-level IR, it is important to define the principles that
canonicalizations and other general purpose transformations should optimize for.
There are often many different ways to represent a piece of logic in the IR, and
things will work better together if we keep the compiler consistent.

First, unlike something like LLVM IR, keep in mind that the RTL dialect is a
model of hardware -- each operation generally corresponds to an instance of
hardware, it is not an "instruction" that is executed by an imperative CPU.
As such, the primary concerns are area and latency, not "number of operations
executed".  As such, here are important concerns that general purpose
transformations should consider, ordered from most important to least important.

**Simple transformations are always profitable**

Many simple transformations are always a good thing, this includes:

1) Constant folding.
2) Simple strength reduction (e.g. divide to shift).
3) Common subexpression elimination.

These generally reduce the size of the IR in memory, can reduce the area of a
synthesized design, and often unblock secondary transformations.

**Reducing widths of non-trivial operations is always profitable**

It is always a good idea to reduce the width of non-trivial operands like add,
multiply, shift, divide, and, or (etc) since it produces less hardware and
enables other simplifications.  This is even true if it grows the IR size by
increasing the number of truncations and extensions in the IR.

That said, it is a bad idea to *duplicate* operations to reduce widths: for
example, it is better to have one large multiply with many users than to clone
it because one user only needs some of the output bits.

**Don't get overly tricky with divide and remainder**

Divide operations (particularly those with non-constant divisors) generate a lot
of hardware, and can have long latencies.  As such, it is a generally bad idea
to do anything to an individual instance of a divide that can increase its
latency (e.g. merging a narrow divide with a wider divide and using a subset of
the result bits).

**Constants and moving bits around is free**

Operations for constants, truncation, zero/sign extension, concatenation of 
signals, and other similar operations are considered free since they generally
do not synthesize into hardware.  All things being equal it is good to reduce
the number of instances of these (to reduce IR size and increase canonical form)
but it is ok to introduce more of these to improve on other metrics above.

The SV Dialect
==============

The SV dialect is one of the dialects that can be mixed into the RTL dialect,
providing
access to a range of syntactic and behavioral constructs.  The driving focus of
this dialect is to provide simple and predictable access to SystemVerilog
features: it is not focused primarily on being easy to analyze and transform.

The SV dialect is designed to build on top of the RTL dialect, so it does not
have its own operations for combinational logic, modules, or other base
functionality defined in the RTL dialect.

Type System
-----------

Like the RTL dialect, the SV dialect is designed to tolerate unknown types where
possible, allowing other dialects to mix in with it.  In addition to these
external types, and the types used by the RTL dialect, the SV dialect defines
types for SystemVerilog interfaces.

TODO: Describe interface types, modports, etc.

Operations
----------

Because the SV dialect aims to align with the textual nature of SystemVerilog,
many of the constructs in the SV dialect have an "AST" style of representation.
The major classes of operations you'll find are:

1) Statements like `sv.if`, `sv.ifdef`, `sv.always` and `sv.initial` that
   expose primary task-like operations and the behavioral model.
1) Procedural assignment operators, including the `sv.bpassign` and `sv.passign`
   operators that expose the blocking (`x = y`) and non-blocking (`x <= y`)
   procedural operators.
1) Directives like `sv.finish` and `sv.alias` and behavioral functions like
   `sv.fwrite`.
1) Access to verification constructs with `sv.assert`, `sv.assume`, and
   `sv.cover`.
1) Escape hatches that allow direct integration of textual expressions
   (`sv.textual_value`) and full statements (`sv.verbatim`).

These operations are designed to directly model the syntax of the SystemVerilog
language and to be easily printable by the ExportVerilog pass.  While there are
still many things in SystemVerilog that we cannot currently express in the SV
dialect, this design makes it easy to incrementally build out new capabilities
over time.

Cost Model
----------

The SV dialect is primarily designed for human consumption, not machines.  As
such, transformations should aim to reduce redundancy, eliminate useless
constructs (e.g. eliminate empty ifdef and if blocks), etc.

Future Directions
=================

There are many possible future directions that we anticipate tackling, when and
if the need arises:

**First Class Parametric IR**

Many in the CIRCT community are interested in adding first-class support for
parametric modules -- similar but more general than SystemVerilog module
parameters.  It isn't clear yet whether this should be part of the RTL dialect
or something higher level.

Separate from a "good" representation of parametric modules, the SV dialect
could grow direct support for representing the SystemVerilog functionality
in this space, including even things like "generate" blocks.

**EDA Tool-specific Subdialects**

The EDA tool ecosystem is filled with a wide range of tools with different
capabilities -- for example [see this
table](https://symbiflow.github.io/sv-tests-results/) for one compilation of
different systems and their capabilities.  As such, we expect that the day will
come where a frontend wants to generate fancy features for some modern systems,
but cannot afford to break compatibility with other ecosystem tools.

Given the design of the RTL/SV dialects, there is no need to resort to "lowest
common denominator" approach here: we can allow frontends to generate "fancy"
features, then use progressive lowering when dealing with tools that can't
handle them.  This can also allow IP providers to decide what flavor
of features they want to provide to their customers (or provide multiple
different choices).

**SystemVerilog Parser**

As the SV dialect grows out, it becomes natural to think about building a high
quality parser that reads SystemVerilog source code and parses it into the SV
dialect.  Such functionality could be very useful to help build tooling for the
SystemVerilog ecosystem.

Such a parser should follow clang-style principles of producing high quality
diagnostics, preserving source location information, being built as a library,
etc.
