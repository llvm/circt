# HW Dialect Rationale

This document describes various design points of the HW dialects as well as
global perspective on the HW, Comb, and SV dialects, why
they are the way they are, and current status.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).  For more
information about the other dialects, please see the
[Comb Dialect Rationale](RationaleComb.md) and [SV Dialect
Rationale](RationaleSV.md).

- [HW Dialect Rationale](#hw-dialect-rationale)
  - [General Introduction](#general-introduction)
  - [Introduction to the `hw` Dialect](#introduction-to-the-hw-dialect)
  - [`hw` Type System](#hw-type-system)
  - [`hw` Operations](#hw-operations)
  - [Parameterized Modules](#parameterized-modules)
  - [Symbols and Visibility](#symbols-and-visibility)
  - [Future Directions](#future-directions)

## General Introduction

[SystemVerilog](https://en.wikipedia.org/wiki/SystemVerilog) is an industry
standard language for hardware design and verification, is known by a large
number of engineers who write it directly, and is
an important interchange format between EDA tools.  However, while it is
ubiquitous, SystemVerilog is not easy to generate or transform.  Furthermore, it
is non-trivial for compiler tools to generate high quality human readable
SystemVerilog.

The HW, Comb and SV dialects attempt to address these problems with several
major contributions: 

 1) By providing the `hw` dialect, which provides unifying structure and
    abstractions that are useful for a wide range of hardware modeling problems.
    Is allows other dialects to "mix in" with it to provide higher level
    functionality. `hw` is roughly akin to the "std" dialect in MLIR (but better
    currated).
 2) The `comb` dialect provides a common set of abstractions
    for combinational logic.  This dialect is designed to allow easy analysis
    and transformation.
 3) The `sv` dialect provides direct access to a wide variety
    of SystemVerilog constructs, including behavioral constructs, syntactic
    sugar constructs, and even idioms like `ifdef` blocks.
 4) By providing a high quality implementation and a number of useful compiler
    passes for analyzing and transforming these dialects, and a SystemVerilog
    emitter that generates pretty output.

The combination of these capabilities provides a useful suite of functionality
for compiler tools that want to generate high quality SystemVerilog.

## Introduction to the `hw` Dialect

The HW dialect defines a set of common functionality, such as `hw.module` and 
`hw.instance` for representing hardware modules, as well as common types and
attributes.   It is *not* designed to model SystemVerilog or any other hardware
design language directly, and doesn't contain combinational or sequential
operations.  Instead, it is designed to be a flexible and extensible substrate
that may be extended with higher level dialects mixed into it (like `sv`,
`comb`, `seq`, etc).

## `hw` Type System

TODO: Describe inout types.  Analogy to lvalues vs rvalues.  Array indices for
both forms.  Arrays, structs,
moving [UnpackedArray](https://github.com/llvm/circt/issues/389) to SV someday.

InOut types live at the SV dialect level and not the HW dialect level.  This
allows connects, wires and other syntactic constructs that aren't necessary for
combinational logic, but are nonetheless pretty useful when generating Verilog.


**Zero Bit Integers**

Certain operations support zero bit declarations:

 - The `hw.module` operation allows zero bit ports, since it supports an open
   type system.  They are dropped from Verilog emission.
 - Interface signals are allowed to be zero bits wide.  They are dropped from
   Verilog emission.

## `hw` Operations

TODO: Spotlight on module.  Allows arbitrary types for ports.

## Parameterized Modules

TODO

## Symbols and Visibility

Verilog has a broad notion of what can be named outside the context of its
declaration.  This is compounded by the many tools which have additional source
files which refer to verilog names (e.g. tcl files).  However, we do not want to
require that every wire, register, instance, localparam, port, etc which can be
named not be touched by passes.  We want only entities marked as public facing
to impede transformation.

For this reason, wires, registers, and instances may optionally define a symbol.
When the symbol is defined, the entity is considered part of the visible
interface and should be preserved in transformation.  Entities without a symbol
defined are considered private and may be changed by transformation.

**Implementation constraints**

Currently, MLIR restricts symbol resolution to looking in and downward through
any nested symbol tables when resolving symbols.  This assumption has
implications for verification, the pass manager, and threading.  Until symbol
references are more general, SV and HW dialects do not define symbol tables for
modules.  Therefore, wires, registers, and interfaces exist in the same
namespace as modules.  It is encouraged that one prefaces the names to avoid
conflict with modules.  The symbol names on these entities has no bearing on the
output verilog, each of these entities has a defined way to assign its name (SSA
value name for wires and regs, a non-optional string for instances).

As MLIR symbol support improves, it is desired to move to per-module symbol
tables and to unify names with symbol names.

**Ports**

Module ports are remotely namable entities in Verilog, but are not easily named
with symbols.  A suggested workaround is to attach a wire to a port and use its
symbol for remote references.

Instance ports have a similar problem.

## Future Directions

There are many possible future directions that we anticipate tackling, when and
if the need arises:

**More support for IR**

Many in the CIRCT community are interested in adding first-class support for
parametric modules -- similar but more general than SystemVerilog module
parameters.  It isn't clear yet whether this should be part of the HW dialect
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

Given the design of the HW/SV dialects, there is no need to resort to "lowest
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
