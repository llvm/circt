FIRRTL Dialect Rationale
========================

This document describes various design points of the FIRRTL dialect, why it is
the way it is, and current status and progress.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

Introduction
============

[The FIRRTL project](https://github.com/chipsalliance/firrtl) is an existing
open source compiler infrastructure used by the Chisel framework to lower ".fir"
files to Verilog.  It provides a number of useful compiler passes and
infrastructure that allows the development of domain specific passes.  The
FIRRTL project includes a [well documented IR specification](https://github.com/chipsalliance/firrtl/blob/master/spec/spec.pdf)
that explains the semantics of its IR and [Antlr 
grammar](https://github.com/chipsalliance/firrtl/blob/master/src/main/antlr4/FIRRTL.g4)
includes some extensions beyond it.

The FIRRTL dialect in CIRCT is designed to follow the FIRRTL IR very closely,
but does diverge for a variety of reasons.  Here we capture some of those
reasons and why.  They generally boil down to simplifying compiler
implementation and taking advantages of the properties of MLIR's SSA form.
This document generally assumes that you've read and have a basic grasp of the
IR spec, and it can be occasionally helpful to refer to the grammar.

Status
------

The FIRRTL dialect and FIR parser is a generally complete implementation of
the FIRRTL specification (including some undocumented features, and "CHIRRTL")
and is actively maintained, tracking new enhancements.

There are some exceptions:

1) We don't support the `'raw string'` syntax for strings.
2) We don't support the `Fixed` types for fixed point numbers, and some
   primitives associated with them.

Some of these may be research efforts that didn't gain broad adoption, in which
case we don't want to support them.  However, if there is a good reason and a
community that would benefit from adding support for these, we can do so.

Type system
===========

Not using standard types
------------------------

At one point we tried to use the integer types in the standard dialect, like
`si42` instead of `!firrtl.sint<42>`, but we backed away from this.  While it
originally seemed appealing to use those types, FIRRTL
operations generally need to work with "unknown width" integer types (i.e. 
`!firrtl.sint`).

Having the known width and unknown width types implemented with two different
C++ classes was awkward, led to casting bugs, and prevented having a
`FIRRTLType` class that unified all the FIRRTL dialect types.

Canonicalized Flip Types
------------------------

One significant difference between the CIRCT and Scala implementation of FIRRTL
is that the CIRCT implementation of the FIRRTL type system always canonicalizes
flip types, according to the following rules:

1) `flip(flip(x))` == `x`.
2) `flip(analog(x))` == `analog(x)` since analog types are implicitly
    bidirectional.
2) `flip(bundle(a,b,c,d))` == `bundle(flip(a), flip(b), flip(c), flip(d))` when
   the bundle has non-passive type or contains an analog type.  This forces the
   flip into the subelements, where it recursively merges with the non-passive
   subelements and analogs.
3) `flip(vector(a, n))` == `vector(flip(a), n)` when the vector has non-passive
   type or analogs.  This forces the flip into the element type, generally
   canceling it out.
4) `bundle(flip(a), flip(b), flip(c), flip(d))` == `flip(bundle(a, b, c, d)`.
   Due to the other rules, the operand to a flip must be a passive type, so the
   entire bundle will be passive, and rule #3 won't be recursively reinvoked.

The result of this is that FIRRTL types are guaranteed to have a canonical
representation, and can therefore be tested for pointer equality.  The
canonicalization of analog types means that both input and output ports of
analog type have the same representation.

As a further consequence of this, there are a few things to be aware of:

1) the `FlipType::get(x)` method may not return a flip type!
2) the notion of "flow" in the FIRRTL dialect is different from the notion
   described in the FIRRTL spec.

Operations
==========

`input` and `output` Module Ports
---------------------------------

The FIRRTL specification describes two kinds of ports: `input` and `output`.
In the `firrtl.module` declaration we don't track this distinction, instead we
just represent `output` ports as having a flipped type compared to its
specification.

**Rationale:**: This simplifies the IR and provides a more canonical
representation of the same concept.  This also composes with flip type
canonicalization nicely.

More things are represented as primitives
-----------------------------------------

We describe the `mux` and `validif` expressions as "primitives", whereas the IR
spec and grammar implement them as special kinds of expressions.

We do this to simplify the implementation: These expression
have the same structure as primitives, and modeling them as such allows reuse
of the parsing logic instead of duplication of grammar rules.

`invalid` Invalidate Operation is an expression
-----------------------------------------------

The FIRRTL spec describes an `x is invalid` statement that logically computes
an invalid value and connects it to `x` according to flow semantics.  This
behavior makes analysis and transformation a bit more complicated, because there
are now two things that perform connections: `firrtl.connect` and the
`x is invalid` operation.

To make things easier to reason about, we split the `x is invalid` operation
into two different ops: an `firrtl.invalidvalue` op that takes no operands
and returns an invalid value, and a standard `firrtl.connect` operation that
connects the invalid value to the destination (or a `firrtl.attach` for analog
values).  This has the same expressive power as the standard FIRRTL
representation but is easier to work with.
