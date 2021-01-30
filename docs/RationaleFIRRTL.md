# FIRRTL Dialect Rationale

This document describes various design points of the FIRRTL dialect, why it is
the way it is, and current status and progress.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

[The FIRRTL project](https://github.com/chipsalliance/firrtl) is an existing
open source compiler infrastructure used by the Chisel framework to lower ".fir"
files to Verilog.  It provides a number of useful compiler passes and
infrastructure that allows the development of domain specific passes.  The
FIRRTL project includes a [well documented IR
specification](https://github.com/chipsalliance/firrtl/blob/master/spec/spec.pdf)
that explains the semantics of its IR, an [Antlr
grammar](https://github.com/chipsalliance/firrtl/blob/master/src/main/antlr4/FIRRTL.g4)
includes some extensions beyond it, and a compiler implemented in Scala which we
refer to as the _Scala FIRRTL Compiler_ (SFC).

The FIRRTL dialect in CIRCT is designed to follow the FIRRTL IR very closely,
but does diverge for a variety of reasons.  Here we capture some of those
reasons and why.  They generally boil down to simplifying compiler
implementation and take advantages of the properties of MLIR's SSA form.
This document generally assumes that you've read and have a basic grasp of the
IR spec, and it can be occasionally helpful to refer to the grammar.

## Status

The FIRRTL dialect and FIR parser is a generally complete implementation of the
FIRRTL specification and is actively maintained, tracking new enhancements. The
FIRRTL dialect supports some undocumented features and the "CHIRRTL" flavor of
FIRRTL IR that is produced from Chisel.

There are some exceptions:

1) We don't support the `'raw string'` syntax for strings.
2) We don't support the `Fixed` types for fixed point numbers, and some
   primitives associated with them.
3) We don't support `Interval` types

Some of these may be research efforts that didn't gain broad adoption, in which
case we don't want to support them.  However, if there is a good reason and a
community that would benefit from adding support for these, we can do so.

## Type system

### Not using standard types

At one point we tried to use the integer types in the standard dialect, like
`si42` instead of `!firrtl.sint<42>`, but we backed away from this.  While it
originally seemed appealing to use those types, FIRRTL
operations generally need to work with "unknown width" integer types (i.e.
`!firrtl.sint`).

Having the known width and unknown width types implemented with two different
C++ classes was awkward, led to casting bugs, and prevented having a
`FIRRTLType` class that unified all the FIRRTL dialect types.

### Canonicalized Flip Types

One significant difference between the CIRCT and Scala implementation of FIRRTL
is that the CIRCT implementation of the FIRRTL type system always canonicalizes
flip types, according to the following rules:

1) `flip(flip(x))` == `x`.
2) `flip(analog(x))` == `analog(x)` since analog types are implicitly
    bidirectional.
3) `flip(bundle(a,b,c,d))` == `bundle(flip(a), flip(b), flip(c), flip(d))` when
   the bundle has non-passive type or contains an analog type.  This forces the
   flip into the subelements, where it recursively merges with the non-passive
   subelements and analogs.
4) `flip(vector(a, n))` == `vector(flip(a), n)` when the vector has non-passive
   type or analogs.  This forces the flip into the element type, generally
   canceling it out.
5) `bundle(flip(a), flip(b), flip(c), flip(d))` == `flip(bundle(a, b, c, d)`.
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
3) canonicalization may make types equivalent that the SFC views as different

As an example of (3), consider the following module:

```
module Foo:
  output a: { flip a: UInt<1> }
  output b: { a: UInt<1> }
  b <= a
```

The connection `b <= a` _is illegal_ in the SFC due to a type mismatch. However,
type canonicalization in CIRCT converts the above module to the following
module:

```
module Foo:
  input a: { a: UInt<1> }
  output b: { a: UInt<1> }
  b <= a
```

Here, the connection `b <= a` _is legal_ in the SFC. Consequently, type
canonicalization may result in us accepting more circuits than the SFC.

Conversely, the SFC views the following two modules as exactly the same (but
with different names):

```
module Foo:
  input a: { a: UInt<1> }
  output b: { a: UInt<1> }
  b <= a
module Bar:
  output a: { flip a: UInt<1> }
  input b: { flip a: UInt<1> }
  a <= b
```

**Note**: the left-hand-side and right-hand-side of `b <= a` must change to `a
<= b` or the SFC will reject the circuit. CIRCT type canonicalization does not
extend to reversing connection directions and this requires relaxing what it
means about left-hand-side and right-hand-side connections. CIRCT consequently
accepts all permutations of `a <= b` and types which canonicalize to the same
representation.

## Operations

### Multiple result `firrtl.instance` operation

The FIRRTL spec describes instances as returning a bundle type, where each
element of the bundle corresponds to one of the ports of the module being
instanced.  This makes sense in the Scala FIRRTL implementation, given that it
does not support multiple ports.

The MLIR FIRRTL dialect takes a different approach, having each element of the
bundle result turn into its own distinct result on the `firrtl.instance`
operation.  This is made possible by MLIR's robust support for multiple value
operands, and makes the IR much easier to analyze and work with.

### Module bodies require def-before-use dominance instead of allowing graphs

MLIR allows regions with arbitrary graphs in their bodies, and this is used by
the RTL dialect to allow direct expression of cyclic graphs etc.  While this
makes sense for hardware in general, the FIRRTL dialect is intended to be a
pragmatic infrastructure focused on lowering of Chisel code to the RTL dialect,
it isn't intended to be a "generally useful IR for hardware".

We recommend that non-Chisel frontends target the RTL dialect, or a higher level
dialect of their own creation that lowers to RTL as appropriate.

### `input` and `output` Module Ports

The FIRRTL specification describes two kinds of ports: `input` and `output`.
In the `firrtl.module` declaration we don't track this distinction, instead we
just represent `output` ports as having a flipped type compared to its
specification.

**Rationale:**: This simplifies the IR and provides a more canonical
representation of the same concept.  This also composes with flip type
canonicalization nicely.

The FIRRTL specification uses the same type for module ports and instance ports.
Consequently, the FIRRTL specification relies on "flow" to set the root
directionality of a module port vs. an instance port. Computation of "flow"
requires tracking the "kind" of a reference. For modules and instances this
differentiation happens via "port kind" and "instance kind".

E.g., in the following module, when inside `module Foo`, the SFC views the types
of `a` and `bar.a` as the same. The SFC then computed the "kind" of each
reference to infer that `bar.a` is "sink flow" (something l-value like) and `a`
is "source flow" (something r-value like).

```
module Bar:
  input a: UInt<1>
module Foo:
  input a: UInt<1>
  inst bar of Bar
```

We diverge from this in CIRCT and represent all information in the type. I.e.,
the type of `bar.a` inside module `Foo` is flipped:

```mlir
firrtl.circuit "Foo" {
  firrtl.module @Bar(%a: !firrtl.uint<1>) {
  }
  firrtl.module @Foo(%a: !firrtl.uint<1>) {
    %bar = firrtl.instance @Bar {name = "bar"} : !firrtl.flip<bundle<a: uint<1>>>
  }
}
```

**Note**: The SFC provides a similar public utility,
[`firrtl.Utils.module_type`](https://www.chisel-lang.org/api/firrtl/latest/firrtl/Utils$.html#module_type(m:firrtl.ir.DefModule):firrtl.ir.BundleType),
that returns a bundle representing a `firrtl.module`.  However, this uses the
inverse convention where `input` is flipped and `output` is unflipped.

### `firrtl.mem`

Unlike the SFC, the FIRRTL dialect represents each memory port as a distinct
result value of the `firrtl.mem` operation.  Also, the `firrtl.mem` node does
not allow zero port memories for simplicity.  Zero port memories are dropped
by the .fir file parser.

### More things are represented as primitives

We describe the `mux` and `validif` expressions as "primitives", whereas the IR
spec and grammar implement them as special kinds of expressions.

We do this to simplify the implementation: These expression
have the same structure as primitives, and modeling them as such allows reuse
of the parsing logic instead of duplication of grammar rules.

### `invalid` Invalidate Operation is an expression

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
