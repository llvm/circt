# Symbol and Inner Symbol Rationale

This document describes various design points of the major CIRCT dialects 
relating to the use of symbols and the introduction of inner symbols and 
related types.  This follows in the spirit of other 
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

Verilog and FIRRTL have, from a software compiler perspective, an unusual 
number of nameable entities which can be referred to non-locally.  These entities 
have deep nesting in the code structures.  The requirements of dealing with 
these entities and references entails more complexity than provided by MLIR's 
symbols and symbol tables.  Several CIRCT dialects, therefore, share a common 
supplemental mechanism called "Inner Symbols" to manage these requirements.  
Inner Symbols necessarily deviate from MLIR nested symbol tables to enable 
representation of the behavior of Verilog and FIRRTL.

## Use of MLIR symbols

MLIR symbols are directly used for items in the global scope.  This primarily 
consists of `hw` or `firrtl` modules, though other entities, such as `sv` 
interfaces and bind statements and `firrtl` non-local anchors, also share this 
space.  Modules and instances of them are well suited to MLIR symbols.  They 
are analogous in scoping and structure to functions and call instructions.  The 
top-level `builtin.module` or `firrtl.circuit` operations define a symbol table, and all 
modules contained define symbols, with instances referring by symbol to their 
instantiated module.

## Inner Symbol

Within a `firrtl` or `hw` module, many entities may exist which can be referenced 
outside the module.  Operations and ports (and memory ports), need to define 
symbol-like data to allow forming non-SSA linkage between disparate elements.  
To accomplish this, an attribute named `inner_sym` is attached, providing a 
scoped symbol-like name to the element.  An operation with an `inner_sym`
resides in arbitrarily-nested regions of a region that defines an
`InnerSymbolTable`.  `InnerSymbolTable` operations can themselves be arbitrarily
nested, and thus must reside within either an `InnerRefNamespace` or `InnerSymbolTable`.
Ops which implement `InnerSymbolTables` must themselves also have an MLIR symbol
or `inner_sym`, as appropriate, so their scope can be referenced.
The `inner_sym` attribute must be an `InnerSymAttr` which defines the inner
symbols attached to the operation and its fields. Operations containing an inner
symbol must implement the `InnerSymbol` interface.

Inner Symbols are different from normal symbols due to MLIR symbol table 
resolution rules.  Specifically, normal symbols are resolved by first going up 
to the closest parent symbol table and resolving down from there (recursing 
back down for nested symbol paths). Conversely, Inner symbol symbol resolution
is rooted at some encapsulating `InnerRefNamespace`, which defines the top-level
symbol table hierarchy. In FIRRTL and HW, modules define a symbol in a
`firrtl.circuit` or `builtin.module` symbol table, which is then interpreted as an
`InnerRefNamespace`.  For instances to be able to resolve the 
modules they instantiate, the symbol use in an instance must resolve in the 
top-level `InnerRefNamespace`.  If a module were a symbol table, instances resolving a 
symbol would start from their own module, never seeing other modules (since 
resolution would start in the parent module of the instance and be unable to go 
to the global scope).  The second problem arises from nesting. MLIR Symbol 
defining operations must be immediate children of a symbol table.  FIRRTL and HW/SV 
operations which define an `inner_sym` are grandchildren, at least, of a symbol 
table and may be much further nested.  Lastly, ports need to define `inner_sym`, 
something not allowed by normal symbols.

Given that all `InnerSym` references are rooted at some top-level `InnerRefNamespace`,
inner symbols are not able to be resolved as relative references (i.e. if a
reference occurs somewhere within a set of nested `InnerSymbolTable` operations,
and a relative reference is used to reference e.g. a cousin symbol).  
This design has been chosen to make inner symbol resolution both more robust
as well as simpler to implement, based on an observation that relative references
in languages are usually a human convenience, and not necessarily a good IR
construct. In general, front-ends will generate fully qualified names early in
the compilation process suitable for use in the IR. And if references truly are
relative, then using a dataflow (SSA) representation is more appropriate.
Specifically, references which are relative to an instance hierarchy are dynamically scoped, which is a different implementation than symbols, which are lexically scoped.

## Inner Symbol Reference Attribute

An attribute `InnerRefAttr` is provided to encapsulate references to inner 
symbols.  This attribute stores the path to an inner symbol through the
`InnerRefNamespace` symbol table hierarchy - this path being a combination of
[`InnerSymbolTable`-operation symbols..., target inner symbol]. The path must be non-empty.  
`InnerRefAttr` may also refer to top-level symbols (i.e. a single-element path containing only the target).
In this case, it is legal to refer to MLIR symbols residing at the top-level in
the `InnerRefNamespace`.

For `firrtl` and `hw` usage, this typically amounts to two levels of nesting,
e.g. `@<module>::@<target>`.  This provides a uniform type for storing and
manipulating references to inner symbols.
An `InnerRefAttr` resolves in an `InnerRefNamespace`.

Operations using `InnerRefAttr` should implement the `verifyInnerRefs` method
of `InnerRefUserOpInterface` to verify these references efficiently.

## Traits and Classes

### InnerSymbolTable

Similar to MLIR's `SymbolTable`, `InnerSymbolTable` is both a trait and a class.

#### Trait

The trait is used by Operations to define a new scope for inner symbols
contained within.  These operations must have either the `Symbol` trait or
implement the `InnerSymbolOpInterface` and be
immediate children of either an `InnerRefNamespace` or another `InnerSymbolTable` operation.
Non-`InnerSymbolTable` Operations must use the `InnerSymbol` interface to provide
a symbol, regardless of presence of attributes.

#### Class

The class is used either manually constructed or as an analysis to track and
resolve inner symbols within an operation with the trait.

The class is also used in verification.

### InnerSymbolTableCollection

This class is used to construct the inner symbol tables
for all `InnerSymbolTable`s (e.g., a Module) within an `InnerRefNamespace`
(e.g., a circuit), either on-demand or eagerly in parallel.

`InnerSymbolTableCollection` is not publicly available, and only used as a
utility for `InnerRefNamespace`.


### InnerRefNamespace

This is also both a class and a trait.

#### Class


Combines `InnerSymbolTableCollection` with a `SymbolTable` for resolution of
`InnerRefAttr`s, primarily used during verification as argument to the
`verifyInnerRefs` hook which operations may use for more efficient checking of
`InnerRefAttr`s.

#### Trait

The `InnerRefNamespace` trait is used by Operations to define a new scope for
`InnerSymbolTable`s. Presently the only user is `CircuitOp`.

## Cost

Inner symbols are more costly than normal symbols, precisely from the 
relaxation of MLIR symbol constraints.  Since nested regions and nested
`InnerSymbolTable`s are allowed, finding all operations defining an `inner_sym`
requires a recursive IR scan.  
Verification is likewise trickier, partly due the significant increase in 
non-local references.

For this reason, verification is driven as a trait verifier on
`InnerRefNamespace` which constructs and verifies `InnerSymbolTable`s in
parallel, and uses these to conduct a per-top-level-`InnerSymbolTable` parallelized walk
to verify references by calling the `verifyInnerRefs` hook on
`InnerRefUserOpInterface` operations.

## Common Use

The most common use for `InnerRefAttr`s are to build paths through the instantiation 
graph to use a subset of the instances of an entity in some way.  This may 
be reading values via SystemVerilog's cross-module references (XMRs),
specifying SV bind constraints, 
specifying placement constraints, or representing non-local attributes (FIRRTL).

The common element for building paths of instances through the instantiation 
graph is with a `NameRefArrayAttr` attribute.  This is used, for example, by 
`hw.GlobalRefOp` and `firrtl.hierpath`.
