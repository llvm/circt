# Symbol and Inner Symbol Rationale

This document describes various design points of the major CIRCT dialects 
relating to the use of symbols and the introduction of inner symbols and 
related types.  This follows in the spirit of other 
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

Verilog and FIRRTL have, from a software compiler perspective, an unusual 
number of namable entities which can be referred to non-locally.  These entities 
have deep nesting in the code structures.  The requirements of dealing with 
these entities and references entails a more complexity than provided by MLIR's 
symbols and symbol tables.  Several CIRCT dialects, therefor, share a common 
supplemental mechanism called "Inner Symbols" to manage these requirements.  
Inner Symbols necessarily deviates from MLIR nested symbol tables to enable 
representation of the behavior of verilog and FIRRTL.

## Use of MLIR symbols

MLIR symbols are directly used for items in the global scope.  This primarily 
consists of `hw` or `firrtl` modules, though other entities, such as `sv` 
interfaces and bind statements and `firrtl` non-local anchors, also share this 
space.  Modules and instances of them are well suited to MLIR symbols.  They 
are analogous in scoping and structure to functions and call instructions.  The 
top-level `mlir.module` or `firrtl.circuit` operations define a symbol table, and all 
modules contained define symbols, with instances referring by symbol to their 
instantiated module.

## Inner Symbol

Within a `firrtl` or `hw` module, many entities may exist which can be referenced 
outside the module.  Operations and ports (and memory ports), need to define 
symbol-like data to allow forming non-ssa linkage between disparate elements.  
To accomplish this, an attribute named `inner_sym` is attached a scoped 
symbol-like name to the element.  An operation with an `inner_sym` resides in 
arbitrarily-nested regions of a region that defines an `InnerSymbolTable` and
a `Symbol` .

Inner Symbols are different from normal symbols due to MLIR symbol table 
resolution rules.  Specifically normal symbols are resolved by first going up 
to the closest parent symbol table and resolving down from there (recursing 
back down for nested symbol paths).  In FIRRTL and SV, modules define a symbol in a 
`firrtl.circuit` or `std.module` symbol table.  For instances to be able to resolve the 
modules they instantiate, the symbol use in an instance must resolve in the 
top-level symbol table.  If a module were a symbol table, instances resolving a 
symbol would start from their own module, never seeing other modules (since 
resolution would start in the parent module of the instance and be unable to go 
to the global scope).  The second problem arises from nesting.  Symbols 
defining operations must be immediate children of a symbol table.  FIRRTL and SV 
operations which define an `inner_sym` are grandchildren, at least, of a symbol 
table and may be much further nested.  Lastly, ports need to define `inner_sym`, 
something not allowed by normal symbols.

## Inner Symbol Reference Attribute

An attribute `InnerRefAttr` is provided to encapsulated references to inner 
symbols.  This attribute stores the parent symbol and the inner symbol.  This 
provides a uniform type for storing and manipulating references to inner 
symbols.

## Cost

Inner symbols are more costly than normal symbols, precisely from the 
relaxation of MLIR symbol constraints.  Since nested regions are allowed, 
finding all operations defining an `inner_sym` requires a recursive ir scan.  
Verification is likewise trickier, partly due the significant increase in 
non-local references.

## Common Use

The most common use for `InnerRefAttr`s are to build paths through the instantiation 
graph to use a subset of the instances of an entity is some way.  This may 
be reading values via System Verilog's  cross-module references (XMRs),
specifying SV bind constraints, 
specifying placement constraints, or representing non-local attributes (FIRRTL).

The common element for building paths of instances through the instantiation 
graph is with a `NameRefArrayAttr` attribute.  This is used, for example, by 
`sv.GlobalRefOp` and `firrtl.NonLocalAnchor` .
