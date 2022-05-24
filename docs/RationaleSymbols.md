# Symbol and Inner Symbol Rationale

This document describes various design points of the major CIRCT dialects relating to the use of symbols and the introduction of inner symbols and related types.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

Verilog and FIRRTL have, from a software compiler perspective, an unusual number of namable entities which can be refered to non-locally.  These entities have deep nesting in the code structures.  The requirements of dealing with these entities and references entails a more complexity than provided by MLIR's symbols and symbol tables.  Several Circt dialects, therefor, share a common supplimental mechanism called "Inner Symbols" to manage these requirements.  Inner Symbols necessarily deviates from MLIR nested symbol tables to enable representation of the behavior of verilog and firrtl.

## Use of MLIR symbols

MLIR symbols are directly used for items in the global scope.  This primarily consists of hw or firrtl modules, though other entities, such as system verilog interfaces and firrtl non-local anchors and bind statements, also share this space.  Modules and instances of them are well suited to mlir symbols.  They are analogous in scoping and structure to functions and call instructions.  The top-level mlir module or firrtl circuit op define a symbol table and all modules contained define symbols, with instances refering by symbol to their instantiated module.

## Inner Symbol

Within a module (firrtl or hw), many entities may exist which can be referenced outside the module.

This interface describes an operation that may define an
    `inner_sym`.  An `inner_sym` operation resides 
    in arbitrarily-nested regions of a region that defines a
    `InnerSymbolTable`.
    Inner Symbols are different from normal symbols due to 
    MLIR symbol table resolution rules.  Specifically normal
    symbols are resolved by first going up to the closest
    parent symbol table and resolving from there (recursing
    down for complex symbol paths).  In FIRRTL and SV, modules
    define a symbol in a circuit or std.module symbol table.
    For instances to be able to resolve the modules they
    instantiate, the symbol use in an instance must resolve 
    in the top-level symbol table.  If a module were a
    symbol table, instances resolving a symbol would start from 
    their own module, never seeing other modules (since 
    resolution would start in the parent module of the 
    instance and be unable to go to the global scope).
    The second problem arises from nesting.  Symbols defining 
    ops must be immediate children of a symbol table.  FIRRTL
    and SV operations which define a inner_sym are grandchildren,
    at least, of a symbol table and may be much further nested.
    Lastly, ports need to define inner_sym, something not allowed
    by normal symbols.