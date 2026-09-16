# PIR Dialect

This dialect provides operations and types that fully encode SystemVerilog Assertions (SVA), in a manner that provides interoperation with [Yosys's internal SVA support](https://yosyshq.readthedocs.io/projects/property-ir/en/latest/introduction.html).

[TOC]

## Rationale

Temporal logic is highly used across hardware verification to properly specify hardware designs. This can be found in many forms but the most common is SystemVerilog Assertions (SVA), which is a sub-language in SystemVerilog that allows for `properties` and `sequences`, i.e. predicates that require more than one cycle to evaluate. While this language exists and is mimicked in many front-ends, it's back-end support is often limited to either commercial tools or adhoc support for a minimal subset in open-source tools. 

This is where Property IR (PIR) comes in. The goal of PIR is to support formal verification flows, while decoupling front-end tasks (parsing, name resolution, etc.) from checker circuit synthesis and optimization. This is achieved by providing a unified representation for assertions, automata, and circuits. This means that PIR acts as a proper intermediate representation for manipulating and lowering SVA properties and sequences end-to-end. Unlike other dialects like `verif` or `ltl`, `pir` aims to fully encode and support all of SVA, as defined in the [IEEE specification](https://ieeexplore.ieee.org/document/10458102). Not only that, but PIR is designed and developed to share infrastructure and interoperate between CIRCT and Yosys. This allows CIRCT to gain access to the much more developed SVA support in Yosys while also allowing Yosys to gain access to all of CIRCT's front-ends and core optimizations. As such this dialect acts both as a an interface for interoperating with Yosys' SVA tooling, and as a full Abtract Syntax Tree for encoding SVA.

The property IR flow looks as follows:
```
  |------------------|
  | CIRCT Frontends  |
  |------------------|    |------------|
            |-----------> | CIRCT PIR  |
                          |------------|     |------------|
                                + ---------> | Yosys PIR  |
                          |------------|     |------------|        
                          | CIRCT Core |           |
                          |------------|           v
                                ^            |----------------------------| ---¬
                                |            | Yosys PIR automaton model  | <--|
                                |            |----------------------------| 
                                |                  |
                                |                  v
                                |            |----------------------------|
                                |------------| Yosys PIR Checker circuit  | 
                                             |----------------------------|       
```

## Enums

[include "Dialects/PIREnums.md"]

## Types

[include "Dialects/PIRTypes.md"]

## Operations

[include "Dialects/PIROps.md"]
