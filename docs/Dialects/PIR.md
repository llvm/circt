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

### Structure of PIR Property
The general structure of a PIR block is a a `pir.cell`, which represents a property being asserted, assumed, cover, etc. which can have inputs that represent signals which it inherits from outside of the cell (by default it is isolated), and a top-level `pir.assert_property`, `pir.assume_property`, `pir.coover_property`, or `pir.restrict_property`. 

Unlike in other interpretations of SVA, PIR follows a strict typing isolation where there is no inherit hierarchy between the types. For example in `ltl`, `!ltl.property` > `!ltl.sequence` > `i1` so a boolean is a sequence which itself is a property. In `pir`, explicit casts or clocking operations are needed to move between types, e.g. `!pir.sequence` must be explicitly clocked to become a `!pir.clocked_sequence`, same for `!pir.property` and `!pir.clocked_property`, and of course `i1` to any clocked expression. This reduces the amount of analyses required, and potential inference, to retrieve information as a `!pir.clocked_property` will always be explicitly tied to a clock somewhere. This same reasoning is why all assertlike operations only accept `!pir.clocked_property` as their inputs.

Here's an example of how an SVA property gets encoded in PIR:
Given a SystemVerilog property to assert:
```sv
prop_ assert property (a ##1 b |-> always(c))
```
this becomes the following textual property IR (s-expressions): 
```lisp
(documnent
  (declare-input a)
  (declare-input b)
  (declare-input c)
  (declare prop 
    (clk-prop-overlapped-implication
      (clk-seq-concat
        (clk-seq-bool a)
        (clk-seq-bool b)
      )
      (clk-prop-always
        (clk-prop-bool c)
      )
    )
  )
  (assert-property prop)
)
```
which is equivalent to the following `pir` dialect:
```mlir
pir.cell @prop {} {
  %a, %b, %c = pir.input : i1, i1, i1
  %clk_s_a = pir.bool_to_clocked_seq %a
  %clk_s_b = pir.bool_to_clocked_seq %b
  %clk_p_c = pir.bool_to_clocked_prop %c

  %aconcatb = pir.concat %clk_s_a, %clk_s_b 
  %always_c = pir.always %clk_p_c
  %prop = pir.overlapped_implication %aconcatb, %always_c 

  pir.assert_property %prop : !pir.clocked_property
}
```

This is a simple example but shows the general structure of a PIR property. 
Assert-like operations function as terminators for a cell's region, meaning that each property should exist within its own cell.
Note that by default, type conversions will tie a property or sequence to the global clock, 
that is, the clock defined by the steps taken by a model checker. 
This is what distinguishes "clocking operations" with "type conversion operations". 
A neat addition is that the type signature in the pretty printed mlir assembly can be omitted in most 
cases since all of the operations only accept a single type of operand.  
The general structure is close enough to actual SVA to make a much cleaner interface than `ltl`, which should be used 
as a core dialect and not an interfacing dialect.  

## Enums

[include "Dialects/PIREnums.md"]

## Types

[include "Dialects/PIRTypes.md"]

## Operations

[include "Dialects/PIROps.md"]
