# 'datapath' Dialect Rationale

This document outlines the rationale of the datapath dialect, a dialect used for the efficient construction of arithmetic circuits.

[TOC]

## Introduction to the `datapath` dialect
Datapath design refers to the process of lowering arithmetic circuits to gate-level designs. When lowering arithmetic operations such as `comb.add` and `comb.mul` to gate-level implementations there are many component-level implementations to choose from. At the core of efficient gate-level implementations are a collection of circuit constructs that are composed to perform an arithmetic
operation on bitvectors. For example, in computer arithmetic a `comb.mul` would be comprised of three steps:
1. Partial Product generation
2. Compressor tree reduction to carry-save format
3. Carry-propagate adder

Decomposing arithmetic operators into their efficient sub-circuits creates opportunities to optimise and fuse arithmetic operations together.
[Reto Zimmerman's work](https://iis-people.ee.ethz.ch/~zimmi/publications/sop_synthesis.pdf) provides a good overview of some of the optimisations 
that the `datapath` dialect can expose. The core idea is to avoid inserting expensive carry-propagate adders and to leverage carry-save representations. 
The `datapath` operations themselves should be thought of more as generators than concretelty specified mathematical operations. 

## Intended Usage
The `datapath` dialect is designed to be used as part of a synthesis flow, lowering a subset of `comb` operations to `datapath` operations. Optimisation passes on the `datapath` dialect can merge operations to form larger datapath blocks, where carry-propagation is deferred. The `datapath` operations can then be lowered into a subset of `comb`, consisting of only gate-level operations. 

## Example
In a simple example, we can fold a*b+c using the datapath dialect to remove a carry-propagate adder:
```mlir
%0 = comb.mul %a, %b : i4
%1 = comb.add %0, %c : i4
```
Which is equivalent to:
```mlir
%0:4 = datapath.pp %a, %b : 4 x i4
%1:2 = datapath.compress %0#0, %0#1, %0#2, %0#3, %c : 5 x i4 -> (i4, i4)
%2 = comb.add %1#0, %1#1 : i4
```
