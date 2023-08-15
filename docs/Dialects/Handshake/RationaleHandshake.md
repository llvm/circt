# Handshake Dialect Rationale

[TOC]

This document also explains in a high-level manner how different components are 
organized, the principles behind them and the conventions we followed.
The document assume that you have basic understanding of 
[asynchronous digital circuits](https://en.wikipedia.org/wiki/Asynchronous_circuit) 
at the behavioral level of abstraction.

## Principle

Handshake/dataflow IR describes independent, unsynchronized processes
communicating data through First-in First-out (FIFO) communication channels. 
This can be implemented in many ways, such as using synchronous logic, or with 
processors. 

## Choice of MLIR

MLIR is a common infrastructure to build your own specific IR to target 
different architectures and needs. We use MLIR because of its extensibility. We 
can apply the various transformations and optimization of MLIR on this IR. We 
can also lower the std MLIR produced by different frontends to Handshake IR. 

     TensorFlow     LLVM       Pytorch
          |           |           | 
     |-----------------------------------|    
     |   MLIR                            |
     |         -----------------         |
     |         | opt/transform |         |
     |         -----------------         |
     |                                   |
     |         -----------------         |
     |         | opt/transform |         |
     |         -----------------         |
     |                                   |
     |-----------------------------------|
        |        |        |             | 
       GPU      LLVM    Affine     **Dataflow**

## IR Representation

Simple Handshake IR snippet for an add function looks like this -
```
handshake.func @simple_addi(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
        %0 = addi %arg0, %arg1 : index
        handshake.return %0, %arg2 : index, none
}
```
It accepts two input streams (modeled as MLIR operands) and produces one 
output stream (modeled as an MLIR result).

## Conventions

The Handshake dialect adopts the following conventions for IR:
- The prefix for all Handshake types and operations are `handshake.`.

## Talks, Resources and Related Publications

- (10/2022) [Multi-Level Rewriting for Stream Processing to RTL compilation (M.Sc. Thesis) - Christian Ulmann](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/578713/1/Ulmann_Christian.pdf)
- (03/2022) [HLS from PyTorch to System Verilog with MLIR and CIRCT (Workshop paper) - Mike Urbach, Morten Borup Petersen](https://capra.cs.cornell.edu/latte22/paper/2.pdf)
- (01/2022) [A Dynamically Scheduled HLS Flow in MLIR (M.Sc. Thesis) - Morten Borup Petersen](https://infoscience.epfl.ch/record/292189)
- (06/2020) MLIR Handshake Dialect-[slides](https://drive.google.com/file/d/1UYQAfHrzcsdXUZ93bHPTPNwrscwx89M-/view?usp=sharing) by Stephen Neuendorffer (Xilinx) + Lana Josipović (EPFL) 

## Operation definitions

[include "Dialects/Handshake.md"]
