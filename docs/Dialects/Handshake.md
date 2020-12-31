# 'handshake' Dialect

[TOC]

This document also explains in a high-level manner how different components are 
organized and implemented in the code and gives steps to follow for extending 
them.
It targets the new aspiring hackers of CIRCT. Wherever necessary it also give 
the good introduction of MLIR/LLVM project's APIs. The document assume 
that you have basic understanding of [asynchronous digital circuits](https://en.wikipedia.org/wiki/Asynchronous_circuit) 
at the behavioral level abstraction.

## Principle

Handshake/dataflow IR is targeted to asynchronous digital circuits. An 
asynchronous circuit, is a sequential digital logic circuit which is not 
governed by a clock circuit or global clock signal. Instead it uses signals that
indicate completion of instructions and operations, specified by simple data 
transfer protocols such as two-phase and four-phase handshake protocol e.k.a. 
Non-Return-to-Zero (NRZ) and Return-to-Zero (RZ) encoding respectively.

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

Simple Handshake IR snippet for add function look like this -
```
handshake.func @simple_addi(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
        %0 = addi %arg0, %arg1 : index
        handshake.return %0, %arg2 : index, none
}
```
It contain simple_addi function which take two actual argument and return one 
actual argument.

## Conventions

The Handshake dialect adopts the following conventions for IR:
- The prefix for all Handshake types and operations are handshake..

## Types
To be written...

## Operations

In Handshake, instruction is a generalized concept; a Handshake function is just
a sequence of instructions. Declaring types, expressing computations, annotating 
result ids, expressing control flows and others are all in the form of 
instructions.

Operation documentation is written in each op’s Op Definition Spec using 
TableGen. A markdown version of the doc can be generated using `mlir-tblgen 
-gen-op-doc` and is attached in the [Operation definitions](#operation-definitions) 
section.

## Interfaces

MergeLikeOpInterface - Returns an operand range over the data signals being merged.
Useful verification purposes during lowering from other dialects.

The interface declaration is auto-generated from TableGen definitions 
included in HandshakeOps.td .

## Conversions

HandshakeBuilder - Handshake Builder class \
HandshakeFuncOpLowering - HandshakeTofoo lowering class

## Code organization 

### The dialect 

The code for the Handshake dialect resides in a few places-

  Public headers are placed in include/circt/Dialect/Handshake . \
  Libraries are placed in lib/Dialect/Handshake . \
  IR tests are placed in test/Dialect/Handshake . 

The dialect itself, including all types and ops, is in the CIRCTHandshakeOps 
library. 

### Op definitions

We use Op Definition Spec to define all Handshake ops. They are written in 
TableGen syntax and placed in HandshakeOps.td file in the header directory. 

This file defines the common classes and utilities used by various op 
definitions. It contains the TableGen Handshake dialect definition, Handshake 
versions, known extensions, and base op classes, etc.

### Dialect conversions 

The code for conversions from/to other dialects to/from the Handshake dialect 
also resides in a few places:

   From Standard dialect: headers are at include/circt/Conversion/StandardToHandshake ; \
   libraries are at lib/Conversion/StandardToHandshake . \
   tests are at test/Conversion/StandardToHandshake/ . 
   
   To FIRRTL dialect: headers are at include/circt/Conversion/HandshakeToFIRRTL ; \
   libraries are at lib/Conversion/HandshakeToFIRRTL . \
   tests are at test/Conversion/HandshakeToFIRRTL/ .

These dialect to dialect conversions have their dedicated libraries, 
CIRCTStandardToHandshake and CIRCTHandshakeToFIRRTL, respectively.

There are also common utilities when targeting Handshake from any dialect:
- /include/circt/Dialect/Handshake/Visitor.h contain visitors that make it easier 
to work with Handshake IR.

## Contributions

### Add a new op 
HandshakeOps.td is the filename for hosting the new op and 
Handshake_Op is the direct base class the newly defined op will derive 
from. 
Together with standard mlir-tblgen backends, we auto-generate all op classes.

The parser and printer should be placed in HandshakeOps.cpp with the following 
signatures:
```
static ParseResult parse<handshake-op-symbol>Op(OpAsmParser &parser, OperationState &result);
static void print(OpAsmPrinter &p, handshake::<handshake-op-symbol>Op op);
```
Verification should be provided for the new op to cover all the rules described
in the Handshake specification. Choosing the proper ODS types and attribute 
kinds, which can be found in HandshakeInterfaces.td , can help here. Still 
sometimes we need to manually write additional verification logic in 
handshakeOps.cpp in a function with the following signature:
```
static LogicalResult verify(handshake::<handshake-op-symbol>Op op);
```
Tests for the op’s custom assembly form and verification should be added to the
proper file in test/Dialect/Handshake/.

Also see MLIR's [Operation Definition Specification](https://mlir.llvm.org/docs/OpDefinitions/)
 
## Resources

MLIR Handshake Dialect-[slides](https://drive.google.com/file/d/1UYQAfHrzcsdXUZ93bHPTPNwrscwx89M-/view?usp=sharing) by Stephen Neuendorffer (Xilinx) + Lana Josipović (EPFL)

## Operations

[include "Dialects/HandshakeOps.md"]

