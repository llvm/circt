# LoopSchedule Dialect Rationale

This document describes various design points of the `loopschedule` dialect, why it is
the way it is, and current status. This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

The `loopschedule` dialect provides a collection of ops to represent software-like loops
after scheduling. There are currently two main kinds of loops that can be represented:
pipelined and sequential. Pipelined loops allow multiple iterations of the loop to be
in-flight at a time and have an associated initiation interval (II) to specify the number
of cycles between the start of successive loop iterations. In contrast, sequential loops
are guaranteed to only have one iteration in-flight at any given time.

A primary goal of the `loopschedule` dialect, as opposed to many other High-Level Synthesis
(HLS) representations, is to maintain the structure of loops after scheduling. As such, the
`loopschedule` ops are inspired by the `scf` and `affine` dialect ops.

## Status

Initial dialect addition, more documentation and rationale to come as ops are added.
