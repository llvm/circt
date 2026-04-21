# Intrinsics

Intrinsics provide an implementation-specific way to extend the FIRRTL language
with new operations.

[TOC]

## Motivation

Intrinsics provide a way to add functionality to FIRRTL without having to extend
the FIRRTL language. This allows a fast path for prototyping new operations to 
rapidly respond to output requirements.  Intrinsics maintain strict definitions
and type checking.

## Supported Intrinsics

[include "Dialects/FIRRTLIntrinsics.md.inc"]
