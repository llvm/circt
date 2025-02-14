# `blif` Dialect Rationale

This document describes various design points of the BLIF dialect, a dialect
matched to the Berkeley Logic Interchange Format.

## Introduction to the `blif` Dialect

The `blif` dialect provides a collection of operations that map to the
Berkeley Logic Interchange Format.  This format is used for some tools in
synthesis and place-and-route flows.

## Type System for `blif` Dialect

The `blif` dialect uses basic integer types for all operations.  There is no 
notion of signedness in this representation.  There are no parametric types.

## BLIF Operations

BLIF has very few operations.  Namely it has registers and truth-tables and 
library (external) versions of those ops.
