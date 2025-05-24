# `ice40` Dialect Rationale

This document describes various design points of the ice40 dialect, a 
dialect that matches the technology library used in the Lattice ICE40 family
of FPGAs.

## Introduction to the `ice40` Dialect

The `ice40` dialect provides a collection of operations that match the
documented operations for the ICE40 family of FPGAs.  The dialect hues closely
to the documentation.

The intent of the dialect is as a target of synthesis.

## Type System for `ice40` Dialect

The dialect uses signless 1-bit integers for all values.
