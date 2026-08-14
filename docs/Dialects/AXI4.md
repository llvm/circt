# AXI4 Dialect

This dialect provides operations and types to describe AXI4 connections and networks.

[TOC]

## Rationale

Designing AXI networks can be high-effort and fraught with the risk of creating networks that allow for invalid requests. This dialect aims to provide a high-level representation of AXI networks that uses types and verifiers to catch cases where an invalid or non-compliant network has been specified.

This dialect is designed to target three primary use-cases:

- Using the AXI dialect (or more practically some front-end that compiles to it) as a high-level description language; Once you have RTL specifications of your endpoints, you can use the dialect or front-end to describe your desired network and lower to RTL to generate an implementation.
- Using the AXI dialect to abstractly model an architecture; the dialect is designed to allow a network to be specified without concrete RTL sources. This allows the dialect to be used to specify a network early in the design process and validate that it meets sanity checks and validation criteria. A later lowering could also use such a description to generate an interconnect-only RTL implementation with top-level ports corresponding to the abstract ports given in the model.
- Extracting existing full RTL models to the AXI dialect; from an RTL system design with a sufficiently identifiable structure for AXI interfaces, it would theoretically be possible to produce a higher-level description raised to the AXI dialect. This allows users with existing RTL to benefit from the static analysis provided by the dialect, along with other tooling as it develops.

## Attributes

[include "Dialects/AXI4Attributes.md"]

## Enums

[include "Dialects/AXI4Enums.md"]

## Types

[include "Dialects/AXI4Types.md"]

## Operations

[include "Dialects/AXI4Ops.md"]
