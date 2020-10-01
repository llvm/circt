# Elastic Silicon Interconnect: Concepts

## Overview

The Elastic Silicon Interconnect project is fundamentally an advanced silicon
(FPGA/ASIC) interconnect generator which raises the level of abstraction for
inter/intra/CPU communication. A hardware-centric type system enables strong
type safety. Latency insensitive (elastic) connections by default allows the
compiler to make optimization decisions. Essentially, the intent is to
separate/abstract the wire signaling layer from the message layer and allow
designers to not worry about the wire signaling layer. These traits enable
high-level reasoning which includes -- but is not limited to -- the following
features:

- Inter-language communication
- Type checking to reduce bugs at interface boundaries
- Correct-by-construction building of communication fabric (including clock domain crossings)
- Automated decision making about the physical signaling between modules
- Automated, typed software API generation which bridges over PCIe, network, or simulation
- Automated endianness conversions
- Automatic pipelining and floor planning to reduce timing closure pressure
- Compatibility between modules with different bandwidths (automatic gearboxing)
- Type and signal aware debuggers/monitors in communication fabric
- Common interface for board support packages
- Extensible services to support global resources (e.g. telemetry)

This set of rough documents provides a draft of the fundamental ESI concepts.
Early work will not have most of the features discussed herein. One should
read this documentation as more of a roadmap than current features. We will
do our best to add (and keep up to date) notes about implementation status.

## Terminology

Module interfaces are defined by a set of **typed ports** over which
**messages** are exchanged. **Message types** include parameterized sized
**integer**s, **fixed** point numbers, **float**ing point numbers, fixed size
**arrays**, **enum**s, **union**s, **structs**, and variably-sized **lists**.
Additionally, **data windows** can be specified on messages, allowing modules
to specify a narrower port than the message type. Streaming connections
between **ESI ports** are called **channels**. Modules with only Elastic Silicon
Interconnect (ESI) ports are called **ESI modules**. Modules with mixed
wire-level ports and ESI ports are allowed, but functionality which requires
ESI ports may not function on these modules. Modules can expose **ESI MMIO**
regions for memory-mapped I/O.

It is important to note that the message type does not specify anything
pertaining to the wire/cycle level. A large struct may be serialized across
several clock cycles and re-assembled in the receiver. Structs and other
atomic data types (everything except lists and data windows), however, are
guaranteed to be complete before they are handed to receivers.

## A note on syntax

This document contains pseudocode examples mostly written in [MLIR
intermediate representation textual
syntax](https://mlir.llvm.org/docs/LangRef/), as ESI is an MLIR dialect.
Different HDLs will have different syntaxes for ESI types, but the semantics
should be identical.

There are also some examples in C-style pseudocode. We do this to be
demonstrative of concepts where the MLIR syntax is too terse.

## Table of contents

1) [Base Type System](types.md)
2) [Streaming Channels](streaming.md)
3) [Memory-Mapped Interfaces](mmio.md)
4) [Software APIs](software_api.md)
5) [Services](services.md)
6) [Misc. Notes](notes.md)

## Language integrations

- [SystemVerilog](system_verilog.md)
