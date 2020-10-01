[Back to table of contents](index.md#Table-of-contents)

# ESI Memory Mapped (Read/Write) Interfaces

**This section is not fully thought out or written about.** And certainly not
implemented!

## MMIO Regions

The basic idea is to present ESI memory mapped **regions** exposed by
modules. There could be any number of these regions exposed and they
would work somewhat like input (request) / output (response) port pairs,
but with implicit request-response signaling and structure. The MMIO
space itself would be defined by a statically sized ESI struct (so lists
would be disallowed), with address offsets implicitly or explicitly
defined. The method for base address assignment is yet to be decided.
These regions can support atomic reads or writes of arbitrary size or
limited size.

## MMIO Requests (read/write)

Modules connected to the same MMIO bus could specify read/write
requests in several ways:

- As a data window with blinds (or similar construct) specifying the
  struct fields to read/write
- As a list of address offsets and sizes or data to read/write

Along with the request content, atomicity of the request would have to be
specified or, alternatively, atomicity could be declared statically by either
the requestor or responder.

The response type on a read would correspond to the read request -- either a
list of bytes read or a data window with the requested data filled in. For a
write, a simple acknowledgement or error would suffice for the response.

## Automatic self-description

An MMIO region's data type can be used to automatically generate a
self-describing software data type and an access API. Additionally, a per
MMIO bus table could be generated with the base addresses for each connected
region and a descriptor for each.

## Software access

The MMIO struct would become a software struct which software could mmap
to the base address as a typed pointer. This way, software access would
be simpler (just a normal struct pointer dereference -- e.g. p-\>field1)
and safer as it knows the correct address symbolically and the type of
that field. Additionally, if the MMIO region knows the processor's
endianness it could respond in the correct endianness. How to initiate
an atomic read/write of multiple fields from software is undecided yet,
though there may be some merit to exposing data windows to software.

## Additional Type: any

An MMIO space can be have type `any` to allow access to memory spaces which
are fundamentally untyped (e.g. a memory-mapped RAM). In this case, the
requestor can specify the type. For instance, to access 64-bit values the
requestor would send a request with a given address and specify a uint<64>
type response or write. Alignment declarations on `any` fields are TBD.

## Implementation

The MMIO system *could* be implemented on top of the streaming portion.
Parameterized message types for requests and responses could be defined.

## Examples

```c++
struct ConfigSpace {
    bool Enable;
    uint<16> PeriodInCycles;
}

module PulseOutput {
    responder MMIO<ConfigSpace> Config;
    output wire Pulse;
}
```

```c++
module HostProcessor {
    responder MMIO<any> Memory; // Untyped access to host memory
}
```
