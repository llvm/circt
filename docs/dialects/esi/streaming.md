[Back to table of contents](index.md#Table-of-contents)

# ESI Streaming (Send/Receive) Interfaces

This section describes the use of the ESI type system to provide an API
abstraction for streaming hardware modules. We will describe memory mapped
interfaces later. We will use C-style pseudocode here for clarity. MLIR
ccheme (ops/blocks) is TBD.

## Type declarations

ESI types may be declared and named at the global and local scope.

## Ports

Verilog-style ports are extended to support an ESI data-type as the port
type. Port directions (`input`, `output`) are required and work as expected.
Any type, named or anonymous, may be used as a port declaration.

The default semantics are that the data-type in an interface is
presented atomically (single-cycle) to or from the module, except for
lists. Fine-grained control of presentation scheduling for large
data-types is described below in the [data windows](#data-windows)
section.

Parameterized types used in port declarations must have parameters fully
specified. Module parameters may be used to fully specify them. Multiplexing
a port (using it for multiple messages) can be achieved by using a
discriminated union as the port type.

### Port example

```c++
union RamRWReq<uint64 aWidth> {
    uint<aWidth> as ReadAddr;
    struct {
        uint<aWidth> addr;
        uint<64> data;
    } Write;
};

//Dual ported, 64-bit wide RAM with second port being read only
module RAM <uint64 depth> {
    LOCALPARAM aWidth = $clog2(depth);
    input RamRWReq<aWidth> portA;
    input uint<aWidth> portB; // Just accepts an address
    output uint64 dataOutA; // Output data if read
    output uint64 dataOutB; // Output data if read
}
```

(The above example uses parameterized types, which is not yet supported.)

## Data Windows

**Data windows are not supported yet** The features discussed here are subject
*to change.

By default, an entire message is guaranteed to be presented to the
receiving module in one clock cycle (excepting `lists`). For particularly
large messages, this is not ideal as it requires a data path equal to
the size of the message. Data windows specify which parts of a message a
module can accept on each clock cycle for immediate consumption. For
structs, they specify which members are accepted on which cycles. For
arrays and lists, they specify how many items (potentially fractional)
can be accepted each clock cycle.

It is important to note that data windows do not affect port
compatibility. In other words, a designer can connect ports using two
different windows into the same data type. This can be used to connect
different modules with different bandwidths. A data window merely
specifies the logical "gasket" used to connect two differently sized
ports.

For example, while the Compressor example below may appear to have a
bandwidth of 1 byte/cycle this need not be the case. That same message
data type can be used by compression blocks with any bandwidth. The
compiler just has to be smart enough to know that the module can accept
up to N items per cycle, then design a wire-level interface to suit. Data
windows are the way ESI specifies this. When components have different
bandwidths, the compiler can design the proper hardware to narrow or widen
the data path.

### Data Window Examples

```c++
typedef list uint8 DataList; // A variably-sized list of bytes
window(DataList) TenPerCycle { 10 } // 10 bytes / cycle (80-bit wide channel)

// A module that reads up to 10 bytes / cycle and writes up to
// 7 bytes / cycle
module Compressor {
    input TenPerCycle dataIn;
    output window(DataList) { 7 } dataOut;
}
```

```c++
// An example of breaking up a large struct with data windows

// A large struct
struct DataRecord {
    uint64 ID;
    uint512 CryptoHash;
    uint8[16] DisplayName;
    uint16 a, b, c, d, e;
    list uint8 Blob;
}

// Windows break up structs as well
window (DataRecord) DataRecordParts {
    Part1 {
        ID, CryptoHash  // Accept ID and CryptoHash
                        // on the first cycle
    }
    Part2 { DisplayName } // Accept the CryptoHash on the second
    Part3 {a .. e} // Accept a, b, c, d, and e
    PartRemainder {
        Blob { 64 } // Accept 64 bytes / cycle
    }
}
// The total width of this window is the width of the largest part. In this
// case, Part1 is the largest at 576 bits.

// Use the window as the data type on the port
module RecordProcessor {
    input DataRecordParts in;
}

// This module can be connected to 'RecordProcessor'. This module outputs the
// statically sized part of the struct (everything up until 'Blob') in one cycle
// and the list ('Blob') one uint8 (byte) at a time
module RecordCreator {
    output DataRecord out;
}
```

## Data Blinds

**Data blinds are not currently supported.** We have some questions about
their usefulness. A more generic data type converter could be designed as a
replacement. Future work.

By default, all the data in a message must be sent/received. Data blinds
are a feature of data windows which allows modules to specify that they
either don't want to receive or do not send a particular field. If a
data blind is specified on an output, there are two options depending on
the level of safety required: (1) fill in that field with the default
value or (2) disallow any input interface which does not have a data
window also includes that blind.

### Data Blind Example

```c++
// Builds on previous example
window (DataRecord) DataRecordBlinds {
    -CryptoHash, // Don't send/recv CrypoHash
    -a..c // or a, b, or c
}
```
