# Intrinsics

Intrinsics provide an implementation-specific way to extend the FIRRTL language
with new operations.

Intrinsics are currently implemented as annotated external modules.  We expect
that native FIRRTL support for intrinsics will be added to the language.

## Motivation

Intrinsics provide a way to add functionality to FIRRTL without having to extend
the FIRRTL language. This allows a fast path for prototyping new operations to 
rapidly respond to output requirements.  Intrinsics maintain strict definitions
and type checking.

## Supported Intrinsics

Annotations here are written in their JSON format. A "reference target"
indicates that the annotation could target any object in the hierarchy,
although there may be further restrictions in the annotation.

### circt.sizeof

Returns the size of a type.  The input port is not read from and may be any 
type, including uninferred types.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| i          | input     | Any      | value whose type is to be returned  |
| size       | output    | UInt<32> | Size of type of i                   |

### circt.isX

Tests if the value is a literal `x`.  FIRRTL doesn't have a notion of 'x per-se, 
but x can come in to the system from external modules and from SV constructs.  
Verification constructs need to explicitly test for 'x.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| i          | input     | Any      | value test                          |
| found      | output    | UInt<1>  | i is `x`                            |

### circt.plusargs.value

Tests and extracts a value from simulator command line options with SystemVerilog
`$value$plusargs`.  This is described in SystemVerilog 2012 section 21.6.

We do not currently check that the format string substitution flag matches the
type of the result.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |
| FORMAT     | string | Format string per SV 21.6                         |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| found      | output    | UInt<1>  | found in args                       |
| result     | output    | AnyType  | found in args                       |


### circt.plusargs.test

Tests simulator command line options with SystemVerilog `$test$plusargs`.  This
is described in SystemVerilog 2012 section 21.6.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |
| FORMAT     | string | Format string per SV 21.6                         |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| found      | output    | UInt<1>  | found in args                       |
