# Intrinsics

Intrinsics provide an implementation-specific way to extend the firrtl language
with new operations.

Intrinsics are currently implemented as annotated external modules.  We expect
that native firrtl support for intrinsics will be added to the language.

## Motivation

Intrinsics provide a way to add functionality to firrtl without having to extend
the firrtl language. This allows a fast path for prototyping new operations to 
rapidly repsond to output requirements.  Intrinsics maintain strict definitions
and type checking.

## Supported Intrinsics

Annotations here are written in their JSON format. A "reference target"
indicates that the annotation could target any object in the hierarchy,
although there may be further restrictions in the annotation.

### circt.sizeof

Returns the size of a type.  The input port is not read from and may be any 
type, including uninfered types.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| i          | input     | Any      | value whose type is to be returned  |
| size       | output    | UInt<32> | Size of type of i                   |

