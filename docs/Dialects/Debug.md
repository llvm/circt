# Debug Dialect

This dialect provides operations and types to interleave debug information (DI)
with other parts of the IR.

[TOC]


## Rationale

The main goal of the debug dialect is to provide a mechanism to track the
correspondence between values, types, and hierarchy of a source language and the
IR being compiled and transformed. This allows simulators, synthesizers, and
other debugging tools to reconstruct a source language view into the processed
hardware and allow for easier debugging by humans.

Debug information in CIRCT follows these principles:

- **It is best effort:** DI is meant as a tool to aid humans in their debugging
  effort, not a contractual obligation to retain all source language semantics
  through the compilation pipeline. We preserve information as well as possible
  and reasonable, but accept the fact that certain optimizations may cause
  information to be discarded.

- **It affects the output:** Enabling the tracking of DI is expected to block
  certain optimizations. We undertake an effort to minimize the impact of DI on
  the output quality, size, simulation speed, or synthesis results, but accept
  the fact that preserving visibility and observability of source language
  constructs may prevent certain optimizations from running.


### Representations

There are two mechanisms in MLIR that lend themselves to conveying debug
information:

- **Attributes** attached to existing operations. This is similar to LLVM's
  approach of tracking DI in the operation's metadata. Translated to MLIR, an
  operation's location would be an obvious choice to do this tracking, since
  locations are well-preserved by passes and difficult to accidentally drop.
  MLIR currently does not support custom location attributes, which would
  require DI attributes to be attached to a `FusedLoc` as metadata.

- **Operations** interleaved with the rest of the IR. This makes DI a
  first-class citizen, but also causes debug information to potentially intefere
  with optimizations. For example, debug dialect ops introduce additional uses
  of values that might have otherwise been deleted by DCE. However, there may be
  alternative ways to dealing with such situations. For example, Verilog
  emission may simply ignore operations that are only used by debug ops,
  therefore achieving the same effect as DCE would have.

The debug dialect uses _operations_ to represent debug info. This decision was
based on discussions with various people in the LLVM and MLIR community, where
DI was commonly quoted as one of LLVM's weak points, with its living in metadata
space making it more of a second-class citizen rather than a first-class
concern. Since we want to represent source language types and constructs as
accurately as possible, and we want to track if values are type-lowered,
constant-folded, outlined, or adjusted in some other way, using operations seems
like a natural choice. MLIR ops already have all the machinery needed to refer
to values in the IR, and many passes will already do the right thing with them.


## Representing Source Language Constructs

The `dbg.variable` op is the key mechanism to establish a mapping between
high-level source language values and low-level values in the IR that are
transformed by the compiler. Consider the following source language pseudocode:

```plain
struct Req {
  data: i42,
  valid: i1,
  ready: &i1,
}
struct Resp {
  result: i42,
  done: i1,
}
module Foo {
  parameter Depth: uint;
  const Width: uint = 2**Depth;
  input req: Req;
  output resps: Resp[2];
  let x = req;
}
```

A frontend for this language could generate the following debug variables as
part of the body of module `Foo`, in order to track the structs, arrays,
parameters, constants, and local bindings present in the source language:

```mlir
hw.module @Foo_Width12(
  in %req_data: i42,
  in %req_valid: i1,
  out req_ready: i1,
  out resps0_result: i42,
  out resps0_done: i1,
  out resps1_result: i42,
  out resps1_done: i1
) {
  // %req_ready = ...
  // %resps0_result = ...
  // %resps0_done = ...
  // %resps1_result = ...
  // %resps1_done = ...

  // parameter Depth
  %c12_i32 = hw.constant 12 : i32
  dbg.variable "Depth", %c12_i32 : i32

  // const Width
  %c4096_i32 = hw.constant 4096 : i32
  dbg.variable "Width", %c4096_i32 : i32

  // input req: Req
  %0 = dbg.struct {"data": %req_data, "valid": %req_valid, "ready": %req_ready} : i42, i1, i1
  dbg.variable "req", %0 : !dbg.struct

  // output resps: Resp[2]
  %1 = dbg.struct {"result": %resps0_result, "done": %resps0_done} : i42, i1
  %2 = dbg.struct {"result": %resps1_result, "done": %resps1_done} : i42, i1
  %3 = dbg.array [%1, %2] : !dbg.struct, !dbg.struct
  dbg.variable "resps", %3 : !dbg.array

  // let x = req
  dbg.variable "x", %0 : !dbg.struct

  hw.output %req_ready, %resps0_result, %resps0_done, %resps1_result, %resps1_done : i1, i42, i1, i42, i1
}
```

Despite the fact that the `Req` and `Resp` structs, and `Resp[2]` array were
unrolled and lowered into separate scalar values in the IR, and the `ready: &i1`
input of `Req` having been turned into a `ready: i1` output, the `dbg.variable`
op accurately tracks how the original source language values can be
reconstructed. Note also how monomorphization has turned the `Depth` parameter
and `Width` into constants in the IR, but the corresponding `dbg.variable` ops
still expose the constant values under the name `Depth` and `Width` in the debug
info.


## Types


### Overview

The debug dialect does not precisely track the type of struct and array
aggregate values. Aggregates simply return the type `!dbg.struct` and
`!dbg.array`, respectively.

Extracting and emitting the debug information of a piece of IR involves looking
through debug ops to find actually emitted values that can be used to
reconstruct the source language values. Therefore the actual structure of the
debug ops is important, but their return type is not instrumental. The
distinction between struct and array types is an arbitrary choice that can be
changed easily, either by collapsing them into one aggregate type, or by more
precisely listing field/element types and array dimensions if the need arises.

[include "Dialects/DebugTypes.md"]


## Operations

[include "Dialects/DebugOps.md"]
