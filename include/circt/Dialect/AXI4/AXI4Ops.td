//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4OPS_TD
#define CIRCT_DIALECT_AXI4_AXI4OPS_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "circt/Dialect/AXI4/AXI4Types.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class AXI4Op<string mnemonic, list<Trait> traits = []> :
    Op<AXI4Dialect, mnemonic, traits>;

// Constrains each `!axi4.port` result to at most one use
def PortResultsAtMostOneUse : NativeOpTrait<"PortResultsAtMostOneUse"> {
  let cppNamespace = "::circt::axi4::OpTrait";
}

// Constrain an argument to the hw representation of a given AXI channel
class PayloadMatches<string channel, string arg> : TypesMatchWith<
    channel # " payload matches the port type", "port", arg,
    "::circt::axi4::getChannelPayloadType("
      "::llvm::cast<::circt::axi4::PortType>($_self), "
      "::circt::axi4::AXI4Channel::" # channel # ")">;

// Ordered list of AXI4 channels
defvar axi4Channels = ["AW", "W", "B", "AR", "R"];

// Base class for the ops bridging an !axi4.port and its hw channel structs,
// with automatic struct type enforcement
class AXI4ChannelStructOp<string mnemonic, list<Trait> traits = []> :
    AXI4Op<mnemonic, !listconcat(traits,
        !foreach(c, axi4Channels, PayloadMatches<c, !tolower(c)>))>;

def AbstractManagerOp : AXI4Op<"abstract_manager", [PortResultsAtMostOneUse]> {
  let summary = "A stand-alone AXI4 manager endpoint";
  let description = [{
    Models a manager endpoint with no concrete implementation, for abstract
    network modelling. Returns the `!axi4.port` it drives.

    Example:
    ```mlir
    %mgr = axi4.abstract_manager %clk, %rst_ni : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset);
  let results = (outs PortType:$port);

  let assemblyFormat =
      "$clock `,` $reset attr-dict `:` qualified(type($port))";
}

def AbstractSubordinateOp : AXI4Op<"abstract_subordinate"> {
  let summary = "A stand-alone AXI4 subordinate endpoint";
  let description = [{
    Models a subordinate endpoint with no concrete implementation, for abstract
    network modelling. Takes the upstream `!axi4.port` it responds to.

    Example:
    ```mlir
    axi4.abstract_subordinate %clk, %rst_ni, %mgr : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);

  let assemblyFormat =
      "$clock `,` $reset `,` $upstream attr-dict `:` qualified(type($upstream))";
}

def ChannelStructsToPortOp : AXI4ChannelStructOp<"channel_structs_to_port",
    [PortResultsAtMostOneUse]> {
  let summary = "Converts a HW expression of an AXI4 interface to a port";
  let description = [{
    Bridges from a HW dialect expression of an AXI4 interface to an
    `!axi4.port`. Takes the AW, W and AR channel payloads and valids, and the B
    and R readys; returns the port along with the AW, W and AR readys and the B
    and R payloads and valids.

    Example:
    ```mlir
    %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid =
      axi4.channel_structs_to_port %clk, %rst_ni
        aw %aw, %aw_valid w %w, %w_valid b %b_ready
        ar %ar, %ar_valid r %r_ready
      : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset,
                       StructType:$aw, I1:$aw_valid,
                       StructType:$w, I1:$w_valid,
                       I1:$b_ready,
                       StructType:$ar, I1:$ar_valid,
                       I1:$r_ready);
  let results = (outs PortType:$port,
                      I1:$aw_ready, I1:$w_ready,
                      StructType:$b, I1:$b_valid,
                      I1:$ar_ready,
                      StructType:$r, I1:$r_valid);

  let assemblyFormat = [{
    $clock `,` $reset
      `aw` $aw `,` $aw_valid `w` $w `,` $w_valid `b` $b_ready
      `ar` $ar `,` $ar_valid `r` $r_ready
      attr-dict `:` qualified(type($port))
  }];
}

def PortToChannelStructsOp : AXI4ChannelStructOp<"port_to_channel_structs"> {
  let summary = "Converts an AXI4 port to a HW expression of an AXI4 interface";
  let description = [{
    Bridges from an `!axi4.port` to a HW dialect expression of an AXI4
    interface. Takes the port, the AW, W and AR channel readys, and the B and R
    payloads and valids; returns the AW, W and AR payloads and valids, and the
    B and R readys.

    Example:
    ```mlir
    %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready =
      axi4.port_to_channel_structs %clk, %rst_ni, %port
        aw %aw_ready w %w_ready b %b, %b_valid
        ar %ar_ready r %r, %r_valid
      : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$port,
                       I1:$aw_ready, I1:$w_ready,
                       StructType:$b, I1:$b_valid,
                       I1:$ar_ready,
                       StructType:$r, I1:$r_valid);
  let results = (outs StructType:$aw, I1:$aw_valid,
                      StructType:$w, I1:$w_valid,
                      I1:$b_ready,
                      StructType:$ar, I1:$ar_valid,
                      I1:$r_ready);

  let assemblyFormat = [{
    $clock `,` $reset `,` $port
      `aw` $aw_ready `w` $w_ready `b` $b `,` $b_valid
      `ar` $ar_ready `r` $r `,` $r_valid
      attr-dict `:` qualified(type($port))
  }];
}

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_TD
