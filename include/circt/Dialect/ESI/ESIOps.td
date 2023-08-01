//===- ESIOps.td - ESI dialect operation definitions --------- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where any operations for the ESI dialect live.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_OPS_TD
#define CIRCT_DIALECT_ESI_OPS_TD

def ChannelBufferOp : ESI_Abstract_Op<"buffer", [
    Pure,
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Control options for an ESI channel.";
  let description = [{
    A channel buffer (`buffer`) is essentially a set of options on a channel.
    It always adds at least one cycle of latency (pipeline stage) to the
    channel, but this is configurable.

    This operation is inserted on an ESI dataflow edge. It must exist
    previous to SystemVerilog emission but can be added in a lowering pass.

    A `stages` attribute may be provided to specify a specific number of cycles
    (pipeline stages) to use on this channel. Must be greater than 0.

    A `name` attribute may be provided to assigned a name to a buffered
    connection.

    Example:

    ```mlir
    %esiChan = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    // Allow automatic selection of options.
    %bufferedChan = esi.buffer %esiChan : i1
    hw.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // Alternatively, specify the number of stages.
    %fourStageBufferedChan = esi.buffer %esiChan { stages = 4 } : i1
    ```
  }];

  let arguments = (ins I1:$clk, I1:$rst, ChannelType:$input,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<1>]>>:$stages,
    OptionalAttr<StrAttr>:$name);
  let results = (outs ChannelType:$output);
  let hasCustomAssemblyFormat = 1;
}

def PipelineStageOp : ESI_Physical_Op<"stage", [
    Pure,
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "An elastic buffer stage.";
  let description = [{
    An individual elastic pipeline register. Generally lowered to from a
    ChannelBuffer ('buffer'), though can be inserted anywhere to add an
    additional pipeline stage. Adding individually could be useful for
    late-pass latency balancing.
  }];

  let arguments = (ins I1:$clk, I1:$rst, ChannelType:$input);
  let results = (outs ChannelType:$output);
  let hasCustomAssemblyFormat = 1;
}

def CosimEndpointOp : ESI_Physical_Op<"cosim", []> {
  let summary = "Co-simulation endpoint";
  let description = [{
    A co-simulation endpoint is a connection from the simulation to some
    outside process, usually a software application responsible for driving
    the simulation (driver).

    ESI uses a serialization protocol called Cap'n Proto (capnp for short).
    The advantage of capnp is the decoding overhead: for value types (ints,
    structs, etc.) there is none! This stands in contrast to Protocol Buffers
    and Bond as their messages contain metadata for each field which must be
    interpreted.

    The advantage of using a well-supported serialization protocol is
    language support -- driver applications can be written in any language
    supported by the specific protocol.
  }];

  let arguments = (ins I1:$clk, I1:$rst, ChannelType:$send, StrAttr:$name);
  let results = (outs ChannelType:$recv);

  let assemblyFormat = [{
    $clk `,` $rst `,` $send `,` $name attr-dict
      `:` qualified(type($send)) `->` qualified(type($recv))
  }];
}

def RtlBitArrayType : Type<CPred<"$_self.isa<::circt::hw::ArrayType>()"
 " && $_self.cast<::circt::hw::ArrayType>().getElementType() =="
 "   ::mlir::IntegerType::get($_self.getContext(), 1)">, "an HW bit array">;

def CapnpDecodeOp : ESI_Physical_Op<"decode.capnp", [Pure]> {
  let summary = "Translate bits in Cap'nProto messages to HW typed data";

  let arguments = (ins I1:$clk, I1:$valid, RtlBitArrayType:$capnpBits);
  let results = (outs AnyType:$decodedData);

  let assemblyFormat = [{
    $clk $valid $capnpBits attr-dict `:` qualified(type($capnpBits)) `->`
                                         qualified(type($decodedData))
  }];
}

def CapnpEncodeOp : ESI_Physical_Op<"encode.capnp", [Pure]> {
  let summary = "Translate HW typed data to Cap'nProto";

  let arguments = (ins I1:$clk, I1:$valid, AnyType:$dataToEncode);
  let results = (outs RtlBitArrayType:$capnpBits);

  let assemblyFormat = [{
    $clk $valid $dataToEncode attr-dict `:` qualified(type($dataToEncode))
                                          `->` qualified(type($capnpBits))
  }];
}

def NullSourceOp : ESI_Physical_Op<"null", [Pure]> {
  let summary = "An op which never produces messages.";

  let arguments = (ins);
  let results = (outs ChannelType:$out);

  let assemblyFormat = [{ attr-dict `:` qualified(type($out)) }];
}

def WrapWindow : ESI_Physical_Op<"window.wrap", [Pure]> {
  let summary = "wrap a union into a data window";

  let arguments = (ins UnionType:$frame);
  let results = (outs ESIWindowType:$window);
  let hasVerifier = 1;
  let hasFolder = 1;

  let assemblyFormat = [{
     $frame attr-dict `:` custom<InferWindowRet>(type($frame), type($window))
  }];
}

def UnwrapWindow : ESI_Physical_Op<"window.unwrap", [
      Pure,
      DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "unwrap a data window into a union";

  let arguments = (ins ESIWindowType:$window);
  let results = (outs UnionType:$frame);
  let hasFolder = 1;

  let assemblyFormat = [{
     $window attr-dict `:` qualified(type($window))
  }];
}

#endif // CIRCT_DIALECT_ESI_OPS_TD
