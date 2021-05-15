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

/// Define a struct with all the possible ChannelBuffer options
def ChannelBufferOptions : StructAttr<"ChannelBufferOptions", ESI_Dialect, [
    // 'Stages' is used to specify a specific number of cycles (pipeline stages)
    // to use on this channel. Must be greater than 0.
    StructFieldAttr<"stages", OptionalAttr< Confined<I64Attr, [IntMinValue<1>]> >>,

    // Name assigned to a buffered connection.
    StructFieldAttr<"name", OptionalAttr< StrAttr >>
  ]>;

def ChannelBuffer : ESI_Abstract_Op<"buffer", [NoSideEffect]> {
  let summary = "Control options for an ESI channel.";
  let description = [{
    A channel buffer (`buffer`) is essentially a set of options on a channel.
    It always adds at least one cycle of latency (pipeline stage) to the
    channel, but this is configurable.

    This operation is inserted on an ESI dataflow edge. It must exist
    previous to SystemVerilog emission but can be added in a lowering pass.

    Example:

    ```mlir
    %esiChan = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    // Allow automatic selection of options.
    %bufferedChan = esi.buffer %esiChan { } : i1
    hw.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // Alternatively, specify the number of stages.
    %fourStageBufferedChan = esi.buffer %esiChan { stages = 4 } : i1
    ```
  }];

  let arguments = (ins I1:$clk, I1:$rstn, ChannelType:$input,
    ChannelBufferOptions:$options);
  let results = (outs ChannelType:$output);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def PipelineStage : ESI_Physical_Op<"stage", [NoSideEffect]> {
  let summary = "An elastic buffer stage.";
  let description = [{
    An individual elastic pipeline register. Generally lowered to from a
    ChannelBuffer ('buffer'), though can be inserted anywhere to add an
    additional pipeline stage. Adding individually could be useful for
    late-pass latency balancing.
  }];

  let arguments = (ins I1:$clk, I1:$rstn, ChannelType:$input);
  let results = (outs ChannelType:$output);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def CosimEndpoint : ESI_Physical_Op<"cosim", []> {
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

  let arguments = (ins I1:$clk, I1:$rstn, ChannelType:$send,
    I64Attr:$endpointID);
  let results = (outs ChannelType:$recv);

  let assemblyFormat = [{
    $clk `,` $rstn `,` $send `,` $endpointID attr-dict
      `:` type($send) `->` type($recv)
  }];
}

def RtlBitArray : Type<CPred<"$_self.isa<::circt::rtl::ArrayType>()"
 " && $_self.cast<::circt::rtl::ArrayType>().getElementType() =="
 "   ::mlir::IntegerType::get($_self.getContext(), 1)">, "an HW bit array">;

def CapnpDecode : ESI_Physical_Op<"decode.capnp", [NoSideEffect]> {
  let summary = "Translate bits in Cap'nProto messages to HW typed data";

  let arguments = (ins I1:$clk, I1:$valid, RtlBitArray:$capnpBits);
  let results = (outs AnyType:$decodedData);

  let assemblyFormat = [{
    $clk $valid $capnpBits attr-dict `:` type($capnpBits) `->` type($decodedData)
  }];
}

def CapnpEncode : ESI_Physical_Op<"encode.capnp", [NoSideEffect]> {
  let summary = "Translate HW typed data to Cap'nProto";

  let arguments = (ins I1:$clk, I1:$valid, AnyType:$dataToEncode);
  let results = (outs RtlBitArray:$capnpBits);

  let assemblyFormat = [{
    $clk $valid $dataToEncode attr-dict `:` type($dataToEncode) `->` type($capnpBits)
  }];
}

def NullSourceOp : ESI_Physical_Op<"null", [NoSideEffect]> {
  let summary = "An op which never produces messages.";

  let arguments = (ins);
  let results = (outs ChannelType:$out);

  let assemblyFormat = [{ attr-dict `:` type($out) }];
}
