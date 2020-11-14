//===- ESIOps.td - ESI dialect operation definitions --------- tablegen -*-===//
//
// This is where any operations for the ESI dialect live.
//
//===----------------------------------------------------------------------===//

/// Define a struct with all the possible ChannelBuffer options
def ChannelBufferOptions : StructAttr<"ChannelBufferOptions", ESI_Dialect, [
    // 'Stages' is used to specify a specific number of cycles (pipeline stages)
    // to use on this channel. Must be greater than 0.
    StructFieldAttr<"stages", OptionalAttr< Confined<AnyI64Attr, [IntMinValue<1>]> >>
  ]>;

def ChannelBuffer : ESI_Op<"buffer", [NoSideEffect, SameTypeOperands]> {
  let summary = "Control options for an ESI channel.";
  let description = [{
    A channel buffer (`buffer`) is essentially a set of options on a channel.
    It always adds at least one cycle of latency (pipeline stage) to the
    channel, but this is configurable.

    This operation is inserted on an ESI dataflow edge. It must exist
    previous to SystemVerilog emission but can be added in a lowering pass.

    Example:

    ```mlir
    %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    // Allow automatic selection of options.
    %bufferedChan = esi.buffer %esiChan { } : i1
    rtl.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // Alternatively, specify the number of stages.
    %fourStageBufferedChan = esi.buffer %esiChan { stages = 4 } : i1
    ```
  }];

  let arguments = (ins ChannelType:$input, ChannelBufferOptions:$options);
  let results = (outs ChannelType:$output);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}
