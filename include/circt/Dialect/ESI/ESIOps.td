//===- ESIOps.td - ESI dialect operation definitions --------- tablegen -*-===//
//
// This is where any operations for the ESI dialect live.
//
//===----------------------------------------------------------------------===//


class ESI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ESI_Dialect, mnemonic, traits>;

/// Define a struct with all the possible ChannelBuffer options
def ChannelBufferOptions : StructAttr<"ChannelBufferOptions", ESI_Dialect,
  [StructFieldAttr<"stages", OptionalAttr<AnyI64Attr>>] >;

def ChannelBuffer : ESI_Op<"buffer", [NoSideEffect, SameTypeOperands]> {
  let summary = "Control options for an ESI channel.";
  let description = [{
    A channel buffer (buffer) is essentially a set of options on a channel.
    It always adds at least one cycle of latency (pipeline stage) to the
    channel, but this is configurable.

    This operation is inserted on an ESI dataflow edge. It must exist
    previous to SystemVerilog emission but can be added in a lowering pass.
  }];

  let arguments = (ins ChannelType:$input, ChannelBufferOptions:$options);
  let results = (outs ChannelType:$output);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}
