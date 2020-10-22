//===- ESIPorts.td - ESI port specifications ---------------*- tablegen -*-===//
//
// ESI ports are the primary mechanism on which the ESI dialect operates. These
// types wrap other types to specify the port as an ESI port. They represent
// ESI communication flows of various kinds. At first, ESI will only support
// 'channels' -- unidirectional, point-to-point, latency-insensitive streams.
//
//===----------------------------------------------------------------------===//

class ESI_Port<string name> : TypeDef<ESI_Dialect, name> {
  let cppClassName = name # "Port";
}

def Channel : ESI_Port<"Channel"> {
  let summary = "An ESI-compatible channel port";
  let description = [{
    An ESI port kind which models a latency-insensitive, unidirectional,
    point-to-point data stream. Channels are typed (like all of ESI). Said
    type can be any MLIR type, but must be lowered to something a backend
    knows how to output (i.e. something emitVerilog knows about).

    Example:

    ```mlir
    rtl.externmodule @Sender() -> ( !esi.channel<i1> { rtl.name = "x"} )
    rtl.module @Reciever(%a: !esi.channel<rtl.array<5xi16>>) { }
    ```
  }];

  let mnemonic = "channel";
  let parameters = (ins "Type":$inner);
}

def ChannelType :
  Type<CPred<"$_self.isa<::circt::esi::ChannelPort>()">, "">;
