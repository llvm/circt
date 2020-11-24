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


//=========
// Operations on ports.

def WrapValidReady : ESI_Op<"wrap.vr", [NoSideEffect]> {
  let summary = "Wrap a value into an ESI port";
  let description = [{
    Wrapping a value into an ESI port type allows modules to send values down
    an ESI port. Wrap data with valid bit, result is the ESI channel and the
    ready signal from the other end of the channel.
  }];

  let arguments = (ins AnyType:$data, I1:$valid);
  let results = (outs ChannelType:$output, I1:$ready);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def UnwrapValidReady : ESI_Op<"unwrap.vr", [NoSideEffect]> {
  let summary = "Unwrap a value from an ESI port";
  let description = [{
    Unwrapping a value allows operations on the contained value. Unwrap the
    channel along with a ready signal that you generate. Result is the data
    along with a valid signal.
  }];

  let arguments = (ins ChannelType:$input, I1:$ready);
  let results = (outs AnyType:$output, I1:$valid);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}
