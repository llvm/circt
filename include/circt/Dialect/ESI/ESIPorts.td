//===- ESIPorts.td - ESI port specifications ---------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
    hw.module.extern @Sender() -> (%x: !esi.channel<i1>)
    hw.module @Reciever(%a: !esi.channel<hw.array<5xi16>>) { }
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

  let arguments = (ins AnyType:$rawInput, I1:$valid);
  let results = (outs ChannelType:$chanOutput, I1:$ready);

  let printer = [{ return ::print$cppClass(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$data, "mlir::Value":$valid)>
  ];
}

def UnwrapValidReady : ESI_Op<"unwrap.vr", [NoSideEffect]> {
  let summary = "Unwrap a value from an ESI port";
  let description = [{
    Unwrapping a value allows operations on the contained value. Unwrap the
    channel along with a ready signal that you generate. Result is the data
    along with a valid signal.
  }];

  let arguments = (ins ChannelType:$chanInput, I1:$ready);
  let results = (outs AnyType:$rawOutput, I1:$valid);

  let printer = [{ return ::print$cppClass(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$inChan, "mlir::Value":$ready)>
  ];
}

def ModportType:
  Type<CPred<"$_self.isa<::circt::sv::ModportType>()">, "sv.interface">;

def WrapSVInterface: ESI_Op<"wrap.iface", [NoSideEffect]> {
  let summary = "Wrap an SV interface into an ESI port";
  let description = [{
    Wrap a SystemVerilog interface into an ESI channel. Interface MUST look
    like an interface produced by ESI meaning it MUST contain valid, ready,
    and data signals. Any other signals will be discarded.
  }];

  let arguments = (ins ModportType:$interfaceSink);
  let results = (outs ChannelType:$output);

  let assemblyFormat = [{
    $interfaceSink attr-dict `:` type($interfaceSink) `->` type($output)
  }];

  let verifier = [{
    return ::verify$cppClass(*this);
  }];
}

def UnwrapSVInterface : ESI_Op<"unwrap.iface", []> {
  let summary = "Unwrap an SV interface from an ESI port";
  let description = [{
    Unwrap an ESI channel into a SystemVerilog interface containing valid,
    ready, and data signals.
  }];

  let arguments = (ins ChannelType:$chanInput, ModportType:$interfaceSource);
  let results = (outs);

  let assemblyFormat = [{
    $chanInput `into` $interfaceSource attr-dict `:` `(` type($chanInput) `,` type($interfaceSource) `)`
  }];

  let verifier = [{
    return ::verify$cppClass(*this);
  }];
}
