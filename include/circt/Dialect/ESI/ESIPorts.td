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

#ifndef CIRCT_DIALECT_ESI_PORTS_TD
#define CIRCT_DIALECT_ESI_PORTS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/ESI/ESIDialect.td"
include "circt/Dialect/ESI/ESIInterfaces.td"
include "circt/Dialect/ESI/ESITypes.td"

//=========
// Operations on ports.

def WrapValidReadyOp : ESI_Op<"wrap.vr", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Wrap a value into an ESI port";
  let description = [{
    Wrapping a value into an ESI port type allows modules to send values down
    an ESI port. Wrap data with valid bit, result is the ESI channel and the
    ready signal from the other end of the channel.
  }];

  let arguments = (ins AnyType:$rawInput, I1:$valid);
  let results = (outs ChannelType:$chanOutput, I1:$ready);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$data, "mlir::Value":$valid)>
  ];
}

def UnwrapValidReadyOp : ESI_Op<"unwrap.vr", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Unwrap a value from an ESI port";
  let description = [{
    Unwrapping a value allows operations on the contained value. Unwrap the
    channel along with a ready signal that you generate. Result is the data
    along with a valid signal.
  }];

  let arguments = (ins ChannelType:$chanInput, I1:$ready);
  let results = (outs AnyType:$rawOutput, I1:$valid);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$inChan, "mlir::Value":$ready)>
  ];
}

def WrapFIFOOp : ESI_Op<"wrap.fifo", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Wrap a value into an ESI port with FIFO signaling";

  let arguments = (ins AnyType:$data, I1:$empty);
  let results = (outs ChannelType:$chanOutput, I1:$rden);
  let hasCanonicalizeMethod = true;
  let hasFolder = true;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $data `,` $empty attr-dict `:`
      custom<WrapFIFOType>(type($data), type($chanOutput))
  }];
}

def UnwrapFIFOOp : ESI_Op<"unwrap.fifo", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Unwrap a value from an ESI port into a FIFO interface";

  let arguments = (ins ChannelType:$chanInput, I1:$rden);
  let results = (outs AnyType:$data, I1:$empty);
  let hasCanonicalizeMethod = true;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $chanInput `,` $rden attr-dict `:` qualified(type($chanInput))
  }];

  let extraClassDeclaration = [{
    static LogicalResult mergeAndErase(UnwrapFIFOOp, WrapFIFOOp, PatternRewriter&);
  }];
}

def ModportType:
  Type<CPred<"$_self.isa<::circt::sv::ModportType>()">, "sv.interface">;

def WrapSVInterfaceOp: ESI_Op<"wrap.iface", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Wrap an SV interface into an ESI port";
  let description = [{
    Wrap a SystemVerilog interface into an ESI channel. Interface MUST look
    like an interface produced by ESI meaning it MUST contain valid, ready,
    and data signals. Any other signals will be discarded.
  }];

  let arguments = (ins ModportType:$interfaceSink);
  let results = (outs ChannelType:$output);

  let assemblyFormat = [{
    $interfaceSink attr-dict `:` qualified(type($interfaceSink)) `->` qualified(type($output))
  }];

  let hasVerifier = 1;
}

def UnwrapSVInterfaceOp : ESI_Op<"unwrap.iface", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Unwrap an SV interface from an ESI port";
  let description = [{
    Unwrap an ESI channel into a SystemVerilog interface containing valid,
    ready, and data signals.
  }];

  let arguments = (ins ChannelType:$chanInput, ModportType:$interfaceSource);
  let results = (outs);

  let assemblyFormat = [{
    $chanInput `into` $interfaceSource attr-dict `:` `(` qualified(type($chanInput)) `,` qualified(type($interfaceSource)) `)`
  }];

  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_ESI_PORTS_TD
