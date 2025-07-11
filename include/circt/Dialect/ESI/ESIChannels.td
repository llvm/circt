//===- ESIChannels.td - All ESI ops related to app channels -- tablegen -*-===//
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

#ifndef CIRCT_DIALECT_ESI_CHANNELS_TD
#define CIRCT_DIALECT_ESI_CHANNELS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"

include "circt/Dialect/ESI/ESIDialect.td"
include "circt/Dialect/ESI/ESIInterfaces.td"
include "circt/Dialect/ESI/ESITypes.td"

//===----------------------------------------------------------------------===//
// ESI channel type
//===----------------------------------------------------------------------===//

// Standard valid-ready signaling.
def ChannelSignalingValidReady : I32EnumAttrCase<"ValidReady", 0>;
// FIFO interface with a read latency of zero. Very similiar to valid-ready, but
// cannot assert ready if data is not available.
def ChannelSignalingFIFO      : I32EnumAttrCase<"FIFO",      1>;

def ChannelSignaling : I32EnumAttr<
    "ChannelSignaling",
    "ESI channel wire signaling standard",
    [ChannelSignalingValidReady, ChannelSignalingFIFO]>{
  let cppNamespace = "::circt::esi";
}

def ChannelTypeImpl : ESI_Type<"Channel"> {
  let summary = "An ESI-compatible channel port";
  let description = [{
    An ESI port kind which models a latency-insensitive, unidirectional,
    point-to-point data stream. Channels are typed (like all of ESI). Said
    type can be any MLIR type, but must be lowered to something a backend
    knows how to output (i.e. something emitVerilog knows about).

    Parameters:
      signaling: the style of the control signals (valid/ready vs FIFO).
      dataDelay: the number of cycles data takes to arrive after the control
      indicates a transaction has occured. For instance, on a FIFO without read
      ahead, this would be 1. Defaults to 0.

    Example:

    ```mlir
    hw.module.extern @Sender() -> (%x: !esi.channel<i1>)
    hw.module @Reciever(%a: !esi.channel<hw.array<5xi16>>) { }
    ```
  }];

  let mnemonic = "channel";
  let parameters = (ins
      "Type":$inner,
      DefaultValuedParameter<
        "::circt::esi::ChannelSignaling",
        "::circt::esi::ChannelSignaling::ValidReady">:$signaling,
      DefaultValuedParameter<"uint64_t", "0">:$dataDelay);

  let assemblyFormat = [{
    `<` $inner (`,` $signaling^ ( `(` $dataDelay^ `)` )? )? `>`
  }];

  let builders = [
    TypeBuilder<(ins "Type":$type), [{
        return Base::get(type.getContext(), type,
                         ::circt::esi::ChannelSignaling::ValidReady, 0);
      }]>,
  ];

  let extraClassDeclaration = [{
    /// Consumers are ones which actually absorb tokens. Non-consumer ops
    /// include any snooping operations.
    static SmallVector<std::reference_wrapper<OpOperand>, 4> getConsumers(
      mlir::TypedValue<ChannelType>);
    static bool hasOneConsumer(mlir::TypedValue<ChannelType>);
    static bool hasNoConsumers(mlir::TypedValue<ChannelType>);
    static LogicalResult verifyChannel(mlir::TypedValue<ChannelType>);

    /// Get the single consumer of a channel. Returns nullptr if there are zero
    /// or more than one.
    static OpOperand* getSingleConsumer(mlir::TypedValue<ChannelType>);
  }];
}

//===----------------------------------------------------------------------===//
// Operations which reveal the internal signals of a channel. They hang off
// channels but don't count towards its user count.
//===----------------------------------------------------------------------===//

def SnoopValidReadyOp : ESI_Physical_Op<"snoop.vr", [InferTypeOpInterface]> {
  let summary = "Get the valid, ready, and data signals from a channel";
  let description = [{
    A snoop allows one to combinationally observe a channel's internal signals.
    It does not count as another user of the channel. Useful for constructing
    control logic which can be combinationally driven. Also potentially useful
    for debugging.
  }];

  let arguments = (ins ChannelType:$input);
  let results = (outs I1:$valid, I1:$ready, AnyType:$data);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input))
  }];

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

def SnoopTransactionOp : ESI_Physical_Op<"snoop.xact", [InferTypeOpInterface]> {
  let summary = "Get the data and transaction signal from a channel";
  let description = [{
    A snoop that observes when a transaction occurs on a channel and provides
    the data being transmitted. The transaction signal indicates when data is
    actually being transferred on the channel, regardless of the underlying
    signaling protocol (ValidReady or FIFO). Like other snoop operations, this
    does not count as another user of the channel. Useful for monitoring data
    flow and debugging.
  }];

  let arguments = (ins ChannelType:$input);
  let results = (outs I1:$transaction, AnyType:$data);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input))
  }];

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

//===----------------------------------------------------------------------===//
// Wrap / unwrap channels to their signaling protocols.
//===----------------------------------------------------------------------===//

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
  Type<CPred<"llvm::isa<::circt::sv::ModportType>($_self)">, "sv.interface">;

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

//===----------------------------------------------------------------------===//
// Channel bundles
//===----------------------------------------------------------------------===//

// Used to indicate the presence of a keyword in asm. Should be used within
// optional parens in the declarative assembly format.
// Upstreaming: https://github.com/llvm/llvm-project/pull/65438
def UnitParameter : AttrOrTypeParameter<"::mlir::UnitAttr", "boolean flag"> {
  let printer = "";
  let parser = "::mlir::UnitAttr::get($_parser.getContext())";
  let defaultValue = "::mlir::UnitAttr()";
}

def ChannelDirection : I32EnumAttr<"ChannelDirection",
  "Direction of channel (see ChannelBundleImpl for details)", [
    I32EnumAttrCase<"to", 1>,
    I32EnumAttrCase<"from", 2>,
  ]> {
  let cppNamespace = "::circt::esi";
}


def ChannelBundleType : ESI_Type<"ChannelBundle"> {
  let summary = "a bundle of channels";

  let description = [{
    A channel bundle (sometimes referred to as just "bundle") is a set of
    channels of associated signals, along with per-channel names and directions.
    The prototypical example for a bundle is a request-response channel pair.

    The direction terminology is a bit confusing. Let us designate the module
    which is outputting the bundle as the "sender" module and a module which has
    a bundle as an input as the "receiver". The directions "from" and "to" are
    from the senders perspective. So, the "to" direction means that channel is
    transmitting messages from the sender to the receiver. Then, "from" means
    that the sender is getting messages from the receiver (typically responses).

    When requesting a bundle from a service, the service is always considered
    the sender; so, "to" means the service is sending messages to the client and
    "from" means the service is receiving messages from the client.
  }];

  let mnemonic = "bundle";
  let parameters = (ins ArrayRefParameter<"BundledChannel">:$channels,
                        UnitParameter:$resettable);
  let assemblyFormat = [{
    `<` `[` $channels `]` (`reset` $resettable^)? `>`
  }];

  let extraClassDeclaration = [{
    // Get a channel bundle with the same types and names but the directions
    // reversed.
    ChannelBundleType getReversed() const;

    // Return the opposite direction.
    static ChannelDirection flip(ChannelDirection dir) {
      return dir == ChannelDirection::from
                               ? ChannelDirection::to
                               : ChannelDirection::from;
    }
  }];
}

def PackBundleOp : ESI_Op<"bundle.pack", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = "pack channels into a bundle";

  let arguments = (ins Variadic<ChannelType>:$toChannels);
  let results =  (outs ChannelBundleType:$bundle,
                       Variadic<ChannelType>:$fromChannels);
  let assemblyFormat = [{
    $toChannels attr-dict `:` custom<UnPackBundleType>(
      type($toChannels), type($fromChannels), type($bundle))
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ChannelBundleType":$bundleType,
                   "mlir::ValueRange":$toChannels)>
  ];
}

def UnpackBundleOp : ESI_Op<"bundle.unpack", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = "unpack channels from a bundle";

  let arguments = (ins ChannelBundleType:$bundle,
                       Variadic<ChannelType>:$fromChannels);
  let results =  (outs Variadic<ChannelType>:$toChannels);
  let assemblyFormat = [{
    $fromChannels `from` $bundle attr-dict `:` custom<UnPackBundleType>(
      type($toChannels), type($fromChannels), type($bundle))
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$bundle, "mlir::ValueRange":$fromChannels)>
  ];
}

//===----------------------------------------------------------------------===//
// Data windows
//===----------------------------------------------------------------------===//

def ESIWindowType : ESI_Type<"Window"> {
  let summary = "a data window";
  let description = [{
    A 'data window' allows designers to break up large messages into multiple
    frames (aka phits) spread across multiple cycles. Windows are specified in
    terms of a mapping of struct fields to frames. The width of a window is the
    maximum frame size + the union tag (log2(#frames)).

    A data window does NOT imply an ESI channel.

    Current restrictions:
    - A field may only appear once.
    - Fields may only be re-ordered (wrt the original message) within a frame.
    - Array fields whose array length is not evenly divisible by 'numItems' will
    have an implicit frame inserted directly after containing the leftover array
    items.
    - Array fields with an array length MUST be in their own frame.
  }];

  let mnemonic = "window";
  let genVerifyDecl = 1;

  let parameters = (ins "StringAttr":$name, "mlir::Type":$into,
                        ArrayRefParameter<"WindowFrameType">:$frames);
  let assemblyFormat = [{ `<` $name `,` $into `,` `[` $frames `]` `>` }];

  let extraClassDeclaration = [{
    /// Get the union which realizes this window during lowering.
    circt::hw::UnionType getLoweredType() const;
  }];
}

def WindowFrameType : ESI_Type<"WindowFrame"> {
  let summary = "Declare a data window frame";
  let description = [{
    A named list of fields which should appear in a given frame.
  }];
  let mnemonic = "window.frame";

  let parameters = (ins "StringAttr":$name,
                        ArrayRefParameter<"WindowFieldType">:$members);

  let assemblyFormat = [{
    `<` $name `,` `[` $members `]` `>`
  }];
}

def WindowFieldType : ESI_Type<"WindowField"> {
  let summary = "a field-in-frame specifier";
  let description = [{
    Specify that a field should appear within the enclosing frame.
  }];
  let mnemonic = "window.field";

  let parameters = (ins
      "StringAttr":$fieldName,
      OptionalParameter<"uint64_t", "# of items in arrays or lists">:$numItems);

  let assemblyFormat = [{
    `<` $fieldName (`,` $numItems^)? `>`
  }];
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

//===----------------------------------------------------------------------===//
// Channel buffering and pipelining.
//===----------------------------------------------------------------------===//

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

  let arguments = (ins ClockType:$clk, I1:$rst, ChannelType:$input,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<1>]>>:$stages,
    OptionalAttr<StrAttr>:$name);
  let results = (outs ChannelType:$output);
  let assemblyFormat = [{
    $clk `,` $rst `,` $input attr-dict
      `:` qualified(type($input)) `->` qualified(type($output))
  }];
  let hasVerifier = 1;
}

def PipelineStageOp : ESI_Physical_Op<"stage", [
    Pure,
    DeclareOpInterfaceMethods<ChannelOpInterface>,
    AllTypesMatch<["input", "output"]>,
  ]> {
  let summary = "An elastic buffer stage.";
  let description = [{
    An individual elastic pipeline register. Generally lowered to from a
    ChannelBuffer ('buffer'), though can be inserted anywhere to add an
    additional pipeline stage. Adding individually could be useful for
    late-pass latency balancing.
  }];

  let arguments = (ins ClockType:$clk, I1:$rst, ChannelType:$input);
  let results = (outs ChannelType:$output);
  let assemblyFormat = [{
    $clk `,` $rst `,` $input attr-dict
      `:` qualified(type($input))
  }];
}

//===----------------------------------------------------------------------===//
// Misc operations
//===----------------------------------------------------------------------===//

def FIFOOp : ESI_Physical_Op<"fifo", []> {
  let summary = "A FIFO with ESI channel connections";
  let description = [{
    A FIFO is a first-in-first-out buffer. This operation is a simple FIFO
    which can be used to connect two ESI channels. The ESI channels MUST have
    FIFO signaling semantics.
  }];

  let arguments = (ins
    ClockType:$clk, I1:$rst, ChannelType:$input,
    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$depth
  );
  let results = (outs ChannelType:$output);

  let assemblyFormat = [{
    `in` $input `clk` $clk `rst` $rst `depth` $depth attr-dict
      `:` type($input) `->` type($output)
  }];
  let hasVerifier = 1;
}

def CosimToHostEndpointOp : ESI_Physical_Op<"cosim.to_host", []> {
  let summary = "Co-simulation endpoint sending data to the host.";
  let description = [{
    A co-simulation endpoint is a connection from the simulation to some
    outside process, usually a software application responsible for driving
    the simulation (driver).

    It is uni-directional, in this case sending data from the simulation to the
    host.

    NOTE: $id MUST be unique across all endpoints at simulation runtime.
  }];

  let arguments = (ins ClockType:$clk, I1:$rst,
                       ChannelType:$toHost, StrAttr:$id);

  let assemblyFormat = [{
    $clk `,` $rst `,` $toHost`,` $id attr-dict `:` qualified(type($toHost))
  }];
  let hasVerifier = 1;
}

def CosimFromHostEndpointOp : ESI_Physical_Op<"cosim.from_host", []> {
  let summary = "Co-simulation endpoint receiving data from the host";
  let description = [{
    A co-simulation endpoint is a connection from the simulation to some
    outside process, usually a software application responsible for driving
    the simulation (driver).

    It is uni-directional, in this case receiving data from the host for the
    simulation.

    NOTE: $id MUST be unique across all endpoints at simulation runtime.
  }];

  let arguments = (ins ClockType:$clk, I1:$rst, StrAttr:$id);
  let results = (outs ChannelType:$fromHost);

  let assemblyFormat = [{
    $clk `,` $rst `,` $id attr-dict `:` qualified(type($fromHost))
  }];
  let hasVerifier = 1;
}

def NullSourceOp : ESI_Physical_Op<"null", [Pure]> {
  let summary = "An op which never produces messages.";

  let arguments = (ins);
  let results = (outs ChannelType:$out);

  let assemblyFormat = [{ attr-dict `:` qualified(type($out)) }];
}

def SinkChannelAttr : AttrDef<ESI_Dialect, "NullChannel"> {
  let summary = "An attribute which indicates a channel is unconnected.";
  let mnemonic = "null";
  let parameters = (ins "TypeAttr":$type);
  let assemblyFormat = [{ $type }];
}

#endif // CIRCT_DIALECT_ESI_CHANNELS_TD
