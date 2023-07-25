//===- ESITypes.td - ESI type specifications ---------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI will have a rich, high level type system. Currently, it is much more
// minimal.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_TYPES_TD
#define CIRCT_DIALECT_ESI_TYPES_TD

include "circt/Dialect/ESI/ESIDialect.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/EnumAttr.td"

class ESI_Type<string name> : TypeDef<ESI_Dialect, name> { }

class ESI_Port<string name> : ESI_Type<name> {}

// Standard valid-ready signaling.
def ChannelSignalingValidReady : I32EnumAttrCase<"ValidReady", 0>;
// FIFO interface with a read latency of zero. Very similiar to valid-ready, but
// cannot assert ready if data is not available.
def ChannelSignalingFIFO0      : I32EnumAttrCase<"FIFO0",      1>;
def ChannelSignaling : I32EnumAttr<
    "ChannelSignaling",
    "ESI channel wire signaling standard",
    [ChannelSignalingValidReady, ChannelSignalingFIFO0]>{
  let cppNamespace = "::circt::esi";
}

def ChannelType : ESI_Port<"Channel"> {
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
  let parameters = (ins
      "Type":$inner,
      DefaultValuedParameter<
        "::circt::esi::ChannelSignaling",
        "::circt::esi::ChannelSignaling::ValidReady">:$signaling);

  let assemblyFormat = "`<` $inner (`,` $signaling^)? `>`";
 
  let builders = [
    TypeBuilder<(ins "Type":$type), [{
        return Base::get(type.getContext(), type,
                         ::circt::esi::ChannelSignaling::ValidReady);
      }]>,
  ];
}

def ESIAnyType : ESI_Type<"Any"> {
  let summary = "any type";
  let description = [{
    Used to state that any type is accepted. The specific type will be
    determined later in compilation.
  }];

  let mnemonic = "any";
  let assemblyFormat = "";

  let extraClassDeclaration = [{
    static AnyType get(MLIRContext *context);
  }];
}

def ESIListType : ESI_Type<"List"> {
  let summary = "a runtime-variably sized list";
  let description = [{
    In software, a chunk of memory with runtime-specified length. In hardware, a
    stream of runtime-specified amount of data transmitted over many cycles in
    compile-time specified specified windows (chunks).
  }];

  let mnemonic = "list";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = [{
    `<` $elementType `>`
  }];
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

#endif // CIRCT_DIALECT_ESI_TYPES_TD
