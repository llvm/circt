//===- ESIStdServices.td - ESI standard services ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESISTDSERVICES_TD
#define CIRCT_DIALECT_ESI_ESISTDSERVICES_TD

include "circt/Dialect/ESI/ESIServices.td"
include "circt/Dialect/HW/HWTypes.td"

def RandomAccessMemoryDeclOp: ESI_Op<"mem.ram",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
           SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Random access memory service";
  let description = [{
    Declares a service which is backed by a memory of some sort. Allows random
    access of the inner elements.

    Ports:
      read(address: clog2(depth)) -> data: innerType
      write({address: clog2(depth), data: innerType}) -> done: i0

    Users can ensure R/W ordering by waiting for the write "done" message before
    issuing a potentially dependant read. Ordering of R/W messages in flight is
    undefined.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttr:$innerType,
    I64Attr:$depth);

  let assemblyFormat = [{
    $sym_name $innerType `x` $depth attr-dict
  }];

  let extraClassDeclaration = [{
    ServicePortInfo readPortInfo();
    ServicePortInfo writePortInfo();
    static constexpr auto ReqDirChannelIdx = 0;
    static constexpr auto RespDirChannelIdx = 0;
  }];
}

def ChannelServiceDeclOp : ESI_Op<"service.std.channel",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
           SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Channel service";
  let description = [{
    Declares a service which provides a single channel connection between the
    host and client in either direction.

    ports:
      from_host -> channel<any>
      to_host <- channel<any>
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.channel"; }
  }];
}

def FuncServiceDeclOp : ESI_Op<"service.std.func",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
           SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Function service";
  let description = [{
    Declares a service which provides a function call interface to a client.
    
    Ports:
      to_client call(args: any) -> result: any
        Client exposes a function call interface to the user and does not allow
        out-of-order returns.

      TODO: ports for out-of-order returns
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.func"; }
  }];
}

def CallServiceDeclOp : ESI_Op<"service.std.call",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
           SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Service against which hardware can call into software";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.call"; }
  }];
}

def MMIOServiceDeclOp: ESI_Op<"service.std.mmio",
          [HasParent<"::mlir::ModuleOp">, SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "MMIO service";
  let description = [{
    Declares a service to be backed by a MMIO interface, which is platform
    dependent. Must be implemented by a BSP.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.mmio"; }
  }];
}

def HostMemServiceDeclOp: ESI_Op<"service.std.hostmem",
          [HasParent<"::mlir::ModuleOp">, SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Host memory service";
  let description = [{
    Declares a service to read/write host memory. Used for DMA services. Must be
    implemented by a BSP.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    ServicePortInfo readPortInfo();
    ServicePortInfo writePortInfo();
    ServicePortInfo readListPortInfo();
    std::optional<StringRef> getTypeName() { return "esi.service.std.hostmem"; }
    LogicalResult verifyRequest(const ServicePortInfo &portInfo,
                                ChannelBundleType reqBundleType,
                                Operation *reqOp);
  }];
}

def DramServiceDeclOp: ESI_Op<"service.std.dram",
          [HasParent<"::mlir::ModuleOp">, SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Device (on-board) DRAM service";
  let description = [{
    Declares a service to read/write device-attached DRAM. Must be implemented
    by a BSP. The client-facing ports are identical to those of
    `esi.service.std.hostmem` (`read`, `read_list`, and `write`) with one
    addition: a single-message `write` request struct may carry an optional
    `byteenable` field (a signless integer with one bit per byte of `data`,
    bit 0 corresponding to the least significant byte) to perform sparse,
    masked writes. Bytes whose enable bit is clear are left unmodified.

    Addresses are byte addresses, but every request address must be aligned to
    the implementation's memory word (its data bus width); the low-order bits
    of an unaligned address are ignored. The service does not shift data:
    request data starts at the first byte of the addressed word, so a client
    writes part of a word by supplying word-positioned data and a `byteenable`
    mask. Burst (list) elements are laid out at a stride of the element size
    rounded up to a whole number of words, so every element starts on a word.

    Each `esi.service.std.dram` symbol represents one physical DRAM channel.
    Clients use a channel by requesting ports of that channel's symbol, and the
    BSP implements each symbol separately. The manifest therefore records which
    channel each client uses.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    ServicePortInfo readPortInfo();
    ServicePortInfo writePortInfo();
    ServicePortInfo readListPortInfo();
    std::optional<StringRef> getTypeName() { return "esi.service.std.dram"; }
    LogicalResult verifyRequest(const ServicePortInfo &portInfo,
                                ChannelBundleType reqBundleType,
                                Operation *reqOp);
  }];
}

def TelemetryServiceDeclOp: ESI_Op<"service.std.telemetry",
          [HasParent<"::mlir::ModuleOp">, SymbolName, SymbolVisibility, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Telemetry service";
  let description = [{
    Declares a service to send telemetry data. Has one port 'report' for
    something to request telemetry data (via a 'get' channel to the client and a
    'data' channel for the return value).
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.telemetry"; }
  }];
}

#endif // CIRCT_DIALECT_ESI_ESISTDSERVICES_TD
