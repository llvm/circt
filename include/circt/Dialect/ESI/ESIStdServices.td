//===- ESIStdServices.td - ESI standard services ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/ESI/ESIServices.td"
include "circt/Dialect/HW/HWTypes.td"

def RandomAccessMemoryDeclOp: ESI_Op<"mem.ram",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
          Symbol, DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
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

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$innerType, I64Attr:$depth);

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

def FuncServiceDeclOp : ESI_Op<"service.std.func",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
           Symbol, DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Function service";
  let description = [{
    Declares a service which provides a function call interface to a client.
    
    Ports:
      to_client call(args: any) -> result: any
        Client exposes a function call interface to the user and does not allow
        out-of-order returns.

      TODO: ports for out-of-order returns
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.func"; }
  }];
}

def CallServiceDeclOp : ESI_Op<"service.std.call",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
           Symbol, DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Service against which hardware can call into software";

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.call"; }
  }];
}

def MMIOServiceDeclOp: ESI_Op<"service.std.mmio",
          [HasParent<"::mlir::ModuleOp">, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "MMIO service";
  let description = [{
    Declares a service to be backed by a MMIO interface, which is platform
    dependent. Must be implemented by a BSP.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.mmio"; }
  }];
}

def HostMemServiceDeclOp: ESI_Op<"service.std.hostmem",
          [HasParent<"::mlir::ModuleOp">, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Host memory service";
  let description = [{
    Declares a service to read/write host memory. Used for DMA services. Must be
    implemented by a BSP.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    ServicePortInfo readPortInfo();
    ServicePortInfo writePortInfo();
    std::optional<StringRef> getTypeName() { return "esi.service.std.hostmem"; }
  }];
}

def TelemetryServiceDeclOp: ESI_Op<"service.std.telemetry",
          [HasParent<"::mlir::ModuleOp">, Symbol,
           DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Telemetry service";
  let description = [{
    Declares a service to send telemetry data. Has one port 'report' for
    something to request telemetry data (via a 'get' channel to the client and a
    'data' channel for the return value).
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<StringRef> getTypeName() { return "esi.service.std.telemetry"; }
  }];
}
