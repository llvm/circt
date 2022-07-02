//===- ESIServices.td - ESI services ops -------------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"

def ServiceDeclOp : ESI_Op<"service.decl",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">, Symbol]> {
  let summary = "An ESI service interface declaration";
  let description = [{
    A declaration of an ESI service interface. Defines a contract between a
    service provider and its clients.

    Example:

    ```mlir
    esi.service.decl @HostComms {
      esi.service.to_server send : !esi.channel<!esi.any>
      esi.service.to_client recieve : !esi.channel<i8>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$ports);

  let assemblyFormat = [{
    $sym_name $ports attr-dict
  }];
}

def ServiceInstanceOp : ESI_Op<"service.instance"> {
  let summary = "Instantiate a server module";

  let description = [{
    Instantiate a service adhering to a service declaration interface.
    
    A pass collects all of the connection requests to the service this op
    implements from the containing modules' descendants (in the instance
    hierarchy). It bubbles them all up to the module containing this op,
    creating the necessary ESI channels as it moves them. It fills in
    `conn_requests` with the inner_syms of each connection request.

    Some later lowering is responsible for combining this op and all of the
    referenced connection request ops with a concrete implementation.

    Since implementing the server will usually need "normal" I/O, `inputs` and
    `results` act like normal `hw.instance` ports.
  }];

  let arguments = (ins FlatSymbolRefAttr:$service_symbol,
                       StrAttr:$identifier,
                       Variadic<AnyType>:$inputs,
                       OptionalAttr<FlatSymbolRefAttr>:$conn_requests);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $identifier $service_symbol `(` $inputs `)` attr-dict
      `:` functional-type($inputs, results)
  }];
}

def ToServerOp : ESI_Op<"service.to_server",
                        [HasParent<"::circt::esi::ServiceDeclOp">]> {
  let summary = "An ESI service port headed to the service";

  let arguments = (ins SymbolNameAttr:$inner_sym, TypeAttr:$type);
  let assemblyFormat = [{
    $inner_sym attr-dict `:` $type
  }];
}

def ToClientOp : ESI_Op<"service.to_client",
                        [HasParent<"::circt::esi::ServiceDeclOp">]> {
  let summary = "An ESI service port headed to a particular client";

  let arguments = (ins SymbolNameAttr:$inner_sym, TypeAttr:$type);
  let assemblyFormat = [{
    $inner_sym  attr-dict `:` $type
  }];
}

def RequestToServerConnection : ESI_Op<"service.req.to_server", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Request a connection to send data";

  let arguments = (ins HWInnerRefAttr:$servicePort, 
                       ChannelType:$sending, StrAttr:$clientName);
  let assemblyFormat = [{
    $sending `->` $servicePort `(` $clientName`)` attr-dict `:` type($sending)
  }];
}

def RequestToClientConnection : ESI_Op<"service.req.to_client", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Request a connection to receive data";

  let arguments = (ins HWInnerRefAttr:$servicePort,
                       StrAttr:$clientName);
  let results = (outs ChannelType:$receiving);
  let assemblyFormat = [{
    $servicePort `(` $clientName `)` attr-dict `:` type($receiving)
  }];
}
