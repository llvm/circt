//===- ESIInterfaces.td - ESI Interfaces -----------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the ESI dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_INTERFACES_TD
#define CIRCT_DIALECT_ESI_INTERFACES_TD

include "mlir/IR/OpBase.td"

def ChannelOpInterface : OpInterface<"ChannelOpInterface"> {
  let cppNamespace = "circt::esi";
  let description =
      [{"An interface for operations which carries channel semantics."}];

  let methods = [
    InterfaceMethod<
      [{"Returns the channel type of this operation."}],
      "circt::esi::ChannelType", "channelType"
    >,
    InterfaceMethod<
        [{"Returns the inner type of this channel. This will be the type of the
           data value of the channel, if the channel carries data semantics. Else,
           return NoneType."}],
        "mlir::Type",
        "innerType",
        (ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.channelType().getInner();
        }]>
  ];
}

def HasAppIDOpInterface : OpInterface<"HasAppID"> {
  let cppNamespace = "circt::esi";
  let description = [{
    Op can be identified by an AppID.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the AppID of this operation.",
      "::circt::esi::AppIDAttr", "getAppID", (ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.getAppID();
        }]
    >,
  ];
}

def IsManifestData : OpInterface<"IsManifestData"> {
  let cppNamespace = "circt::esi";
  let description = [{
    Op's attributes should be represented in the manifest.
  }];

  let methods = [
    StaticInterfaceMethod<
      "Get the class name for this op.",
      "StringRef", "getManifestClass", (ins)
    >,
    InterfaceMethod<
      "Populate results with the manifest data.",
      "void", "getDetails", (ins "SmallVectorImpl<NamedAttribute>&":$results),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          auto attrs = $_op->getAttrs();
          results.append(attrs.begin(), attrs.end());
        }]
    >,
    InterfaceMethod<
      "Get the manifest data from this op as an attribute.",
      "DictionaryAttr", "getDetailsAsDict", (ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          SmallVector<NamedAttribute, 8> attrs;
          $_op.getDetails(attrs);
          return DictionaryAttr::get($_op.getContext(), attrs);
        }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
//
// Service-related interfaces.
//
//===----------------------------------------------------------------------===//

def ServiceDeclOpInterface : OpInterface<"ServiceDeclOpInterface"> {
  let cppNamespace = "circt::esi";
  let description = [{
    Any op which represents a service declaration should implement this
    interface.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the list of interface ports for this service interface.",
      "void", "getPortList",
      (ins "llvm::SmallVectorImpl<ServicePortInfo>&":$ports)
    >,
    InterfaceMethod<
      "Return a well-known name for this service type.",
      "std::optional<StringRef>", "getTypeName",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      "Get info on a particular port.",
      "FailureOr<ServicePortInfo>", "getPortInfo",
      (ins "StringAttr":$portName),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<ServicePortInfo> ports;
        $_op.getPortList(ports);
        auto f = llvm::find_if(ports, [&](ServicePortInfo port) {
          return port.port.getTarget() == portName;
        });
        if (f != ports.end()) return *f;
        return failure();
      }]
    >
  ];
}

#endif // CIRCT_DIALECT_ESI_INTERFACES_TD
