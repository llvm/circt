//===- ESIManifest.td - System manifest stuff ---------------- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_MANIFEST_TD
#define CIRCT_DIALECT_ESI_MANIFEST_TD

include "circt/Dialect/ESI/ESIChannels.td"
include "circt/Dialect/HW/HWAttributesNaming.td"

include "mlir/IR/AttrTypeBase.td"

def AppIDAttr : ESI_Attr<"AppID"> {
  let summary = "An application relevant instance identifier";
  let description = [{
    Identifies an instance which is visible through multiple hierarchy levels.
    Indended to make locating an instance easier in the instance hierarchy.
  }];

  let parameters = (ins "StringAttr":$name,
                        OptionalParameter<"std::optional<uint64_t>">:$index);
  let mnemonic = "appid";
  let assemblyFormat = [{
    `<` $name (`[` $index^ `]`)? `>`
  }];

  let extraClassDeclaration = [{
    static constexpr StringRef AppIDAttrName = "esi.appid";
  }];
}

def AppIDArrayAttr :
  TypedArrayAttrBase<AppIDAttr, "Array of AppIDs">;

def AppIDPathAttr : ESI_Attr<"AppIDPath"> {
  let summary = "An application-specific hierarchical path through a design";
  let description = [{
    A list of AppID components which specifies a specific dynamic instance
    in the design.
  }];

  let parameters = (ins "FlatSymbolRefAttr":$root,
                        ArrayRefParameter<"AppIDAttr">:$path);
  let mnemonic = "appid_path";
  let assemblyFormat = [{
    `<` $root `[` $path `]` `>`
  }];

  let extraClassDeclaration = [{
    // Get an AppIDPathAttr without the last component. Returns None if the path
    // is simply the root.
    AppIDPathAttr getParent();
  }];
}

def BundleDirection : I32EnumAttr<"BundleDirection",
  "Direction of original request", [
    I32EnumAttrCase<"toServer", 1>,
    I32EnumAttrCase<"toClient", 2>,
  ]> {
  let cppNamespace = "::circt::esi";
}

def ServiceRequestRecordOp : ESI_Op<"manifest.req", [
        DeclareOpInterfaceMethods<IsManifestData>,
        DeclareOpInterfaceMethods<HasAppIDOpInterface>]> {
  let summary = "Record of a service request";
  let description = [{
    A record of a service request, including the requestor, the service
    requested, and the parameters of the request. Emitted before connecting the
    service to preserve metadata about the original request.
  }];

  let arguments = (ins AppIDAttr:$requestor,
                       InnerRefAttr:$servicePort,
                       OptionalAttr<StrAttr>:$stdService,
                       BundleDirection:$direction,
                       TypeAttrOf<ChannelBundleType>:$bundleType);

  let assemblyFormat = [{
    qualified($requestor) `,` $servicePort (`std` $stdService^)?
    `,` $direction `,` $bundleType attr-dict
  }];

  let extraClassDeclaration = [{
    AppIDAttr getAppID() {
      return getRequestor();
    }
    // Misc information about the original request which must appear in the
    // manifest.
    void getDetails(SmallVectorImpl<NamedAttribute> &results);
  }];
}

def ServiceImplRecordOp : ESI_Op<"manifest.service_impl", [
        NoTerminator, 
        DeclareOpInterfaceMethods<IsManifestData>,
        DeclareOpInterfaceMethods<HasAppIDOpInterface>]> {
  let summary = "Record of a service implementation";
  let description = [{
    A record of a service implementation. Optionally emitted by the service
    implementation. Contains information necessary to connect to the service and
    service clients.
  }];

  let arguments = (ins AppIDAttr:$appID,
                       OptionalAttr<FlatSymbolRefAttr>:$service,
                       OptionalAttr<StrAttr>:$stdService,
                       StrAttr:$serviceImplName,
                       DictionaryAttr:$implDetails);
  let regions = (region SizedRegion<1>:$reqDetails);
  let assemblyFormat = [{
    qualified($appID) (`svc` $service^)? (`std` $stdService^)?
   `by` $serviceImplName `with` $implDetails
    attr-dict-with-keyword custom<ServiceImplRecordReqDetails>($reqDetails)
  }];

  let extraClassDeclaration = [{
    // Get information which needs to appear in the manifest for the host to
    // connect to this service.
    void getDetails(SmallVectorImpl<NamedAttribute> &results);
  }];
}

def ServiceImplClientRecordOp : ESI_Op<"manifest.impl_conn", [
        DeclareOpInterfaceMethods<IsManifestData>,
        HasParent<"ServiceImplRecordOp">]> {
  let summary = "Details of a service implementation client connection";
  let description = [{
    A record containing all the necessary details of how to connect to a client
    which the parent service record is servicing.
  }];

  let arguments = (ins AppIDArrayAttr:$relAppIDPath,
                       InnerRefAttr:$servicePort,
                       TypeAttrOf<ChannelBundleType>:$bundleType,
                       DictionaryAttr:$implDetails);
  let assemblyFormat = [{
    $relAppIDPath `req` $servicePort `(` $bundleType `)`
      `with` $implDetails attr-dict
  }];

  let extraClassDeclaration = [{
    // Get information which needs to appear in the manifest for the host to
    // connect to this client through the parent service.
    void getDetails(SmallVectorImpl<NamedAttribute> &results);
  }];
}

def AppIDHierRootOp : ESI_Op<"manifest.hier_root",
                             [HasParent<"mlir::ModuleOp">,
                             NoTerminator, SingleBlock]> {
  let summary = "The root of an appid instance hierarchy";

  let arguments = (ins FlatSymbolRefAttr:$topModuleRef);
  let regions = (region SizedRegion<1>:$children);

  let assemblyFormat = [{
    $topModuleRef attr-dict-with-keyword $children
  }];
}

def AppIDHierNodeOp : ESI_Op<"manifest.hier_node", [
      ParentOneOf<["circt::esi::AppIDHierRootOp",
                   "circt::esi::AppIDHierNodeOp"]>,
      NoTerminator, SingleBlock]> {
  let summary = "A node in the AppID hierarchy";

  let arguments = (ins AppIDAttr:$appID, FlatSymbolRefAttr:$moduleRef);
  let regions = (region SizedRegion<1>:$children);

  let assemblyFormat = [{
    qualified($appID) `mod` $moduleRef attr-dict-with-keyword $children
  }];
}

def SymbolMetadataOp : ESI_Op<"manifest.sym", [
  DeclareOpInterfaceMethods<IsManifestData>]> {
  let summary = "Metadata about a symbol";
  let description = [{
    Metadata about a symbol, including its name, repository, commit hash,
    version, and summary. All are optional, but strongly encouraged. Any
    additional metadata which users wish to attach should go as discardable
    attributes.
  }];

  let arguments = (ins FlatSymbolRefAttr:$symbolRef,
                       OptionalAttr<StrAttr>:$name,
                       OptionalAttr<StrAttr>:$repo,
                       OptionalAttr<StrAttr>:$commitHash,
                       OptionalAttr<StrAttr>:$version,
                       OptionalAttr<StrAttr>:$summary);
                       
  let assemblyFormat = [{
    $symbolRef
    (`name` $name^)?
    (`repo` $repo^)?
    (`commit` $commitHash^)?
    (`version` $version^)?
    (`summary` $summary^)?
    attr-dict
  }];
}

def BlobAttr : ESI_Attr<"Blob"> {
  let summary = "A binary blob";

  let parameters = (ins ArrayRefParameter<"char">:$data);
  let mnemonic = "blob";
  let hasCustomAssemblyFormat = 1;
}

def CompressedManifestOp : ESI_Op<"manifest.compressed", []> {
  let summary = "A zlib-compressed JSON manifest";
  let arguments = (ins BlobAttr:$compressedManifest);
  let assemblyFormat = [{
    $compressedManifest attr-dict
  }];
}

#endif // CIRCT_DIALECT_ESI_MANIFEST_TD
