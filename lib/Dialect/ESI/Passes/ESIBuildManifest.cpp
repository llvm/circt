//===- ESIBuildManifest.cpp - Build ESI system manifest ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/AppID.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"

#include "circt/Dialect/HW/HWSymCache.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/JSON.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_ESIBUILDMANIFEST
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace esi;

// TODO: The code herein is a bit ugly, but it works. Consider cleaning it up.

namespace {
struct ESIBuildManifestPass
    : public circt::esi::impl::ESIBuildManifestBase<ESIBuildManifestPass> {
  void runOnOperation() override;

private:
  /// Get a JSON representation of a type. 'useTable' indicates whether to use
  /// the type table to determine if the type should be emitted as a reference
  /// if it already exists in the type table.
  llvm::json::Value json(Operation *errorOp, Type, bool useTable = true);
  /// Get a JSON representation of a type. 'elideType' indicates to not print
  /// the type if it would have been printed.
  llvm::json::Value json(Operation *errorOp, Attribute, bool elideType = false);

  // Output a node in the appid hierarchy.
  void emitNode(llvm::json::OStream &, AppIDHierNodeOp nodeOp,
                ArrayRef<Attribute> parentPath);
  // Output the manifest data of a node in the appid hierarchy. 'nodePath' is
  // the absolute appID path of the node which owns 'block'.
  void emitBlock(llvm::json::OStream &, Block &block, StringRef manifestClass,
                 ArrayRef<Attribute> nodePath);

  AppIDHierRootOp appidRoot;

  /// Get a JSON representation of the manifest.
  std::string json();

  /// Walk the appID hierarchy and record the absolute appID paths of all client
  /// ports whose channels are bound to a host-reachable engine (i.e. which
  /// appear in some engine's channel assignment table). These are the only
  /// ports which are "user accessible": present in the runtime's Accelerator
  /// design tree.
  void collectAssignedPorts(Block &block, SmallVectorImpl<Attribute> &path);

  /// Walk the appID hierarchy and populate 'keepNodes' with the hierarchy nodes
  /// which contain (or have a descendant which contains) a user-accessible
  /// client port. Returns true if 'block' or any descendant is to be kept.
  bool computeKeepNodes(Block &block, SmallVectorImpl<Attribute> &path);

  /// Is the client port with appID 'portID' under the node at 'nodePath' user
  /// accessible?
  bool isPortAccessible(ArrayRef<Attribute> nodePath, Attribute portID) {
    SmallVector<Attribute> portPath(nodePath.begin(), nodePath.end());
    portPath.push_back(portID);
    return assignedPortPaths.contains(ArrayAttr::get(&getContext(), portPath));
  }

  // Type table.
  std::string useType(Type type) {
    std::string typeID;
    llvm::raw_string_ostream(typeID) << type;

    if (typeLookup.count(type))
      return typeID;
    typeLookup[type] = types.size();
    types.push_back(type);
    return typeID;
  }
  SmallVector<Type, 8> types;
  DenseMap<Type, size_t> typeLookup;

  // Symbols / modules which are referenced.
  DenseSet<SymbolRefAttr> symbols;
  DenseSet<SymbolRefAttr> modules;

  // Absolute appID paths of user-accessible client ports (those bound to a
  // host-reachable engine via a channel assignment).
  DenseSet<ArrayAttr> assignedPortPaths;
  // AppID hierarchy nodes to retain (those whose subtree contains a
  // user-accessible client port).
  DenseSet<Operation *> keepNodes;

  hw::HWSymbolCache symCache;
};
} // anonymous namespace

void ESIBuildManifestPass::runOnOperation() {
  MLIRContext *ctxt = &getContext();
  Operation *mod = getOperation();
  symCache.addDefinitions(mod);
  symCache.freeze();

  // Find the top level appid hierarchy root.
  for (auto root : mod->getRegion(0).front().getOps<AppIDHierRootOp>())
    if (root.getTopModuleRef() == top)
      appidRoot = root;
  if (!appidRoot)
    return;

  // Determine which client ports are user-accessible and which hierarchy nodes
  // need to be retained as a result. This must happen before JSONification
  // since the latter relies on these sets to prune the manifest.
  {
    SmallVector<Attribute> path;
    collectAssignedPorts(appidRoot.getChildren().front(), path);
    path.clear();
    computeKeepNodes(appidRoot.getChildren().front(), path);
  }

  // JSONify the manifest.
  std::string jsonManifest = json();

  std::error_code ec;
  llvm::raw_fd_ostream os("esi_system_manifest.json", ec);
  if (ec) {
    mod->emitError() << "Failed to open file for writing: " << ec.message();
    signalPassFailure();
  } else {
    os << jsonManifest << "\n";
  }

  // If zlib is available, compress the manifest and append it to the module.
  SmallVector<uint8_t, 10 * 1024> compressedManifest;
  if (llvm::compression::zlib::isAvailable()) {
    // Compress the manifest.
    llvm::compression::zlib::compress(
        ArrayRef((uint8_t *)jsonManifest.data(), jsonManifest.length()),
        compressedManifest, llvm::compression::zlib::BestSizeCompression);

    llvm::raw_fd_ostream bos("esi_system_manifest.json.zlib", ec);
    if (ec) {
      mod->emitError() << "Failed to open compressed file for writing: "
                       << ec.message();
      signalPassFailure();
    } else {
      bos.write((char *)compressedManifest.data(), compressedManifest.size());
    }

    OpBuilder b(symCache.getDefinition(appidRoot.getTopModuleRefAttr())
                    ->getRegion(0)
                    .front()
                    .getTerminator());
    CompressedManifestOp::create(b, b.getUnknownLoc(),
                                 BlobAttr::get(ctxt, compressedManifest));
  } else {
    mod->emitWarning()
        << "zlib not available but required for manifest support";
  }
}

void ESIBuildManifestPass::collectAssignedPorts(
    Block &block, SmallVectorImpl<Attribute> &path) {
  for (auto svc : block.getOps<ServiceImplRecordOp>())
    for (auto client :
         svc.getReqDetails().front().getOps<ServiceImplClientRecordOp>()) {
      DictionaryAttr chanAssigns = client.getChannelAssignmentsAttr();
      // A client port is only user-accessible if its channels are bound to a
      // host-reachable engine, recorded via a non-empty channel assignment.
      if (!chanAssigns || chanAssigns.empty())
        continue;
      SmallVector<Attribute> portPath(path.begin(), path.end());
      for (auto id : client.getRelAppIDPath().getAsRange<AppIDAttr>())
        portPath.push_back(id);
      assignedPortPaths.insert(ArrayAttr::get(&getContext(), portPath));
    }
  for (auto node : block.getOps<AppIDHierNodeOp>()) {
    path.push_back(node.getAppIDAttr());
    collectAssignedPorts(node.getChildren().front(), path);
    path.pop_back();
  }
}

bool ESIBuildManifestPass::computeKeepNodes(Block &block,
                                            SmallVectorImpl<Attribute> &path) {
  bool keep = false;
  for (auto req : block.getOps<ServiceRequestRecordOp>())
    if (isPortAccessible(path, req.getRequestorAttr()))
      keep = true;
  for (auto node : block.getOps<AppIDHierNodeOp>()) {
    path.push_back(node.getAppIDAttr());
    bool childKeep = computeKeepNodes(node.getChildren().front(), path);
    path.pop_back();
    if (childKeep) {
      keepNodes.insert(node);
      keep = true;
    }
  }
  return keep;
}

void ESIBuildManifestPass::emitNode(llvm::json::OStream &j,
                                    AppIDHierNodeOp nodeOp,
                                    ArrayRef<Attribute> parentPath) {
  SmallVector<Attribute> nodePath(parentPath.begin(), parentPath.end());
  nodePath.push_back(nodeOp.getAppIDAttr());
  std::set<StringRef> classesToEmit;
  for (auto manifestData : nodeOp.getOps<IsManifestData>())
    classesToEmit.insert(manifestData.getManifestClass());
  j.object([&] {
    j.attribute("appID", json(nodeOp, nodeOp.getAppIDAttr()));
    j.attribute("instanceOf", json(nodeOp, nodeOp.getModuleRefAttr()));
    for (StringRef manifestClass : classesToEmit)
      j.attributeArray(manifestClass.str() + "s", [&]() {
        emitBlock(j, nodeOp.getChildren().front(), manifestClass, nodePath);
      });
    j.attributeArray("children", [&]() {
      for (auto childOp :
           nodeOp.getChildren().front().getOps<AppIDHierNodeOp>())
        if (keepNodes.contains(childOp))
          emitNode(j, childOp, nodePath);
    });
  });
}

void ESIBuildManifestPass::emitBlock(llvm::json::OStream &j, Block &block,
                                     StringRef manifestClass,
                                     ArrayRef<Attribute> nodePath) {
  for (auto manifestData : block.getOps<IsManifestData>()) {
    if (manifestData.getManifestClass() != manifestClass)
      continue;
    // Prune client ports which are not user-accessible (i.e. whose channels are
    // not bound to a host-reachable engine). Such ports never appear in the
    // runtime's Accelerator design tree, so they (and their types) are omitted.
    if (auto req =
            dyn_cast<ServiceRequestRecordOp>(manifestData.getOperation()))
      if (!isPortAccessible(nodePath, req.getRequestorAttr()))
        continue;
    j.object([&] {
      SmallVector<NamedAttribute, 4> attrs;
      manifestData.getDetails(attrs);
      for (auto attr : attrs)
        j.attribute(attr.getName().getValue(),
                    json(manifestData, attr.getValue()));
    });
  }
}

std::string ESIBuildManifestPass::json() {
  auto mod = getOperation();
  std::string jsonStrBuffer;
  llvm::raw_string_ostream os(jsonStrBuffer);
  llvm::json::OStream j(os, 2);

  j.objectBegin(); // Top level object.
  j.attribute("apiVersion", esiApiVersion);

  std::set<StringRef> classesToEmit;
  for (auto manifestData : appidRoot.getOps<IsManifestData>())
    classesToEmit.insert(manifestData.getManifestClass());

  j.attributeObject("design", [&]() {
    j.attribute("instanceOf", json(appidRoot, appidRoot.getTopModuleRefAttr()));
    modules.insert(appidRoot.getTopModuleRefAttr());
    for (StringRef manifestClass : classesToEmit)
      j.attributeArray(manifestClass.str() + "s", [&]() {
        emitBlock(j, appidRoot.getChildren().front(), manifestClass, {});
      });
    j.attributeArray("children", [&]() {
      for (auto nodeOp :
           appidRoot.getChildren().front().getOps<AppIDHierNodeOp>())
        if (keepNodes.contains(nodeOp))
          emitNode(j, nodeOp, {});
    });
  });

  j.attributeArray("serviceDeclarations", [&]() {
    for (auto svcDecl : mod.getBody()->getOps<ServiceDeclOpInterface>()) {
      auto sym = FlatSymbolRefAttr::get(svcDecl);
      if (!symbols.contains(sym))
        continue;
      j.object([&] {
        j.attribute("symbol", json(svcDecl, sym, /*elideType=*/true));
        std::optional<StringRef> typeName = svcDecl.getTypeName();
        if (typeName)
          j.attribute("serviceName", *typeName);
        llvm::SmallVector<ServicePortInfo, 8> ports;
        svcDecl.getPortList(ports);
        j.attributeArray("ports", [&]() {
          for (auto port : ports) {
            j.object([&] {
              j.attribute("name", port.port.getName().getValue());
              // Emit the port type as a plain string rather than registering it
              // in the type table: the runtime never resolves service
              // declaration port types, so they would only bloat the manifest.
              std::string typeID;
              llvm::raw_string_ostream(typeID) << port.type;
              j.attribute("typeID", typeID);
            });
          }
        });
      });
    }
  });

  j.attributeArray("modules", [&]() {
    // Map from symbol to all manifest data ops related to said symbol.
    llvm::MapVector<SymbolRefAttr, SmallVector<IsManifestData>>
        symbolInfoLookup;
    // Ensure that all symbols are present in the lookup even if they have no
    // manifest metadata.
    for (auto modSymbol : modules)
      symbolInfoLookup[modSymbol] = {};
    // Gather all manifest data for each symbol.
    for (auto symInfo : mod.getBody()->getOps<IsManifestData>()) {
      FlatSymbolRefAttr sym = symInfo.getSymbolRefAttr();
      if (!sym)
        continue;
      // Keep a module's metadata if it is instantiated somewhere in the
      // (pruned) design hierarchy. Additionally, always keep module constants:
      // they are semantically significant (e.g. for codegen) even when the
      // module itself is not user-accessible.
      bool isConstants = symInfo.getManifestClass() == "symConsts";
      if (!symbols.contains(sym) && !isConstants)
        continue;
      symbolInfoLookup[sym].push_back(symInfo);
    }

    // Now, emit a JSON object for each symbol.
    for (const auto &symNameInfo : symbolInfoLookup) {
      j.object([&] {
        std::string symbolStr;
        llvm::raw_string_ostream(symbolStr) << symNameInfo.first;
        j.attribute("symbol", symbolStr);
        for (auto symInfo : symNameInfo.second) {
          j.attributeBegin(symInfo.getManifestClass());
          j.object([&] {
            SmallVector<NamedAttribute, 4> attrs;
            symInfo.getDetails(attrs);
            for (auto attr : attrs) {
              if (attr.getName().getValue() == "symbolRef")
                continue;
              j.attribute(attr.getName().getValue(),
                          json(symInfo, attr.getValue()));
            }
          });
          j.attributeEnd();
        }
      });
    }
  });

  j.attributeArray("types", [&]() {
    for (size_t i = 0; i < types.size(); i++)
      j.value(json(mod, types[i], /*useTable=*/false));
  });

  j.objectEnd(); // Top level object.

  return jsonStrBuffer;
}

/// Get a JSON representation of a type.
// NOLINTNEXTLINE(misc-no-recursion)
llvm::json::Value ESIBuildManifestPass::json(Operation *errorOp, Type type,
                                             bool useTable) {
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  if (useTable && typeLookup.contains(type)) {
    // If the type is in the type table, it'll be present in the types
    // section. Just give the circt type name, which is guaranteed to
    // uniquely identify the type.
    std::string typeName;
    llvm::raw_string_ostream(typeName) << type;
    return typeName;
  }

  std::string m;
  Object o =
      // This is not complete. Build out as necessary.
      TypeSwitch<Type, Object>(type)
          .Case([&](ChannelType t) {
            m = "channel";
            return Object({{"inner", json(errorOp, t.getInner(), false)}});
          })
          .Case([&](ChannelBundleType t) {
            m = "bundle";
            Array channels;
            for (auto field : t.getChannels())
              channels.push_back(Object(
                  {{"name", field.name.getValue()},
                   {"direction", stringifyChannelDirection(field.direction)},
                   {"type", json(errorOp, field.type, useTable)}}));
            return Object({{"channels", Value(std::move(channels))}});
          })
          .Case([&](AnyType t) {
            m = "any";
            return Object();
          })
          .Case([&](ListType t) {
            m = "list";
            return Object(
                {{"element", json(errorOp, t.getElementType(), useTable)}});
          })
          .Case([&](hw::ArrayType t) {
            m = "array";
            return Object(
                {{"size", t.getNumElements()},
                 {"element", json(errorOp, t.getElementType(), useTable)}});
          })
          .Case([&](hw::StructType t) {
            m = "struct";
            Array fields;
            for (auto field : t.getElements())
              fields.push_back(
                  Object({{"name", field.name.getValue()},
                          {"type", json(errorOp, field.type, useTable)}}));
            return Object({{"fields", Value(std::move(fields))}});
          })
          .Case([&](hw::UnionType t) {
            m = "union";
            Array fields;
            for (auto field : t.getElements())
              fields.push_back(
                  Object({{"name", field.name.getValue()},
                          {"type", json(errorOp, field.type, useTable)}}));
            return Object({{"fields", Value(std::move(fields))}});
          })
          .Case([&](hw::TypeAliasType t) {
            m = "alias";
            return Object(
                {{"name", t.getTypeDecl(symCache).getPreferredName()},
                 {"inner", json(errorOp, t.getInnerType(), useTable)}});
          })
          .Case([&](IntegerType t) {
            m = "int";
            StringRef signedness =
                t.isSigned() ? "signed"
                             : (t.isUnsigned() ? "unsigned" : "signless");
            return Object({{"signedness", signedness}});
          })
          .Case([&](WindowType t) {
            m = "window";
            Array frames;
            for (auto frame : t.getFrames()) {
              Array fields;
              for (auto field : frame.getMembers()) {
                Object fieldObj{{"name", field.getFieldName().getValue()}};
                if (field.getNumItems() != 0)
                  fieldObj["numItems"] = field.getNumItems();
                if (field.getBulkCountWidth() != 0)
                  fieldObj["bulkCountWidth"] = field.getBulkCountWidth();
                fields.push_back(Value(std::move(fieldObj)));
              }
              frames.push_back(
                  Value(Object({{"name", frame.getName().getValue()},
                                {"fields", Value(std::move(fields))}})));
            }
            return Object(
                {{"name", t.getName().getValue()},
                 {"into", json(errorOp, t.getInto(), useTable)},
                 {"frames", Value(std::move(frames))},
                 {"loweredType", json(errorOp, t.getLoweredType(), useTable)}});
          })
          .Default([&](Type t) {
            errorOp->emitWarning()
                << "ESI system manifest: unknown type: " << t;
            return Object();
          });

  // Common metadata.
  std::string typeID;
  llvm::raw_string_ostream(typeID) << type;
  o["id"] = typeID;

  // Get the hw bitwidth if possible.
  Type hwType = type;
  if (auto chanType = dyn_cast<ChannelType>(type))
    hwType = chanType.getInner();
  if (auto winType = dyn_cast<WindowType>(hwType))
    hwType = winType.getLoweredType();
  int64_t width = hw::getBitWidth(hwType);
  if (width >= 0)
    o["hwBitwidth"] = width;

  o["dialect"] = type.getDialect().getNamespace();
  if (m.length())
    o["mnemonic"] = m;
  return o;
}

// Serialize an attribute to a JSON value.
// NOLINTNEXTLINE(misc-no-recursion)
llvm::json::Value ESIBuildManifestPass::json(Operation *errorOp, Attribute attr,
                                             bool elideType) {

  // This is far from complete. Build out as necessary.
  using llvm::json::Object;
  using llvm::json::Value;
  Value value =
      TypeSwitch<Attribute, Value>(attr)
          .Case([&](FlatSymbolRefAttr ref) {
            symbols.insert(ref);
            std::string value;
            llvm::raw_string_ostream(value) << ref;
            return value;
          })
          .Case([&](StringAttr a) { return a.getValue(); })
          .Case([&](IntegerAttr a) { return a.getValue().getLimitedValue(); })
          .Case([&](TypeAttr a) { return useType(a.getValue()); })
          .Case([&](ArrayAttr a) {
            return llvm::json::Array(llvm::map_range(
                a, [&](Attribute a) { return json(errorOp, a); }));
          })
          .Case([&](DictionaryAttr a) {
            llvm::json::Object dict;
            for (const auto &entry : a.getValue())
              dict[entry.getName().getValue()] =
                  json(errorOp, entry.getValue());
            return dict;
          })
          .Case([&](AppIDAttr appid) {
            llvm::json::Object dict;
            dict["name"] = appid.getName().getValue();
            auto idx = appid.getIndex();
            if (idx)
              dict["index"] = *idx;
            return dict;
          })
          .Default([&](Attribute a) {
            std::string value;
            llvm::raw_string_ostream(value) << a;
            return value;
          });

  // Don't print the type if it's None or we're eliding it.
  auto typedAttr = llvm::dyn_cast<TypedAttr>(attr);
  if (elideType || !typedAttr || isa<NoneType>(typedAttr.getType()))
    return value;

  // Otherwise, return an object with the value and type.
  Object dict;
  dict["value"] = value;
  dict["type"] = useType(typedAttr.getType());
  return dict;
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIBuildManifestPass() {
  return std::make_unique<ESIBuildManifestPass>();
}
