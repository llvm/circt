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
  void emitNode(llvm::json::OStream &, AppIDHierNodeOp nodeOp);
  // Output the manifest data of a node in the appid hierarchy.
  void emitBlock(llvm::json::OStream &, Block &block, StringRef manifestClass);

  AppIDHierRootOp appidRoot;

  /// Get a JSON representation of the manifest.
  std::string json();

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

void ESIBuildManifestPass::emitNode(llvm::json::OStream &j,
                                    AppIDHierNodeOp nodeOp) {
  std::set<StringRef> classesToEmit;
  for (auto manifestData : nodeOp.getOps<IsManifestData>())
    classesToEmit.insert(manifestData.getManifestClass());
  j.object([&] {
    j.attribute("appID", json(nodeOp, nodeOp.getAppIDAttr()));
    j.attribute("instanceOf", json(nodeOp, nodeOp.getModuleRefAttr()));
    for (StringRef manifestClass : classesToEmit)
      j.attributeArray(manifestClass.str() + "s", [&]() {
        emitBlock(j, nodeOp.getChildren().front(), manifestClass);
      });
    j.attributeArray("children", [&]() {
      for (auto nodeOp : nodeOp.getChildren().front().getOps<AppIDHierNodeOp>())
        emitNode(j, nodeOp);
    });
  });
}

void ESIBuildManifestPass::emitBlock(llvm::json::OStream &j, Block &block,
                                     StringRef manifestClass) {
  for (auto manifestData : block.getOps<IsManifestData>()) {
    if (manifestData.getManifestClass() != manifestClass)
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
        emitBlock(j, appidRoot.getChildren().front(), manifestClass);
      });
    j.attributeArray("children", [&]() {
      for (auto nodeOp :
           appidRoot.getChildren().front().getOps<AppIDHierNodeOp>())
        emitNode(j, nodeOp);
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
              j.attribute("typeID", useType(port.type));
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
      if (!sym || !symbols.contains(sym))
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
          .Default([&](Type t) {
            errorOp->emitWarning()
                << "ESI system manifest: unknown type: " << t;
            return Object();
          });

  // Common metadata.
  std::string typeID;
  llvm::raw_string_ostream(typeID) << type;
  o["id"] = typeID;

  int64_t width = hw::getBitWidth(type);
  if (auto chanType = dyn_cast<ChannelType>(type))
    width = hw::getBitWidth(chanType.getInner());
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
