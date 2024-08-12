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

namespace {
struct ESIBuildManifestPass
    : public circt::esi::impl::ESIBuildManifestBase<ESIBuildManifestPass> {
  void runOnOperation() override;

private:
  /// Get the types of an operations, but only if the operation is relevant.
  void gatherFilters(Operation *);
  void gatherFilters(Attribute);

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
  void emitBlock(llvm::json::OStream &, Block &block);

  AppIDHierRootOp appidRoot;

  /// Get a JSON representation of the manifest.
  std::string json();

  // Type table.
  void addType(Type type) {
    if (typeLookup.count(type))
      return;
    typeLookup[type] = types.size();
    types.push_back(type);
  }
  SmallVector<Type, 8> types;
  DenseMap<Type, size_t> typeLookup;

  // Symbols which are referenced.
  DenseSet<SymbolRefAttr> symbols;

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

  // Gather the relevant types under the appid hierarchy root only. This avoids
  // scraping unnecessary types.
  appidRoot->walk([&](Operation *op) { gatherFilters(op); });

  // Also gather types from the manifest data.
  for (Region &region : mod->getRegions())
    for (Block &block : region)
      for (auto manifestInfo : block.getOps<IsManifestData>())
        gatherFilters(manifestInfo);

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
    b.create<CompressedManifestOp>(b.getUnknownLoc(),
                                   BlobAttr::get(ctxt, compressedManifest));
  } else {
    mod->emitWarning()
        << "zlib not available but required for manifest support";
  }
}

void ESIBuildManifestPass::emitNode(llvm::json::OStream &j,
                                    AppIDHierNodeOp nodeOp) {
  j.object([&] {
    j.attribute("app_id", json(nodeOp, nodeOp.getAppIDAttr()));
    j.attribute("inst_of", json(nodeOp, nodeOp.getModuleRefAttr()));
    j.attributeArray("contents",
                     [&]() { emitBlock(j, nodeOp.getChildren().front()); });
    j.attributeArray("children", [&]() {
      for (auto nodeOp : nodeOp.getChildren().front().getOps<AppIDHierNodeOp>())
        emitNode(j, nodeOp);
    });
  });
}

void ESIBuildManifestPass::emitBlock(llvm::json::OStream &j, Block &block) {
  for (auto manifestData : block.getOps<IsManifestData>())
    j.object([&] {
      j.attribute("class", manifestData.getManifestClass());
      SmallVector<NamedAttribute, 4> attrs;
      manifestData.getDetails(attrs);
      for (auto attr : attrs)
        j.attribute(attr.getName().getValue(),
                    json(manifestData, attr.getValue()));
    });
}

std::string ESIBuildManifestPass::json() {
  auto mod = getOperation();
  std::string jsonStrBuffer;
  llvm::raw_string_ostream os(jsonStrBuffer);
  llvm::json::OStream j(os, 2);

  j.objectBegin();
  j.attribute("api_version", esiApiVersion);

  j.attributeArray("symbols", [&]() {
    // First, gather all of the manifest data for each symbol.
    DenseMap<SymbolRefAttr, SmallVector<IsManifestData>> symbolInfoLookup;
    for (auto symInfo : mod.getBody()->getOps<IsManifestData>()) {
      FlatSymbolRefAttr sym = symInfo.getSymbolRefAttr();
      if (!sym || !symbols.contains(sym))
        continue;
      symbolInfoLookup[sym].push_back(symInfo);
    }

    // Now, emit a JSON object for each symbol.
    for (const auto &symNameInfo : symbolInfoLookup) {
      j.object([&] {
        j.attribute("symbol", json(symNameInfo.second.front(),
                                   symNameInfo.first, /*elideType=*/true));
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

  j.attributeObject("design", [&]() {
    j.attribute("inst_of", json(appidRoot, appidRoot.getTopModuleRefAttr()));
    j.attributeArray("contents",
                     [&]() { emitBlock(j, appidRoot.getChildren().front()); });
    j.attributeArray("children", [&]() {
      for (auto nodeOp :
           appidRoot.getChildren().front().getOps<AppIDHierNodeOp>())
        emitNode(j, nodeOp);
    });
  });

  j.attributeArray("service_decls", [&]() {
    for (auto svcDecl : mod.getBody()->getOps<ServiceDeclOpInterface>()) {
      auto sym = FlatSymbolRefAttr::get(svcDecl);
      if (!symbols.contains(sym))
        continue;
      j.object([&] {
        j.attribute("symbol", sym.getValue());
        std::optional<StringRef> typeName = svcDecl.getTypeName();
        if (typeName)
          j.attribute("type_name", *typeName);
        llvm::SmallVector<ServicePortInfo, 8> ports;
        svcDecl.getPortList(ports);
        j.attributeArray("ports", [&]() {
          for (auto port : ports) {
            j.object([&] {
              j.attribute("name", port.port.getName().getValue());
              j.attribute("type", json(svcDecl, TypeAttr::get(port.type)));
            });
          }
        });
      });
    }
  });

  j.attributeArray("types", [&]() {
    for (auto type : types) {
      j.value(json(mod, type, /*useTable=*/false));
    }
  });
  j.objectEnd();

  return jsonStrBuffer;
}

void ESIBuildManifestPass::gatherFilters(Operation *op) {
  for (auto oper : op->getOperands())
    addType(oper.getType());
  for (auto res : op->getResults())
    addType(res.getType());

  // If op is a manifest data op, we only need to include types found in the
  // details it reports.
  SmallVector<NamedAttribute> attrs;
  if (auto manifestData = dyn_cast<IsManifestData>(op))
    manifestData.getDetails(attrs);
  else
    llvm::append_range(attrs, op->getAttrs());
  for (auto attr : attrs)
    gatherFilters(attr.getValue());
}

// NOLINTNEXTLINE(misc-no-recursion)
void ESIBuildManifestPass::gatherFilters(Attribute attr) {
  // This is far from complete. Build out as necessary.
  TypeSwitch<Attribute>(attr)
      .Case([&](TypeAttr a) { addType(a.getValue()); })
      .Case([&](IntegerAttr a) { addType(a.getType()); })
      .Case([&](FlatSymbolRefAttr a) { symbols.insert(a); })
      .Case([&](hw::InnerRefAttr a) { symbols.insert(a.getModuleRef()); })
      .Case([&](ArrayAttr a) {
        for (auto attr : a)
          gatherFilters(attr);
      })
      .Case([&](DictionaryAttr a) {
        for (const auto &entry : a.getValue())
          gatherFilters(entry.getValue());
      });
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
            return Object({{"inner", json(errorOp, t.getInner(), useTable)}});
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
    o["hw_bitwidth"] = width;

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
          .Case([&](StringAttr a) { return a.getValue(); })
          .Case([&](IntegerAttr a) { return a.getValue().getLimitedValue(); })
          .Case([&](TypeAttr a) { return json(errorOp, a.getValue()); })
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
          .Case([&](hw::InnerRefAttr ref) {
            llvm::json::Object dict;
            dict["outer_sym"] = ref.getModule().getValue();
            dict["inner"] = ref.getName().getValue();
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
  dict["type"] = json(errorOp, typedAttr.getType());
  return dict;
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIBuildManifestPass() {
  return std::make_unique<ESIBuildManifestPass>();
}
