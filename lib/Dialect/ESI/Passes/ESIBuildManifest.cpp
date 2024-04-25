//===- ESIBuildManifest.cpp - Build ESI system manifest ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/APIUtilities.h"
#include "circt/Dialect/ESI/AppID.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"

#include "circt/Dialect/HW/HWSymCache.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace esi;

namespace {
struct ESIBuildManifestPass
    : public ESIBuildManifestBase<ESIBuildManifestPass> {
  void runOnOperation() override;

private:
  /// Get the types of an operations, but only if the operation is relevant.
  void gatherFilters(Operation *);
  void gatherFilters(Attribute);

  /// Get a JSON representation of a type.
  llvm::json::Value json(Operation *errorOp, Type);
  /// Get a JSON representation of a type.
  llvm::json::Value json(Operation *errorOp, Attribute);

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
    for (auto symInfo : mod.getBody()->getOps<SymbolMetadataOp>()) {
      if (!symbols.contains(symInfo.getSymbolRefAttr()))
        continue;
      j.object([&] {
        SmallVector<NamedAttribute, 4> attrs;
        symInfo.getDetails(attrs);
        for (auto attr : attrs)
          j.attribute(attr.getName().getValue(),
                      json(symInfo, attr.getValue()));
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
              j.attribute("name", port.port.getTarget().getValue());
              j.attribute("type", json(svcDecl, TypeAttr::get(port.type)));
            });
          }
        });
      });
    }
  });

  j.attributeArray("types", [&]() {
    for (auto type : types) {
      j.value(json(mod, type));
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
      .Case([&](FlatSymbolRefAttr a) { symbols.insert(a); })
      .Case([&](hw::InnerRefAttr a) {
        symbols.insert(FlatSymbolRefAttr::get(a.getRoot()));
      })
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
llvm::json::Value ESIBuildManifestPass::json(Operation *errorOp, Type type) {
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  std::string m;
  Object o =
      // This is not complete. Build out as necessary.
      TypeSwitch<Type, Object>(type)
          .Case([&](ChannelType t) {
            m = "channel";
            return Object({{"inner", json(errorOp, t.getInner())}});
          })
          .Case([&](ChannelBundleType t) {
            m = "bundle";
            Array channels;
            for (auto field : t.getChannels())
              channels.push_back(Object(
                  {{"name", field.name.getValue()},
                   {"direction", stringifyChannelDirection(field.direction)},
                   {"type", json(errorOp, field.type)}}));
            return Object({{"channels", Value(std::move(channels))}});
          })
          .Case([&](AnyType t) {
            m = "any";
            return Object();
          })
          .Case([&](ListType t) {
            m = "list";
            return Object({{"element", json(errorOp, t.getElementType())}});
          })
          .Case([&](hw::ArrayType t) {
            m = "array";
            return Object({{"size", t.getNumElements()},
                           {"element", json(errorOp, t.getElementType())}});
          })
          .Case([&](hw::StructType t) {
            m = "struct";
            Array fields;
            for (auto field : t.getElements())
              fields.push_back(Object({{"name", field.name.getValue()},
                                       {"type", json(errorOp, field.type)}}));
            return Object({{"fields", Value(std::move(fields))}});
          })
          .Case([&](hw::TypeAliasType t) {
            m = "alias";
            return Object({{"name", t.getTypeDecl(symCache).getPreferredName()},
                           {"inner", json(errorOp, t.getInnerType())}});
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
  std::string circtName;
  llvm::raw_string_ostream(circtName) << type;
  o["circt_name"] = circtName;

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
llvm::json::Value ESIBuildManifestPass::json(Operation *errorOp,
                                             Attribute attr) {
  // This is far from complete. Build out as necessary.
  using llvm::json::Value;
  return TypeSwitch<Attribute, Value>(attr)
      .Case([&](StringAttr a) { return a.getValue(); })
      .Case([&](IntegerAttr a) { return a.getValue().getLimitedValue(); })
      .Case([&](TypeAttr a) {
        Type t = a.getValue();

        llvm::json::Object typeMD;
        if (typeLookup.contains(t)) {
          // If the type is in the type table, it'll be present in the types
          // section. Just give the circt type name, which is guaranteed to
          // uniquely identify the type.
          std::string buff;
          llvm::raw_string_ostream(buff) << a;
          typeMD["circt_name"] = buff;
          return typeMD;
        }

        typeMD["type"] = json(errorOp, t);
        return typeMD;
      })
      .Case([&](ArrayAttr a) {
        return llvm::json::Array(
            llvm::map_range(a, [&](Attribute a) { return json(errorOp, a); }));
      })
      .Case([&](DictionaryAttr a) {
        llvm::json::Object dict;
        for (const auto &entry : a.getValue())
          dict[entry.getName().getValue()] = json(errorOp, entry.getValue());
        return dict;
      })
      .Case([&](hw::InnerRefAttr ref) {
        llvm::json::Object dict;
        dict["outer_sym"] = ref.getRoot().getValue();
        dict["inner"] = ref.getTarget().getValue();
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
        std::string buff;
        llvm::raw_string_ostream(buff) << a;
        return buff;
      });
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIBuildManifestPass() {
  return std::make_unique<ESIBuildManifestPass>();
}
