//===- ESIBuildManifest.cpp - Build ESI system manifest ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/APIUtilities.h"
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
  void scrapeTypes(Operation *);

  /// Get a JSON representation of a type.
  llvm::json::Value json(Type);
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

  hw::HWSymbolCache symCache;
};
} // anonymous namespace

void ESIBuildManifestPass::runOnOperation() {
  MLIRContext *ctxt = &getContext();
  Operation *mod = getOperation();
  symCache.addDefinitions(mod);
  symCache.freeze();

  // Gather the relevant types.
  mod->walk([&](Operation *op) { scrapeTypes(op); });

  // JSONify the manifest.
  std::string jsonManifest = json();

  // Append a verbatim with the manifest to the end of the module.
  OpBuilder b = OpBuilder::atBlockEnd(&mod->getRegion(0).getBlocks().front());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, jsonManifest));
  auto outputFileAttr =
      hw::OutputFileAttr::getFromFilename(ctxt, "esi_system_manifest.json");
  verbatim->setAttr("output_file", outputFileAttr);

  // If zlib is available, compress the manifest and append it to the module.
  SmallVector<uint8_t, 10 * 1024> compressedManifest;
  if (llvm::compression::zlib::isAvailable()) {
    // Compress the manifest.
    llvm::compression::zlib::compress(
        ArrayRef((uint8_t *)jsonManifest.data(), jsonManifest.length()),
        compressedManifest, llvm::compression::zlib::BestSizeCompression);

    // Append a verbatim with the compressed manifest to the end of the module.
    auto compressedVerbatim = b.create<sv::VerbatimOp>(
        b.getUnknownLoc(),
        StringAttr::get(ctxt, StringRef((char *)compressedManifest.data(),
                                        compressedManifest.size())));
    auto compressedOutputFileAttr = hw::OutputFileAttr::getFromFilename(
        ctxt, "esi_system_manifest.json.zlib");
    compressedVerbatim->setAttr("output_file", compressedOutputFileAttr);
  } else {
    mod->emitWarning() << "zlib not available, skipping compressed manifest";
  }

  // If directed, write the manifest to a file. Mostly for debugging.
  if (!toFile.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(toFile, ec);
    if (ec) {
      mod->emitError() << "Failed to open file for writing: " << ec.message();
      signalPassFailure();
    } else {
      os << jsonManifest;
    }

    // If the compressed manifest is available, output it also.
    if (!compressedManifest.empty()) {
      llvm::raw_fd_ostream bos(toFile + ".zlib", ec);
      if (ec) {
        mod->emitError() << "Failed to open compressed file for writing: "
                         << ec.message();
        signalPassFailure();
      } else {
        bos.write((char *)compressedManifest.data(), compressedManifest.size());
      }
    }
  }
}

std::string ESIBuildManifestPass::json() {
  std::string jsonStrBuffer;
  llvm::raw_string_ostream os(jsonStrBuffer);
  llvm::json::OStream j(os, 2);

  j.objectBegin();
  j.attribute("api_version", esiApiVersion);
  j.attributeArray("types", [&]() {
    for (auto type : types) {
      j.value(json(type));
    }
  });
  j.objectEnd();

  return jsonStrBuffer;
}

void ESIBuildManifestPass::scrapeTypes(Operation *op) {
  TypeSwitch<Operation *>(op).Case([&](CosimEndpointOp cosim) {
    addType(cosim.getSend().getType());
    addType(cosim.getRecv().getType());
  });
}

/// Get a JSON representation of a type.
// NOLINTNEXTLINE(misc-no-recursion)
llvm::json::Value ESIBuildManifestPass::json(Type type) {
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  std::string m;
  Object o =
      // This is not complete. Build out as necessary.
      TypeSwitch<Type, Object>(type)
          .Case([&](ChannelType t) {
            m = "channel";
            return Object({{"inner", json(t.getInner())}});
          })
          .Case([&](ChannelBundleType t) {
            m = "bundle";
            Array fields;
            for (auto field : t.getChannels())
              fields.push_back(Object(
                  {{"name", field.name.getValue()},
                   {"direction", stringifyChannelDirection(field.direction)},
                   {"type", json(field.type)}}));
            return Object({{"fields", Value(std::move(fields))}});
          })
          .Case([&](AnyType t) {
            m = "any";
            return Object();
          })
          .Case([&](ListType t) {
            m = "list";
            return Object({{"element", json(t.getElementType())}});
          })
          .Case([&](hw::ArrayType t) {
            m = "array";
            return Object({{"size", t.getNumElements()},
                           {"element", json(t.getElementType())}});
          })
          .Case([&](hw::StructType t) {
            m = "struct";
            Array fields;
            for (auto field : t.getElements())
              fields.push_back(Object({{"name", field.name.getValue()},
                                       {"type", json(field.type)}}));
            return Object({{"fields", Value(std::move(fields))}});
          })
          .Case([&](hw::TypeAliasType t) {
            m = "alias";
            return Object({{"name", t.getTypeDecl(symCache).getPreferredName()},
                           {"inner", json(t.getInnerType())}});
          })
          .Case([&](IntegerType t) {
            m = "int";
            StringRef signedness =
                t.isSigned() ? "signed"
                             : (t.isUnsigned() ? "unsigned" : "signless");
            return Object({{"signedness", signedness}});
          })
          .Default([&](Type t) {
            getOperation()->emitWarning()
                << "ESI system manifest: unknown type: " << t;
            return Object();
          });

  // Common metadata.
  std::string circtName;
  llvm::raw_string_ostream(circtName) << type;
  o["circt_name"] = circtName;
  int64_t width = hw::getBitWidth(type);
  if (width >= 0)
    o["hw_bitwidth"] = width;
  o["dialect"] = type.getDialect().getNamespace();
  if (m.length())
    o["mnemonic"] = m;
  return o;
}
std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIBuildManifestPass() {
  return std::make_unique<ESIBuildManifestPass>();
}
