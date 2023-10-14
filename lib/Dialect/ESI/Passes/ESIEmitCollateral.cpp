//===- ESIEmitCollateral.cpp - Emit ESI collateral pass ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit ESI collateral pass. Collateral includes the capnp schema and a JSON
// descriptor of the service hierarchy.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/APIUtilities.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::hw;

static llvm::json::Value toJSON(Type type) {
  // TODO: This is far from complete. Build out as necessary.
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  StringRef dialect = type.getDialect().getNamespace();
  std::string m;
  Object o = TypeSwitch<Type, Object>(type)
                 .Case([&](ChannelType t) {
                   m = "channel";
                   return Object({{"inner", toJSON(t.getInner())}});
                 })
                 .Case([&](AnyType t) {
                   m = "any";
                   return Object();
                 })
                 .Case([&](StructType t) {
                   m = "struct";
                   Array fields;
                   for (auto field : t.getElements())
                     fields.push_back(Object({{"name", field.name.getValue()},
                                              {"type", toJSON(field.type)}}));
                   return Object({{"fields", Value(std::move(fields))}});
                 })
                 .Default([&](Type t) {
                   llvm::raw_string_ostream(m) << t;
                   return Object();
                 });
  o["dialect"] = dialect;
  if (m.length())
    o["mnemonic"] = m;
  return o;
}

// Serialize an attribute to a JSON value.
static llvm::json::Value toJSON(Attribute attr) {
  // TODO: This is far from complete. Build out as necessary.
  using llvm::json::Value;
  return TypeSwitch<Attribute, Value>(attr)
      .Case([&](StringAttr a) { return a.getValue(); })
      .Case([&](IntegerAttr a) { return a.getValue().getLimitedValue(); })
      .Case([&](TypeAttr a) {
        Type t = a.getValue();
        llvm::json::Object typeMD;
        typeMD["type_desc"] = toJSON(t);

        std::string buf;
        llvm::raw_string_ostream(buf) << t;
        typeMD["mlir_name"] = buf;

        if (auto chanType = t.dyn_cast<ChannelType>()) {
          Type inner = chanType.getInner();
          typeMD["hw_bitwidth"] = hw::getBitWidth(inner);
          ESIAPIType cosimSchema(inner);
          typeMD["capnp_type_id"] = cosimSchema.typeID();
          typeMD["capnp_name"] = cosimSchema.name().str();
        } else {
          typeMD["hw_bitwidth"] = hw::getBitWidth(t);
        }
        return typeMD;
      })
      .Case([&](ArrayAttr a) {
        return llvm::json::Array(
            llvm::map_range(a, [](Attribute a) { return toJSON(a); }));
      })
      .Case([&](DictionaryAttr a) {
        llvm::json::Object dict;
        for (const auto &entry : a.getValue())
          dict[entry.getName().getValue()] = toJSON(entry.getValue());
        return dict;
      })
      .Case([&](InnerRefAttr ref) {
        llvm::json::Object dict;
        dict["outer_sym"] = ref.getModule().getValue();
        dict["inner"] = ref.getName().getValue();
        return dict;
      })
      .Default([&](Attribute a) {
        std::string buff;
        llvm::raw_string_ostream(buff) << a;
        return buff;
      });
}

namespace {
/// Run all the physical lowerings.
struct ESIEmitCollateralPass
    : public ESIEmitCollateralBase<ESIEmitCollateralPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESIEmitCollateralPass::runOnOperation() {
  ModuleOp mod = getOperation();
  auto *ctxt = &getContext();

  // Check for cosim endpoints in the design. If the design doesn't have any
  // we don't need a schema.
  WalkResult cosimWalk =
      mod.walk([](CosimEndpointOp _) { return WalkResult::interrupt(); });
  if (!cosimWalk.wasInterrupted())
    return;

  // Generate the schema
  std::string schemaStrBuffer;
  llvm::raw_string_ostream os(schemaStrBuffer);
  if (failed(exportCosimSchema(mod, os))) {
    signalPassFailure();
    return;
  }

  // And stuff if in a verbatim op with a filename, optionally.
  OpBuilder b = OpBuilder::atBlockEnd(mod.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, os.str()));
  if (!schemaFile.empty()) {
    auto outputFileAttr = OutputFileAttr::getFromFilename(ctxt, schemaFile);
    verbatim->setAttr("output_file", outputFileAttr);
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIEmitCollateralPass() {
  return std::make_unique<ESIEmitCollateralPass>();
}
