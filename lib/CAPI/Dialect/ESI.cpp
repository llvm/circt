//===- ESI.cpp - C interface for the ESI dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/ESI.h"
#include "circt/Dialect/ESI/AppID.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

using namespace circt;
using namespace circt::esi;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ESI, esi, circt::esi::ESIDialect)

void registerESIPasses() { circt::esi::registerESIPasses(); }

bool circtESITypeIsAChannelType(MlirType type) {
  return isa<ChannelType>(unwrap(type));
}

MlirType circtESIChannelTypeGet(MlirType inner, uint32_t signaling,
                                uint64_t dataDelay) {
  auto signalEnum = symbolizeChannelSignaling(signaling);
  if (!signalEnum)
    return {};
  auto cppInner = unwrap(inner);
  return wrap(ChannelType::get(cppInner.getContext(), cppInner, *signalEnum,
                               dataDelay));
}

MlirType circtESIChannelGetInner(MlirType channelType) {
  return wrap(cast<ChannelType>(unwrap(channelType)).getInner());
}
uint32_t circtESIChannelGetSignaling(MlirType channelType) {
  return (uint32_t)cast<ChannelType>(unwrap(channelType)).getSignaling();
}
uint64_t circtESIChannelGetDataDelay(MlirType channelType) {
  return cast<ChannelType>(unwrap(channelType)).getDataDelay();
}

bool circtESITypeIsAnAnyType(MlirType type) {
  return isa<AnyType>(unwrap(type));
}
MlirType circtESIAnyTypeGet(MlirContext ctxt) {
  return wrap(AnyType::get(unwrap(ctxt)));
}

bool circtESITypeIsAListType(MlirType type) {
  return isa<ListType>(unwrap(type));
}

MlirType circtESIListTypeGet(MlirType inner) {
  auto cppInner = unwrap(inner);
  return wrap(ListType::get(cppInner.getContext(), cppInner));
}

MlirType circtESIListTypeGetElementType(MlirType list) {
  return wrap(cast<ListType>(unwrap(list)).getElementType());
}

bool circtESICheckInnerTypeMatch(MlirType to, MlirType from) {
  return succeeded(checkInnerTypeMatch(unwrap(to), unwrap(from)));
}

void circtESIAppendMlirFile(MlirModule cMod, MlirStringRef filename) {
  ModuleOp modOp = unwrap(cMod);
  auto loadedMod =
      mlir::parseSourceFile<ModuleOp>(unwrap(filename), modOp.getContext());
  Block *loadedBlock = loadedMod->getBody();
  assert(!modOp->getRegions().empty());
  if (modOp.getBodyRegion().empty()) {
    modOp.getBodyRegion().push_back(loadedBlock);
    return;
  }
  auto &ops = modOp.getBody()->getOperations();
  ops.splice(ops.end(), loadedBlock->getOperations());
}
MlirOperation circtESILookup(MlirModule mod, MlirStringRef symbol) {
  return wrap(SymbolTable::lookupSymbolIn(unwrap(mod), unwrap(symbol)));
}

void circtESIRegisterGlobalServiceGenerator(
    MlirStringRef impl_type, CirctESIServiceGeneratorFunc genFunc,
    void *userData) {
  ServiceGeneratorDispatcher::globalDispatcher().registerGenerator(
      unwrap(impl_type), [genFunc, userData](ServiceImplementReqOp req,
                                             ServiceDeclOpInterface decl,
                                             ServiceImplRecordOp record) {
        return unwrap(genFunc(wrap(req), wrap(decl.getOperation()),
                              wrap(record.getOperation()), userData));
      });
}
//===----------------------------------------------------------------------===//
// Channel bundles
//===----------------------------------------------------------------------===//

bool circtESITypeIsABundleType(MlirType type) {
  return isa<ChannelBundleType>(unwrap(type));
}
MlirType circtESIBundleTypeGet(MlirContext cctxt, size_t numChannels,
                               const CirctESIBundleTypeBundleChannel *channels,
                               bool resettable) {
  MLIRContext *ctxt = unwrap(cctxt);
  SmallVector<BundledChannel, 4> channelsVector(llvm::map_range(
      ArrayRef<CirctESIBundleTypeBundleChannel>(channels, numChannels),
      [](auto channel) {
        return BundledChannel{cast<StringAttr>(unwrap(channel.name)),
                              (ChannelDirection)channel.direction,
                              cast<ChannelType>(unwrap(channel.channelType))};
      }));
  return wrap(ChannelBundleType::get(
      ctxt, channelsVector, resettable ? UnitAttr::get(ctxt) : UnitAttr()));
}
bool circtESIBundleTypeGetResettable(MlirType bundle) {
  return cast<ChannelBundleType>(unwrap(bundle)).getResettable() != UnitAttr();
}
size_t circtESIBundleTypeGetNumChannels(MlirType bundle) {
  return cast<ChannelBundleType>(unwrap(bundle)).getChannels().size();
}
CirctESIBundleTypeBundleChannel circtESIBundleTypeGetChannel(MlirType bundle,
                                                             size_t idx) {
  BundledChannel channel =
      cast<ChannelBundleType>(unwrap(bundle)).getChannels()[idx];
  return CirctESIBundleTypeBundleChannel{
      wrap(channel.name), (unsigned)channel.direction, wrap(channel.type)};
}

//===----------------------------------------------------------------------===//
// AppID
//===----------------------------------------------------------------------===//

bool circtESIAttributeIsAnAppIDAttr(MlirAttribute attr) {
  return isa<AppIDAttr>(unwrap(attr));
}

MlirAttribute circtESIAppIDAttrGet(MlirContext ctxt, MlirStringRef name,
                                   uint64_t index) {
  return wrap(AppIDAttr::get(
      unwrap(ctxt), StringAttr::get(unwrap(ctxt), unwrap(name)), index));
}
MlirAttribute circtESIAppIDAttrGetNoIdx(MlirContext ctxt, MlirStringRef name) {
  return wrap(AppIDAttr::get(
      unwrap(ctxt), StringAttr::get(unwrap(ctxt), unwrap(name)), std::nullopt));
}
MlirStringRef circtESIAppIDAttrGetName(MlirAttribute attr) {
  return wrap(cast<AppIDAttr>(unwrap(attr)).getName().getValue());
}
bool circtESIAppIDAttrGetIndex(MlirAttribute attr, uint64_t *indexOut) {
  std::optional<uint64_t> index = cast<AppIDAttr>(unwrap(attr)).getIndex();
  if (!index)
    return false;
  *indexOut = index.value();
  return true;
}

bool circtESIAttributeIsAnAppIDPathAttr(MlirAttribute attr) {
  return isa<AppIDPathAttr>(unwrap(attr));
}

MlirAttribute circtESIAppIDAttrPathGet(MlirContext ctxt, MlirAttribute root,
                                       intptr_t numElements,
                                       MlirAttribute const *cElements) {
  SmallVector<AppIDAttr, 8> elements;
  for (intptr_t i = 0; i < numElements; ++i)
    elements.push_back(cast<AppIDAttr>(unwrap(cElements[i])));
  return wrap(AppIDPathAttr::get(
      unwrap(ctxt), cast<FlatSymbolRefAttr>(unwrap(root)), elements));
}
MlirAttribute circtESIAppIDAttrPathGetRoot(MlirAttribute attr) {
  return wrap(cast<AppIDPathAttr>(unwrap(attr)).getRoot());
}
uint64_t circtESIAppIDAttrPathGetNumComponents(MlirAttribute attr) {
  return cast<AppIDPathAttr>(unwrap(attr)).getPath().size();
}
MlirAttribute circtESIAppIDAttrPathGetComponent(MlirAttribute attr,
                                                uint64_t index) {
  return wrap(cast<AppIDPathAttr>(unwrap(attr)).getPath()[index]);
}

DEFINE_C_API_PTR_METHODS(CirctESIAppIDIndex, circt::esi::AppIDIndex)

/// Create an index of appids through which to do appid lookups efficiently.
MLIR_CAPI_EXPORTED CirctESIAppIDIndex
circtESIAppIDIndexGet(MlirOperation root) {
  auto *idx = new AppIDIndex(unwrap(root));
  if (idx->isValid())
    return wrap(idx);
  return CirctESIAppIDIndex{nullptr};
}

/// Free an AppIDIndex.
MLIR_CAPI_EXPORTED void circtESIAppIDIndexFree(CirctESIAppIDIndex index) {
  delete unwrap(index);
}

MLIR_CAPI_EXPORTED MlirAttribute
circtESIAppIDIndexGetChildAppIDsOf(CirctESIAppIDIndex idx, MlirOperation op) {
  auto mod = cast<hw::HWModuleLike>(unwrap(op));
  return wrap(unwrap(idx)->getChildAppIDsOf(mod));
}

MLIR_CAPI_EXPORTED
MlirAttribute circtESIAppIDIndexGetAppIDPath(CirctESIAppIDIndex idx,
                                             MlirOperation fromMod,
                                             MlirAttribute appid,
                                             MlirLocation loc) {
  auto mod = cast<hw::HWModuleLike>(unwrap(fromMod));
  auto path = cast<AppIDAttr>(unwrap(appid));
  FailureOr<ArrayAttr> instPath =
      unwrap(idx)->getAppIDPathAttr(mod, path, unwrap(loc));
  if (failed(instPath))
    return MlirAttribute{nullptr};
  return wrap(*instPath);
}
