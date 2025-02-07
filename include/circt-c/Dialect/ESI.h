//===- ESI.h - C interface for the ESI dialect --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_ESI_H
#define CIRCT_C_DIALECT_ESI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(ESI, esi);
MLIR_CAPI_EXPORTED void registerESIPasses(void);

MLIR_CAPI_EXPORTED bool circtESITypeIsAChannelType(MlirType type);
MLIR_CAPI_EXPORTED MlirType circtESIChannelTypeGet(MlirType inner,
                                                   uint32_t signaling,
                                                   uint64_t dataDelay);
MLIR_CAPI_EXPORTED MlirType circtESIChannelGetInner(MlirType channelType);
MLIR_CAPI_EXPORTED uint32_t circtESIChannelGetSignaling(MlirType channelType);
MLIR_CAPI_EXPORTED uint64_t circtESIChannelGetDataDelay(MlirType channelType);

MLIR_CAPI_EXPORTED bool circtESITypeIsAnAnyType(MlirType type);
MLIR_CAPI_EXPORTED MlirType circtESIAnyTypeGet(MlirContext);

MLIR_CAPI_EXPORTED bool circtESITypeIsAListType(MlirType type);
MLIR_CAPI_EXPORTED MlirType circtESIListTypeGet(MlirType inner);
MLIR_CAPI_EXPORTED MlirType
circtESIListTypeGetElementType(MlirType channelType);

MLIR_CAPI_EXPORTED void circtESIAppendMlirFile(MlirModule,
                                               MlirStringRef fileName);
MLIR_CAPI_EXPORTED MlirOperation circtESILookup(MlirModule,
                                                MlirStringRef symbol);

MLIR_CAPI_EXPORTED bool circtESICheckInnerTypeMatch(MlirType to, MlirType from);

//===----------------------------------------------------------------------===//
// Channel bundles
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(modernize-use-using)
typedef struct {
  MlirIdentifier name;
  unsigned direction;
  MlirType channelType; // MUST be ChannelType.
} CirctESIBundleTypeBundleChannel;

MLIR_CAPI_EXPORTED bool circtESITypeIsABundleType(MlirType type);
MLIR_CAPI_EXPORTED MlirType circtESIBundleTypeGet(
    MlirContext, size_t numChannels,
    const CirctESIBundleTypeBundleChannel *channels, bool resettable);
MLIR_CAPI_EXPORTED bool circtESIBundleTypeGetResettable(MlirType bundle);
MLIR_CAPI_EXPORTED size_t circtESIBundleTypeGetNumChannels(MlirType bundle);
MLIR_CAPI_EXPORTED CirctESIBundleTypeBundleChannel
circtESIBundleTypeGetChannel(MlirType bundle, size_t idx);

//===----------------------------------------------------------------------===//
// Services
//===----------------------------------------------------------------------===//

typedef MlirLogicalResult (*CirctESIServiceGeneratorFunc)(
    MlirOperation serviceImplementReqOp, MlirOperation declOp,
    MlirOperation recordOp, void *userData);
MLIR_CAPI_EXPORTED void circtESIRegisterGlobalServiceGenerator(
    MlirStringRef impl_type, CirctESIServiceGeneratorFunc, void *userData);

//===----------------------------------------------------------------------===//
// AppID
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool circtESIAttributeIsAnAppIDAttr(MlirAttribute);
MLIR_CAPI_EXPORTED
MlirAttribute circtESIAppIDAttrGet(MlirContext, MlirStringRef name,
                                   uint64_t index);
MLIR_CAPI_EXPORTED
MlirAttribute circtESIAppIDAttrGetNoIdx(MlirContext ctxt, MlirStringRef name);
MLIR_CAPI_EXPORTED MlirStringRef circtESIAppIDAttrGetName(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool circtESIAppIDAttrGetIndex(MlirAttribute attr,
                                                  uint64_t *index);

MLIR_CAPI_EXPORTED bool circtESIAttributeIsAnAppIDPathAttr(MlirAttribute);
MLIR_CAPI_EXPORTED
MlirAttribute circtESIAppIDAttrPathGet(MlirContext, MlirAttribute root,
                                       intptr_t numElements,
                                       MlirAttribute const *elements);
MLIR_CAPI_EXPORTED MlirAttribute
circtESIAppIDAttrPathGetRoot(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t
circtESIAppIDAttrPathGetNumComponents(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
circtESIAppIDAttrPathGetComponent(MlirAttribute attr, uint64_t index);

// NOLINTNEXTLINE(modernize-use-using)
typedef struct {
  void *ptr;
} CirctESIAppIDIndex;

/// Create an index of appids through which to do appid lookups efficiently.
MLIR_CAPI_EXPORTED CirctESIAppIDIndex circtESIAppIDIndexGet(MlirOperation root);

/// Free an AppIDIndex.
MLIR_CAPI_EXPORTED void circtESIAppIDIndexFree(CirctESIAppIDIndex);

MLIR_CAPI_EXPORTED MlirAttribute
    circtESIAppIDIndexGetChildAppIDsOf(CirctESIAppIDIndex, MlirOperation);

MLIR_CAPI_EXPORTED
MlirAttribute circtESIAppIDIndexGetAppIDPath(CirctESIAppIDIndex,
                                             MlirOperation fromMod,
                                             MlirAttribute appid,
                                             MlirLocation loc);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_ESI_H
