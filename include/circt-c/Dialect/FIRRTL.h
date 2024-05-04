//===- FIRRTL.h - C interface for the FIRRTL dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_FIRRTL_H
#define CIRCT_C_DIALECT_FIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLConvention {
  FIRRTL_CONVENTION_INTERNAL,
  FIRRTL_CONVENTION_SCALARIZED,
} FIRRTLConvention;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLDirection {
  FIRRTL_DIRECTION_IN,
  FIRRTL_DIRECTION_OUT,
} FIRRTLDirection;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLNameKind {
  FIRRTL_NAME_KIND_DROPPABLE_NAME,
  FIRRTL_NAME_KIND_INTERESTING_NAME,
} FIRRTLNameKind;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLRUW {
  FIRRTL_RUW_UNDEFINED,
  FIRRTL_RUW_OLD,
  FIRRTL_RUW_NEW,
} FIRRTLRUW;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLMemDir {
  FIRRTL_MEM_DIR_INFER,
  FIRRTL_MEM_DIR_READ,
  FIRRTL_MEM_DIR_WRITE,
  FIRRTL_MEM_DIR_READ_WRITE,
} FIRRTLMemDir;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLEventControl {
  FIRRTL_EVENT_CONTROL_AT_POS_EDGE,
  FIRRTL_EVENT_CONTROL_AT_NEG_EDGE,
  FIRRTL_EVENT_CONTROL_AT_EDGE,
} FIRRTLEventControl;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLValueFlow {
  FIRRTL_VALUE_FLOW_NONE,
  FIRRTL_VALUE_FLOW_SOURCE,
  FIRRTL_VALUE_FLOW_SINK,
  FIRRTL_VALUE_FLOW_DUPLEX,
} FIRRTLValueFlow;

// NOLINTNEXTLINE(modernize-use-using)
typedef struct FIRRTLBundleField {
  MlirIdentifier name;
  bool isFlip;
  MlirType type;
} FIRRTLBundleField;

// NOLINTNEXTLINE(modernize-use-using)
typedef struct FIRRTLClassElement {
  MlirIdentifier name;
  MlirType type;
  FIRRTLDirection direction;
} FIRRTLClassElement;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirType firrtlTypeGetUInt(MlirContext ctx, int32_t width);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetSInt(MlirContext ctx, int32_t width);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetClock(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetReset(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAsyncReset(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAnalog(MlirContext ctx, int32_t width);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetVector(MlirContext ctx,
                                                MlirType element, size_t count);
MLIR_CAPI_EXPORTED bool firrtlTypeIsAOpenBundle(MlirType type);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetBundle(
    MlirContext ctx, size_t count, const FIRRTLBundleField *fields);
MLIR_CAPI_EXPORTED unsigned
firrtlTypeGetBundleFieldIndex(MlirType type, MlirStringRef fieldName);

MLIR_CAPI_EXPORTED MlirType firrtlTypeGetRef(MlirType target, bool forceable);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAnyRef(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetInteger(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetDouble(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetString(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetBoolean(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetPath(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetList(MlirContext ctx,
                                              MlirType elementType);
MLIR_CAPI_EXPORTED MlirType
firrtlTypeGetClass(MlirContext ctx, MlirAttribute name, size_t numberOfElements,
                   const FIRRTLClassElement *elements);

MLIR_CAPI_EXPORTED MlirType firrtlTypeGetMaskType(MlirType type);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetConvention(MlirContext ctx, FIRRTLConvention convention);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetPortDirs(
    MlirContext ctx, size_t count, const FIRRTLDirection *dirs);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetParamDecl(MlirContext ctx,
                                                        MlirIdentifier name,
                                                        MlirType type,
                                                        MlirAttribute value);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetNameKind(MlirContext ctx,
                                                       FIRRTLNameKind nameKind);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetRUW(MlirContext ctx,
                                                  FIRRTLRUW ruw);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetMemInit(MlirContext ctx,
                                                      MlirIdentifier filename,
                                                      bool isBinary,
                                                      bool isInline);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetMemDir(MlirContext ctx,
                                                     FIRRTLMemDir dir);

MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetEventControl(MlirContext ctx, FIRRTLEventControl eventControl);

// Workaround:
// https://github.com/llvm/llvm-project/issues/84190#issuecomment-2035552035
MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetIntegerFromString(
    MlirType type, unsigned numBits, MlirStringRef str, uint8_t radix);

//===----------------------------------------------------------------------===//
// Utility API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED FIRRTLValueFlow firrtlValueFoldFlow(MlirValue value,
                                                       FIRRTLValueFlow flow);

MLIR_CAPI_EXPORTED bool
firrtlImportAnnotationsFromJSONRaw(MlirContext ctx,
                                   MlirStringRef annotationsStr,
                                   MlirAttribute *importedAnnotationsArray);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
