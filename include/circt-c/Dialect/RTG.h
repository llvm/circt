//===- RTG.h - C interface for the for RTG dialect ----------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_RTG_H
#define CIRCT_C_DIALECT_RTG_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(RTG, rtg);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// If the type is an RTG sequence.
MLIR_CAPI_EXPORTED bool rtgTypeIsASequence(MlirType type);

/// Creates an RTG sequence type in the context.
MLIR_CAPI_EXPORTED MlirType rtgSequenceTypeGet(MlirContext ctxt,
                                               intptr_t numElements,
                                               MlirType const *elementTypes);

/// The number of substitution elements of the RTG sequence.
MLIR_CAPI_EXPORTED unsigned rtgSequenceTypeGetNumElements(MlirType type);

/// The type of of the substitution element at the given index.
MLIR_CAPI_EXPORTED MlirType rtgSequenceTypeGetElement(MlirType type,
                                                      unsigned i);

/// If the type is an RTG randomized sequence.
MLIR_CAPI_EXPORTED bool rtgTypeIsARandomizedSequence(MlirType type);

/// Creates an RTG randomized sequence type in the context.
MLIR_CAPI_EXPORTED MlirType rtgRandomizedSequenceTypeGet(MlirContext ctxt);

/// If the type is an RTG label.
MLIR_CAPI_EXPORTED bool rtgTypeIsALabel(MlirType type);

/// Creates an RTG mode type in the context.
MLIR_CAPI_EXPORTED MlirType rtgLabelTypeGet(MlirContext ctxt);

/// If the type is an RTG set.
MLIR_CAPI_EXPORTED bool rtgTypeIsASet(MlirType type);

/// Creates an RTG set type in the context.
MLIR_CAPI_EXPORTED MlirType rtgSetTypeGet(MlirType elementType);

/// Return the element type of the RTG set.
MLIR_CAPI_EXPORTED MlirType rtgSetTypeGetElementType(MlirType type);

/// If the type is an RTG bag.
MLIR_CAPI_EXPORTED bool rtgTypeIsABag(MlirType type);

/// Creates an RTG bag type in the context.
MLIR_CAPI_EXPORTED MlirType rtgBagTypeGet(MlirType elementType);

/// Return the element type of the RTG bag.
MLIR_CAPI_EXPORTED MlirType rtgBagTypeGetElementType(MlirType type);

/// If the type is an RTG dict.
MLIR_CAPI_EXPORTED bool rtgTypeIsADict(MlirType type);

/// Creates an RTG dict type in the context.
MLIR_CAPI_EXPORTED MlirType rtgDictTypeGet(MlirContext ctxt,
                                           intptr_t numEntries,
                                           MlirAttribute const *entryNames,
                                           MlirType const *entryTypes);

/// If the type is an RTG immediate.
MLIR_CAPI_EXPORTED bool rtgTypeIsAImmediate(MlirType type);

/// Creates an RTG immediate type in the context.
MLIR_CAPI_EXPORTED MlirType rtgImmediateTypeGet(MlirContext ctx,
                                                uint32_t width);

/// Returns the width of the RTG immediate type.
MLIR_CAPI_EXPORTED uint32_t rtgImmediateTypeGetWidth(MlirType type);

/// Creates an RTG array type in the context.
MLIR_CAPI_EXPORTED MlirType rtgArrayTypeGet(MlirType elementType);

/// If the type is an RTG array.
MLIR_CAPI_EXPORTED bool rtgTypeIsAArray(MlirType type);

/// Returns the element type of the RTG array.
MLIR_CAPI_EXPORTED MlirType rtgArrayTypeGetElementType(MlirType type);

/// If the type is an RTG memory.
MLIR_CAPI_EXPORTED bool rtgTypeIsAMemory(MlirType type);

/// Creates an RTG memory type in the context.
MLIR_CAPI_EXPORTED MlirType rtgMemoryTypeGet(MlirContext ctx,
                                             uint32_t addressWidth);

/// Returns the address with of an RTG memory type.
MLIR_CAPI_EXPORTED uint32_t rtgMemoryTypeGetAddressWidth(MlirType type);

/// If the type is an RTG memory block.
MLIR_CAPI_EXPORTED bool rtgTypeIsAMemoryBlock(MlirType type);

/// Creates an RTG memory block type in the context.
MLIR_CAPI_EXPORTED MlirType rtgMemoryBlockTypeGet(MlirContext ctx,
                                                  uint32_t addressWidth);

/// Returns the address with of an RTG memory block type.
MLIR_CAPI_EXPORTED uint32_t rtgMemoryBlockTypeGetAddressWidth(MlirType type);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

enum RTGLabelVisibility {
  RTG_LABEL_VISIBILITY_LOCAL,
  RTG_LABEL_VISIBILITY_GLOBAL,
  RTG_LABEL_VISIBILITY_EXTERNAL
};
typedef enum RTGLabelVisibility RTGLabelVisibility;

/// If the attribute is an RTG label visibility.
MLIR_CAPI_EXPORTED bool rtgAttrIsALabelVisibilityAttr(MlirAttribute attr);

/// Get the RTG label visibility from the attribute.
MLIR_CAPI_EXPORTED RTGLabelVisibility
rtgLabelVisibilityAttrGetValue(MlirAttribute attr);

/// Creates an RTG label visibility attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute
rtgLabelVisibilityAttrGet(MlirContext ctxt, RTGLabelVisibility visibility);

/// If the attribute is an RTG default context.
MLIR_CAPI_EXPORTED bool rtgAttrIsADefaultContextAttr(MlirAttribute attr);

/// Creates an RTG default context attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgDefaultContextAttrGet(MlirContext ctxt,
                                                          MlirType type);

/// Checks if the attribute is an RTG immediate attribute.
MLIR_CAPI_EXPORTED bool rtgAttrIsAImmediate(MlirAttribute attr);

/// Creates an RTG immediate attribute in the context with the given width and
/// value.
MLIR_CAPI_EXPORTED MlirAttribute rtgImmediateAttrGet(MlirContext ctx,
                                                     uint32_t width,
                                                     uint64_t value);

/// Returns the width of the RTG immediate attribute.
MLIR_CAPI_EXPORTED uint32_t rtgImmediateAttrGetWidth(MlirAttribute attr);

/// Returns the value of the RTG immediate attribute.
MLIR_CAPI_EXPORTED uint64_t rtgImmediateAttrGetValue(MlirAttribute attr);

/// If the attribute is an RTG any context attribute.
MLIR_CAPI_EXPORTED bool rtgAttrIsAAnyContextAttr(MlirAttribute attr);

/// Creates an RTG any context attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgAnyContextAttrGet(MlirContext ctxt,
                                                      MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTG_H
