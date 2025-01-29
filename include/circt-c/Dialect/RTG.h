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
MLIR_CAPI_EXPORTED MlirType rtgSequenceTypeGet(MlirContext ctxt);

/// If the type is an RTG label.
MLIR_CAPI_EXPORTED bool rtgTypeIsALabel(MlirType type);

/// Creates an RTG mode type in the context.
MLIR_CAPI_EXPORTED MlirType rtgLabelTypeGet(MlirContext ctxt);

/// If the type is an RTG set.
MLIR_CAPI_EXPORTED bool rtgTypeIsASet(MlirType type);

/// Creates an RTG set type in the context.
MLIR_CAPI_EXPORTED MlirType rtgSetTypeGet(MlirType elementType);

/// If the type is an RTG bag.
MLIR_CAPI_EXPORTED bool rtgTypeIsABag(MlirType type);

/// Creates an RTG bag type in the context.
MLIR_CAPI_EXPORTED MlirType rtgBagTypeGet(MlirType elementType);

/// If the type is an RTG dict.
MLIR_CAPI_EXPORTED bool rtgTypeIsADict(MlirType type);

/// Creates an RTG dict type in the context.
MLIR_CAPI_EXPORTED MlirType rtgDictTypeGet(MlirContext ctxt,
                                           intptr_t numEntries,
                                           MlirAttribute const *entryNames,
                                           MlirType const *entryTypes);

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

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTG_H
