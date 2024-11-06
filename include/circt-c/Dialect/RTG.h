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
                                               intptr_t numArgs,
                                               MlirType const *argTypes);

/// If the type is an RTG mode.
MLIR_CAPI_EXPORTED bool rtgTypeIsAMode(MlirType type);

/// Creates an RTG mode type in the context.
MLIR_CAPI_EXPORTED MlirType rtgModeTypeGet(MlirContext ctxt);

/// If the type is an RTG context resource.
MLIR_CAPI_EXPORTED bool rtgTypeIsAContextResource(MlirType type);

/// Creates an RTG context resource type in the context.
MLIR_CAPI_EXPORTED MlirType rtgContextResourceTypeGet(MlirContext ctxt);

/// If the type is an RTG set.
MLIR_CAPI_EXPORTED bool rtgTypeIsASet(MlirType type);

/// Creates an RTG set type in the context.
MLIR_CAPI_EXPORTED MlirType rtgSetTypeGet(MlirContext ctxt,
                                          MlirType elementType);

/// If the type is an RTG target.
MLIR_CAPI_EXPORTED bool rtgTypeIsATarget(MlirType type);

/// Creates an RTG target type in the context.
MLIR_CAPI_EXPORTED MlirType rtgTargetTypeGet(MlirContext ctxt,
                                             intptr_t numEntries,
                                             MlirAttribute const *entryNames,
                                             MlirType const *entryTypes);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTG_H
