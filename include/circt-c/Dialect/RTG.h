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

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTG_H
