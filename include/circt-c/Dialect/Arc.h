//===- Arc.h - C interface for the Arc dialect ------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_ARC_H
#define CIRCT_C_DIALECT_ARC_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Arc, arc);
MLIR_CAPI_EXPORTED void registerArcPasses(void);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool arcTypeIsAState(MlirType type);
MLIR_CAPI_EXPORTED MlirType arcStateTypeGet(MlirType innerType);
MLIR_CAPI_EXPORTED MlirType arcStateTypeGetType(MlirType type);

MLIR_CAPI_EXPORTED bool arcTypeIsAMemory(MlirType type);
MLIR_CAPI_EXPORTED MlirType arcMemoryTypeGet(unsigned numWords,
                                              MlirType wordType,
                                              MlirType addressType);

MLIR_CAPI_EXPORTED bool arcTypeIsAStorage(MlirType type);
MLIR_CAPI_EXPORTED MlirType arcStorageTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType arcStorageTypeGetWithSize(MlirContext ctx,
                                                       unsigned size);

MLIR_CAPI_EXPORTED bool arcTypeIsASimModelInstance(MlirType type);
MLIR_CAPI_EXPORTED MlirType arcSimModelInstanceTypeGet(MlirAttribute model);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_ARC_H
