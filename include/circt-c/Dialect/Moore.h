//===- Moore.h - C interface for the Moore dialect ----------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_MOORE_H
#define CIRCT_C_DIALECT_MOORE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Moore, moore);

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// Create a void type.
MLIR_CAPI_EXPORTED MlirType mooreVoidTypeGet(MlirContext ctx);
/// Create a string type.
MLIR_CAPI_EXPORTED MlirType mooreStringTypeGet(MlirContext ctx);
/// Create a chandle type.
MLIR_CAPI_EXPORTED MlirType mooreChandleTypeGet(MlirContext ctx);
/// Create an event type.
MLIR_CAPI_EXPORTED MlirType mooreEventTypeGet(MlirContext ctx);
/// Create a two-valued simple bit vector type.
MLIR_CAPI_EXPORTED MlirType mooreIntTypeGetInt(MlirContext ctx, unsigned width);
/// Create a four-valued simple bit vector type.
MLIR_CAPI_EXPORTED MlirType mooreIntTypeGetLogic(MlirContext ctx,
                                                 unsigned width);
/// Create a real type.
MLIR_CAPI_EXPORTED MlirType mooreRealTypeGet(MlirContext ctx, unsigned width);
/// Create a packed open array type.
MLIR_CAPI_EXPORTED MlirType mooreOpenArrayTypeGet(MlirType elementType);
/// Create a packed array type.
MLIR_CAPI_EXPORTED MlirType mooreArrayTypeGet(unsigned size,
                                              MlirType elementType);
/// Create an unpacked open array type.
MLIR_CAPI_EXPORTED MlirType mooreOpenUnpackedArrayTypeGet(MlirType elementType);
/// Create an unpacked array type.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedArrayTypeGet(unsigned size,
                                                      MlirType elementType);
/// Create an unpacked associative array type.
MLIR_CAPI_EXPORTED MlirType mooreAssocArrayTypeGet(MlirType elementType,
                                                   MlirType indexType);
/// Create an unpacked queue type.
MLIR_CAPI_EXPORTED MlirType mooreQueueTypeGet(MlirType elementType,
                                              unsigned bound);
/// Checks whether the passed UnpackedType is a two-valued type.
MLIR_CAPI_EXPORTED bool mooreIsTwoValuedType(MlirType type);
/// Checks whether the passed UnpackedType is a four-valued type.
MLIR_CAPI_EXPORTED bool mooreIsFourValuedType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_MOORE_H
