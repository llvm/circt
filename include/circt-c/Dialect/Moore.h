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

enum MooreRealKind {
  /// A `shortreal`.
  MooreShortReal,
  /// A `real`.
  MooreReal,
  /// A `realtime`.
  MooreRealTime,
};

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
MLIR_CAPI_EXPORTED MlirType mooreRealTypeGet(MlirContext ctx,
                                             enum MooreRealKind kind);
/// Create a packed unsized dimension type.
MLIR_CAPI_EXPORTED MlirType moorePackedUnsizedDimTypeGet(MlirType inner);
/// Create a packed range dimension type.
MLIR_CAPI_EXPORTED MlirType moorePackedRangeDimTypeGet(MlirType inner,
                                                       unsigned size,
                                                       bool upDir, int offset);
/// Create a unpacked unsized dimension type.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedUnsizedDimTypeGet(MlirType inner);
/// Create a unpacked array dimension type.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedArrayDimTypeGet(MlirType inner,
                                                         unsigned size);
/// Create a unpacked range dimension type.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedRangeDimTypeGet(MlirType inner,
                                                         unsigned size,
                                                         bool upDir,
                                                         int offset);
/// Create a unpacked assoc dimension type without index.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedAssocDimTypeGet(MlirType inner);
/// Create a unpacked assoc dimension type width index.
MLIR_CAPI_EXPORTED MlirType
mooreUnpackedAssocDimTypeGetWithIndex(MlirType inner, MlirType indexType);
/// Create a unpacked queue dimension type without bound.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedQueueDimTypeGet(MlirType inner);
/// Create a unpacked queue dimension type with bound.
MLIR_CAPI_EXPORTED MlirType
mooreUnpackedQueueDimTypeGetWithBound(MlirType inner, unsigned bound);
/// Checks whether the passed UnpackedType is a two-valued type.
MLIR_CAPI_EXPORTED bool mooreIsTwoValuedType(MlirType type);
/// Checks whether the passed UnpackedType is a four-valued type.
MLIR_CAPI_EXPORTED bool mooreIsFourValuedType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_MOORE_H
