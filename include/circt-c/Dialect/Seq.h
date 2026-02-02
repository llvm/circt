//===- Seq.h - C interface for the Seq dialect --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_SEQ_H
#define CIRCT_C_DIALECT_SEQ_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Sequential, seq);
MLIR_CAPI_EXPORTED void registerSeqPasses(void);

/// If the type is an clock type
MLIR_CAPI_EXPORTED bool seqTypeIsAClock(MlirType type);

/// Creates an seq clock type
MLIR_CAPI_EXPORTED MlirType seqClockTypeGet(MlirContext ctx);

/// If the type is an immutable type
MLIR_CAPI_EXPORTED bool seqTypeIsAImmutable(MlirType type);

/// Creates a seq immutable type
MLIR_CAPI_EXPORTED MlirType seqImmutableTypeGet(MlirType type);

/// Creates a seq immutable type
MLIR_CAPI_EXPORTED MlirType seqImmutableTypeGetInnerType(MlirType type);

/// If the type is an HLMemType
MLIR_CAPI_EXPORTED bool seqTypeIsAHLMem(MlirType type);

/// Creates an seq HLMem type
MLIR_CAPI_EXPORTED MlirType seqHLMemTypeGet(MlirContext ctx, intptr_t rank,
                                            const int64_t *shape,
                                            MlirType elementType);

/// Gets the element type of an HLMem type
MLIR_CAPI_EXPORTED MlirType seqHLMemTypeGetElementType(MlirType type);

/// Gets the rank of an HLMem type
MLIR_CAPI_EXPORTED intptr_t seqHLMemTypeGetRank(MlirType type);

/// Gets a pointer to the shape array of an HLMem type
/// The array length is equal to the rank.
/// The pointer remains valid as long as the MlirType exists.
MLIR_CAPI_EXPORTED const int64_t *seqHLMemTypeGetShape(MlirType type);

/// If the type is a FirMem type
MLIR_CAPI_EXPORTED bool seqTypeIsAFirMem(MlirType type);

/// Creates a seq FirMem type
/// \param maskWidth A pointer to the mask width. Pass NULL if no mask is
/// present.
MLIR_CAPI_EXPORTED MlirType seqFirMemTypeGet(MlirContext ctx, uint64_t depth,
                                             uint32_t width,
                                             const uint32_t *maskWidth);

/// Gets the depth of a FirMem type
MLIR_CAPI_EXPORTED uint64_t seqFirMemTypeGetDepth(MlirType type);

/// Gets the width of a FirMem type
MLIR_CAPI_EXPORTED uint32_t seqFirMemTypeGetWidth(MlirType type);

/// Returns true if the FirMem type has a mask width defined
MLIR_CAPI_EXPORTED bool seqFirMemTypeHasMask(MlirType type);

/// Gets the mask width of a FirMem type
/// Behavior is undefined if seqFirMemTypeHasMask returns false
MLIR_CAPI_EXPORTED uint32_t seqFirMemTypeGetMaskWidth(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_SEQ_H
