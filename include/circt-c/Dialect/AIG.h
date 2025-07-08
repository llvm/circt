//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_AIG_H
#define CIRCT_C_DIALECT_AIG_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIG, aig);
MLIR_CAPI_EXPORTED void registerAIGPasses(void);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

//===----------------------------------------------------------------------===//
// LongestPathAnalysis
//===----------------------------------------------------------------------===//

// Opaque handle to LongestPathAnalysis
DEFINE_C_API_STRUCT(AIGLongestPathAnalysis, void);

// Opaque handle to LongestPathCollection
DEFINE_C_API_STRUCT(AIGLongestPathCollection, void);

#undef DEFINE_C_API_STRUCT

// Create a LongestPathAnalysis for the given module
MLIR_CAPI_EXPORTED AIGLongestPathAnalysis
aigLongestPathAnalysisCreate(MlirOperation module, bool traceDebugPoints);

// Destroy a LongestPathAnalysis
MLIR_CAPI_EXPORTED void
aigLongestPathAnalysisDestroy(AIGLongestPathAnalysis analysis);

MLIR_CAPI_EXPORTED AIGLongestPathCollection aigLongestPathAnalysisGetAllPaths(
    AIGLongestPathAnalysis analysis, MlirStringRef moduleName,
    bool elaboratePaths);

//===----------------------------------------------------------------------===//
// LongestPathCollection
//===----------------------------------------------------------------------===//

// Check if the collection is valid
MLIR_CAPI_EXPORTED bool
aigLongestPathCollectionIsNull(AIGLongestPathCollection collection);

// Destroy a LongestPathCollection
MLIR_CAPI_EXPORTED void
aigLongestPathCollectionDestroy(AIGLongestPathCollection collection);

// Get the number of paths in the collection
MLIR_CAPI_EXPORTED size_t
aigLongestPathCollectionGetSize(AIGLongestPathCollection collection);

// Get a specific path from the collection as JSON
MLIR_CAPI_EXPORTED MlirStringRef aigLongestPathCollectionGetPath(
    AIGLongestPathCollection collection, int pathIndex);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_AIG_H
