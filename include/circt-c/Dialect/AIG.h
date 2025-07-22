//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_AIG_H
#define CIRCT_C_DIALECT_AIG_H

#include "circt-c/Support/InstanceGraph.h"
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

// Opaque handle to LongestPathObject
DEFINE_C_API_STRUCT(AIGLongestPathObject, void);

// Opaque handle to LongestPathHistory
DEFINE_C_API_STRUCT(AIGLongestPathHistory, void);

// Opaque handle to LongestPathDataflowPath
DEFINE_C_API_STRUCT(AIGLongestPathDataflowPath, void);

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

// Get a specific path from the collection as DataflowPath object
MLIR_CAPI_EXPORTED AIGLongestPathDataflowPath
aigLongestPathCollectionGetDataflowPath(AIGLongestPathCollection collection,
                                        size_t pathIndex);

//===----------------------------------------------------------------------===//
// DataflowPath API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED int64_t
aigLongestPathDataflowPathGetDelay(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED AIGLongestPathObject
aigLongestPathDataflowPathGetFanIn(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED AIGLongestPathObject
aigLongestPathDataflowPathGetFanOut(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED AIGLongestPathHistory
aigLongestPathDataflowPathGetHistory(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED MlirOperation
aigLongestPathDataflowPathGetRoot(AIGLongestPathDataflowPath dataflowPath);

//===----------------------------------------------------------------------===//
// History API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool
aigLongestPathHistoryIsEmpty(AIGLongestPathHistory history);

MLIR_CAPI_EXPORTED void
aigLongestPathHistoryGetHead(AIGLongestPathHistory history,
                             AIGLongestPathObject *object, int64_t *delay,
                             MlirStringRef *comment);

MLIR_CAPI_EXPORTED AIGLongestPathHistory
aigLongestPathHistoryGetTail(AIGLongestPathHistory history);

//===----------------------------------------------------------------------===//
// Object API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED IgraphInstancePath
aigLongestPathObjectGetInstancePath(AIGLongestPathObject object);

MLIR_CAPI_EXPORTED MlirStringRef
aigLongestPathObjectName(AIGLongestPathObject object);

MLIR_CAPI_EXPORTED size_t
aigLongestPathObjectBitPos(AIGLongestPathObject object);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_AIG_H
