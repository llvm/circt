//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_SYNTH_H
#define CIRCT_C_DIALECT_SYNTH_H

#include "circt-c/Support/InstanceGraph.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void registerSynthesisPipeline(void);

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Synth, synth);
MLIR_CAPI_EXPORTED void registerSynthPasses(void);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

//===----------------------------------------------------------------------===//
// LongestPathAnalysis
//===----------------------------------------------------------------------===//

// Opaque handle to LongestPathObject
DEFINE_C_API_STRUCT(SynthLongestPathObject, void);

// Opaque handle to LongestPathHistory
DEFINE_C_API_STRUCT(SynthLongestPathHistory, void);

// Opaque handle to LongestPathDataflowPath
DEFINE_C_API_STRUCT(SynthLongestPathDataflowPath, void);

// Opaque handle to LongestPathAnalysis
DEFINE_C_API_STRUCT(SynthLongestPathAnalysis, void);

// Opaque handle to LongestPathCollection
DEFINE_C_API_STRUCT(SynthLongestPathCollection, void);

#undef DEFINE_C_API_STRUCT

// Create a LongestPathAnalysis for the given module
MLIR_CAPI_EXPORTED SynthLongestPathAnalysis synthLongestPathAnalysisCreate(
    MlirOperation module, bool collectDebugInfo, bool keepOnlyMaxDelayPaths,
    bool lazyComputation);

// Destroy a LongestPathAnalysis
MLIR_CAPI_EXPORTED void
synthLongestPathAnalysisDestroy(SynthLongestPathAnalysis analysis);

MLIR_CAPI_EXPORTED SynthLongestPathCollection synthLongestPathAnalysisGetPaths(
    SynthLongestPathAnalysis analysis, MlirValue value, int64_t bitPos,
    bool elaboratePaths);

MLIR_CAPI_EXPORTED SynthLongestPathCollection
synthLongestPathAnalysisGetAllPaths(SynthLongestPathAnalysis analysis,
                                    MlirStringRef moduleName,
                                    bool elaboratePaths);

//===----------------------------------------------------------------------===//
// LongestPathCollection
//===----------------------------------------------------------------------===//

// Check if the collection is valid
MLIR_CAPI_EXPORTED bool
synthLongestPathCollectionIsNull(SynthLongestPathCollection collection);

// Destroy a LongestPathCollection
MLIR_CAPI_EXPORTED void
synthLongestPathCollectionDestroy(SynthLongestPathCollection collection);

// Get the number of paths in the collection
MLIR_CAPI_EXPORTED size_t
synthLongestPathCollectionGetSize(SynthLongestPathCollection collection);

// Get a specific path from the collection as DataflowPath object
MLIR_CAPI_EXPORTED SynthLongestPathDataflowPath
synthLongestPathCollectionGetDataflowPath(SynthLongestPathCollection collection,
                                          size_t pathIndex);

MLIR_CAPI_EXPORTED void
synthLongestPathCollectionMerge(SynthLongestPathCollection dest,
                                SynthLongestPathCollection src);

//===----------------------------------------------------------------------===//
// DataflowPath API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED int64_t
synthLongestPathDataflowPathGetDelay(SynthLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED SynthLongestPathObject
synthLongestPathDataflowPathGetStartPoint(SynthLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED SynthLongestPathObject synthLongestPathDataflowPathGetEndPoint(
    SynthLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED SynthLongestPathHistory
synthLongestPathDataflowPathGetHistory(
    SynthLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED MlirOperation
synthLongestPathDataflowPathGetRoot(SynthLongestPathDataflowPath dataflowPath);

//===----------------------------------------------------------------------===//
// History API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool
synthLongestPathHistoryIsEmpty(SynthLongestPathHistory history);

MLIR_CAPI_EXPORTED void
synthLongestPathHistoryGetHead(SynthLongestPathHistory history,
                               SynthLongestPathObject *object, int64_t *delay,
                               MlirStringRef *comment);

MLIR_CAPI_EXPORTED SynthLongestPathHistory
synthLongestPathHistoryGetTail(SynthLongestPathHistory history);

//===----------------------------------------------------------------------===//
// Object API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED IgraphInstancePath
synthLongestPathObjectGetInstancePath(SynthLongestPathObject object);

MLIR_CAPI_EXPORTED MlirStringRef
synthLongestPathObjectName(SynthLongestPathObject object);

MLIR_CAPI_EXPORTED size_t
synthLongestPathObjectBitPos(SynthLongestPathObject object);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_SYNTH_H
