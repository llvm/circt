//===-- circt-c/RtgTool.h - C API for the rtgtool -----------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_RTGTOOL_H
#define CIRCT_C_RTGTOOL_H

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Tool Options API.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(CirctRtgToolOptions, void);

#undef DEFINE_C_API_STRUCT

// NOLINTNEXTLINE(modernize-use-using)
typedef enum CiretRtgToolOutputFormat {
  CIRCT_RTGTOOL_OUTPUT_FORMAT_MLIR,
  CIRCT_RTGTOOL_OUTPUT_FORMAT_ELABORATED_MLIR,
  CIRCT_RTGTOOL_OUTPUT_FORMAT_ASM,
} CirctRtgToolOutputFormat;

MLIR_CAPI_EXPORTED CirctRtgToolOptions
circtRtgToolOptionsCreateDefault(unsigned seed);
MLIR_CAPI_EXPORTED void circtRtgToolOptionsDestroy(CirctRtgToolOptions options);

MLIR_CAPI_EXPORTED void
circtRtgToolOptionsSetOutputFormat(CirctRtgToolOptions options,
                                   CirctRtgToolOutputFormat format);

MLIR_CAPI_EXPORTED void circtRtgToolOptionsSetSeed(CirctRtgToolOptions options,
                                                   unsigned seed);

MLIR_CAPI_EXPORTED void
circtRtgToolOptionsSetVerifyPasses(CirctRtgToolOptions options, bool enable);

MLIR_CAPI_EXPORTED void
circtRtgToolOptionsSetVerbosePassExecution(CirctRtgToolOptions options,
                                           bool enable);

MLIR_CAPI_EXPORTED void circtRtgToolOptionsSetUnsupportedInstructions(
    CirctRtgToolOptions options, unsigned numInstr,
    const char **unsupportedInstructions);

MLIR_CAPI_EXPORTED void circtRtgToolOptionsAddUnsupportedInstruction(
    CirctRtgToolOptions options, const char *unsupportedInstruction);

MLIR_CAPI_EXPORTED void
circtRtgToolOptionsSetUnsupportedInstructionsFile(CirctRtgToolOptions options,
                                                  const char *filename);

//===----------------------------------------------------------------------===//
// Pipeline Population API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void
circtRtgToolRandomizerPipeline(MlirPassManager pm, CirctRtgToolOptions options);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_RTGTOOL_H
