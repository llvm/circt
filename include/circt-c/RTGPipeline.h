//===-- circt-c/RTGPipeline.h - C API for random test generation --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_RTGPIPELINE_H
#define CIRCT_C_RTGPIPELINE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CirctRTGOutputMLIR = 0,
  CirctRTGOutputRenderedMLIR = 1,
  CirctRTGOutputASM = 2,
  CirctRTGOutputELF = 3
} CirctRTGOutputFormat;

/// Runs the passes necessary for random test generation.
MLIR_CAPI_EXPORTED MlirLogicalResult circtGenerateRandomTests(
    MlirModule, bool, bool, bool, unsigned, unsigned, char **, const char *,
    CirctRTGOutputFormat, MlirStringCallback, void *userData);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_RTGPIPELINE_H
