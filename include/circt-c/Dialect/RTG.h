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

/// If the type is an RTG snippet.
MLIR_CAPI_EXPORTED bool rtgTypeIsASnippet(MlirType type);

/// Creates an RTG snippet type in the context.
MLIR_CAPI_EXPORTED MlirType rtgSnippetTypeGet(MlirContext ctxt);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTG_H
