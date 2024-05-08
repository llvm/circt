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

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_SEQ_H
