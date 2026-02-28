//===- DC.h - C interface for the DC dialect ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_DC_H
#define CIRCT_C_DIALECT_DC_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(DC, dc);
MLIR_CAPI_EXPORTED void registerDCPasses(void);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_DC_H
