//===- CHIRRTL.h - C interface for the CHIRRTL dialect ------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_CHIRRTL_H
#define CIRCT_C_DIALECT_CHIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CHIRRTL, chirrtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirType chirrtlTypeGetCMemory(MlirContext ctx,
                                                  MlirType elementType,
                                                  uint64_t numElements);

MLIR_CAPI_EXPORTED MlirType chirrtlTypeGetCMemoryPort(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_CHIRRTL_H
