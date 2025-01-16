//===- RTGTest.h - C interface for the for RTGTest dialect --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_RTGTEST_H
#define CIRCT_C_DIALECT_RTGTEST_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(RTGTest, rtgtest);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// If the type is an RTGTest CPUType.
MLIR_CAPI_EXPORTED bool rtgtestTypeIsACPU(MlirType type);

/// Creates an RTGTest CPU type in the context.
MLIR_CAPI_EXPORTED MlirType rtgtestCPUTypeGet(MlirContext ctxt);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

/// If the type is an RTGTest CPUAttr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsACPU(MlirAttribute attr);

/// Creates an RTGTest CPU attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestCPUAttrGet(MlirContext ctxt,
                                                   unsigned id);

/// Returns the core ID represented by the CPU attribute.
MLIR_CAPI_EXPORTED unsigned rtgtestCPUAttrGetId(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTGTEST_H
