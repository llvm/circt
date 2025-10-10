//===- LLHD.h - C interface for the LLHD dialect ------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_LLHD_H
#define CIRCT_C_DIALECT_LLHD_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(LLHD, llhd);

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// Check if a type is a time type.
MLIR_CAPI_EXPORTED bool llhdTypeIsATimeType(MlirType);

/// Create a time type.
MLIR_CAPI_EXPORTED MlirType llhdTimeTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

/// Check if an attribute is a time attribute.
MLIR_CAPI_EXPORTED bool llhdAttrIsATimeAttr(MlirAttribute);

/// Create a time attribute.
MLIR_CAPI_EXPORTED MlirAttribute llhdTimeAttrGet(MlirContext ctx,
                                                 MlirStringRef timeUnit,
                                                 uint64_t seconds,
                                                 uint64_t delta,
                                                 uint64_t epsilon);

/// Get the time unit of a time attribute.
MLIR_CAPI_EXPORTED MlirStringRef llhdTimeAttrGetTimeUnit(MlirAttribute);
/// Get the seconds component of a time attribute.
MLIR_CAPI_EXPORTED uint64_t llhdTimeAttrGetSeconds(MlirAttribute);
/// Get the delta component of a time attribute.
MLIR_CAPI_EXPORTED uint64_t llhdTimeAttrGetDelta(MlirAttribute);
/// Get the epsilon component of a time attribute.
MLIR_CAPI_EXPORTED uint64_t llhdTimeAttrGetEpsilon(MlirAttribute);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_LLHD_H
