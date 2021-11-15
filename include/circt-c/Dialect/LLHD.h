//===-- LLHD.h - C API for LLHD dialect -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_LLHD_H
#define CIRCT_C_DIALECT_LLHD_H

#include "mlir-c/Registration.h"

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
/// Check if a type is a signal type.
MLIR_CAPI_EXPORTED bool llhdTypeIsASignalType(MlirType);
/// Check if a type is a pointer type.
MLIR_CAPI_EXPORTED bool llhdTypeIsAPointerType(MlirType);

/// Create a time type.
MLIR_CAPI_EXPORTED MlirType llhdTimeTypeGet(MlirContext ctx);
/// Create a signal type.
MLIR_CAPI_EXPORTED MlirType llhdSignalTypeGet(MlirType element);
/// Create a pointer type.
MLIR_CAPI_EXPORTED MlirType llhdPointerTypeGet(MlirType element);

/// Get the inner type of a signal.
MLIR_CAPI_EXPORTED MlirType llhdSignalTypeGetElementType(MlirType);
/// Get the inner type of a pointer.
MLIR_CAPI_EXPORTED MlirType llhdPointerTypeGetElementType(MlirType);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_LLHD_H
