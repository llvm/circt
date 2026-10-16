//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_SIM_H
#define CIRCT_C_DIALECT_SIM_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Sim, sim);

//===----------------------------------------------------------------------===//
// DPI function types.
//===----------------------------------------------------------------------===//

typedef enum {
  SIM_DPI_DIRECTION_INPUT,
  SIM_DPI_DIRECTION_OUTPUT,
  SIM_DPI_DIRECTION_INOUT,
  SIM_DPI_DIRECTION_RETURN,
  SIM_DPI_DIRECTION_REF,
} SimDPIDirection;

typedef struct {
  MlirStringRef name;
  MlirType type;
  SimDPIDirection direction;
} SimDPIArgument;

MLIR_CAPI_EXPORTED bool simTypeIsADPIFunction(MlirType type);

/// Create a DPI function type from arguments in declaration order.
MLIR_CAPI_EXPORTED MlirType simDPIFunctionTypeGet(
    MlirContext ctx, intptr_t numArguments, const SimDPIArgument *arguments);

MLIR_CAPI_EXPORTED intptr_t simDPIFunctionTypeGetNumArguments(MlirType type);

/// Return the argument at index. The name is owned by the type's context.
/// Requires 0 <= index < simDPIFunctionTypeGetNumArguments(type).
MLIR_CAPI_EXPORTED SimDPIArgument simDPIFunctionTypeGetArgument(MlirType type,
                                                                intptr_t index);

/// Return the derived call signature, which omits argument names and
/// directions.
MLIR_CAPI_EXPORTED MlirType simDPIFunctionTypeGetFunctionType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_SIM_H
