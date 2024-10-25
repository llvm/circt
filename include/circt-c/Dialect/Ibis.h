//===- Ibis.h - C interface for Ibis dialect ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_IBIS_H
#define CIRCT_C_DIALECT_IBIS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ibis, ibis);
MLIR_CAPI_EXPORTED void registerIbisPasses(void);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_IBIS_H
