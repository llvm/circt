//===- Handshake.h - C interface for Handshake dialect ------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_HANDSHAKE_H
#define CIRCT_C_DIALECT_HANDSHAKE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Handshake, handshake);
MLIR_CAPI_EXPORTED void registerHandshakePasses(void);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_HANDSHAKE_H
