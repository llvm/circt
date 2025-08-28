//===- SV.h - C interface for the SV dialect ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_SV_H
#define CIRCT_C_DIALECT_SV_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SystemVerilog, sv);
MLIR_CAPI_EXPORTED void registerSVPasses(void);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool svAttrIsASVAttributeAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute svSVAttributeAttrGet(MlirContext,
                                                      MlirStringRef name,
                                                      MlirStringRef expression,
                                                      bool emitAsComment);
MLIR_CAPI_EXPORTED MlirStringRef svSVAttributeAttrGetName(MlirAttribute);
MLIR_CAPI_EXPORTED MlirStringRef svSVAttributeAttrGetExpression(MlirAttribute);
MLIR_CAPI_EXPORTED bool svSVAttributeAttrGetEmitAsComment(MlirAttribute);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool svTypeIsAInterfaceType(MlirType);
MLIR_CAPI_EXPORTED MlirType svInterfaceTypeGet(MlirContext, MlirStringRef);
MLIR_CAPI_EXPORTED MlirStringRef svInterfaceTypeGetInterfaceSym(MlirType);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_SV_H
