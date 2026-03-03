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

/// If the type is an RTGTest IntegerRegisterType.
MLIR_CAPI_EXPORTED bool rtgtestTypeIsAIntegerRegister(MlirType type);

/// Creates an RTGTest IntegerRegisterType in the context.
MLIR_CAPI_EXPORTED MlirType rtgtestIntegerRegisterTypeGet(MlirContext ctxt);

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

// Registers.
//===----------------------------------------------------------------------===//

/// If the type is an RTGTest RegZeroAttr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegZero(MlirAttribute attr);

/// Creates an RTGTest RegZero attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegZeroAttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegRaAttr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegRa(MlirAttribute attr);

/// Creates an RTGTest RegRa attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegRaAttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegSpAttr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegSp(MlirAttribute attr);

/// Creates an RTGTest RegSp attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegSpAttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegGpAttr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegGp(MlirAttribute attr);

/// Creates an RTGTest RegGp attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegGpAttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegTpAttr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegTp(MlirAttribute attr);

/// Creates an RTGTest RegTp attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegTpAttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegT0Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegT0(MlirAttribute attr);

/// Creates an RTGTest RegT0 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegT0AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegT1Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegT1(MlirAttribute attr);

/// Creates an RTGTest RegT1 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegT1AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegT2Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegT2(MlirAttribute attr);

/// Creates an RTGTest RegT2 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegT2AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS0Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS0(MlirAttribute attr);

/// Creates an RTGTest RegS0 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS0AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS1Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS1(MlirAttribute attr);

/// Creates an RTGTest RegS1 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS1AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegA0Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA0(MlirAttribute attr);

/// Creates an RTGTest RegA0 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA0AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegA1Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA1(MlirAttribute attr);

/// Creates an RTGTest RegA1 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA1AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegA2Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA2(MlirAttribute attr);

/// Creates an RTGTest RegA2 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA2AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegA3Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA3(MlirAttribute attr);

/// Creates an RTGTest RegA3 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA3AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegA4Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA4(MlirAttribute attr);

/// Creates an RTGTest RegA4 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA4AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegA5Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA5(MlirAttribute attr);

/// Creates an RTGTest RegA5 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA5AttrGet(MlirContext ctxt);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTGTEST_H
