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

// Immediates.
//===----------------------------------------------------------------------===//

/// If the type is an RTGTest Imm5Type.
MLIR_CAPI_EXPORTED bool rtgtestTypeIsAImm5(MlirType type);

/// Creates an RTGTest Imm5 type in the context.
MLIR_CAPI_EXPORTED MlirType rtgtestImm5TypeGet(MlirContext ctxt);

/// If the type is an RTGTest Imm12Type.
MLIR_CAPI_EXPORTED bool rtgtestTypeIsAImm12(MlirType type);

/// Creates an RTGTest Imm12 type in the context.
MLIR_CAPI_EXPORTED MlirType rtgtestImm12TypeGet(MlirContext ctxt);

/// If the type is an RTGTest Imm13Type.
MLIR_CAPI_EXPORTED bool rtgtestTypeIsAImm13(MlirType type);

/// Creates an RTGTest Imm13 type in the context.
MLIR_CAPI_EXPORTED MlirType rtgtestImm13TypeGet(MlirContext ctxt);

/// If the type is an RTGTest Imm21Type.
MLIR_CAPI_EXPORTED bool rtgtestTypeIsAImm21(MlirType type);

/// Creates an RTGTest Imm21 type in the context.
MLIR_CAPI_EXPORTED MlirType rtgtestImm21TypeGet(MlirContext ctxt);

/// If the type is an RTGTest Imm32Type.
MLIR_CAPI_EXPORTED bool rtgtestTypeIsAImm32(MlirType type);

/// Creates an RTGTest Imm32 type in the context.
MLIR_CAPI_EXPORTED MlirType rtgtestImm32TypeGet(MlirContext ctxt);

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

/// If the attribute is an RTGTest RegA6Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA6(MlirAttribute attr);

/// Creates an RTGTest RegA6 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA6AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegA7Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegA7(MlirAttribute attr);

/// Creates an RTGTest RegA7 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegA7AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS2Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS2(MlirAttribute attr);

/// Creates an RTGTest RegS2 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS2AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS3Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS3(MlirAttribute attr);

/// Creates an RTGTest RegS3 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS3AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS4Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS4(MlirAttribute attr);

/// Creates an RTGTest RegS4 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS4AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS5Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS5(MlirAttribute attr);

/// Creates an RTGTest RegS5 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS5AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS6Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS6(MlirAttribute attr);

/// Creates an RTGTest RegS6 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS6AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS7Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS7(MlirAttribute attr);

/// Creates an RTGTest RegS7 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS7AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS8Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS8(MlirAttribute attr);

/// Creates an RTGTest RegS8 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS8AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS9Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS9(MlirAttribute attr);

/// Creates an RTGTest RegS9 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS9AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS10Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS10(MlirAttribute attr);

/// Creates an RTGTest RegS10 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS10AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegS11Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegS11(MlirAttribute attr);

/// Creates an RTGTest RegS11 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegS11AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegT3Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegT3(MlirAttribute attr);

/// Creates an RTGTest RegT3 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegT3AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegT4Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegT4(MlirAttribute attr);

/// Creates an RTGTest RegT4 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegT4AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegT5Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegT5(MlirAttribute attr);

/// Creates an RTGTest RegT5 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegT5AttrGet(MlirContext ctxt);

/// If the attribute is an RTGTest RegT6Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsARegT6(MlirAttribute attr);

/// Creates an RTGTest RegT6 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestRegT6AttrGet(MlirContext ctxt);

// Immediates.
//===----------------------------------------------------------------------===//

/// If the attribute is an RTGTest Imm5Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsAImm5(MlirAttribute attr);

/// Creates an RTGTest Imm5 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestImm5AttrGet(MlirContext ctxt,
                                                    unsigned value);

/// Returns the value represented by the Imm5 attribute.
MLIR_CAPI_EXPORTED unsigned rtgtestImm5AttrGetValue(MlirAttribute attr);

/// If the attribute is an RTGTest Imm12Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsAImm12(MlirAttribute attr);

/// Creates an RTGTest Imm12 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestImm12AttrGet(MlirContext ctxt,
                                                     unsigned value);

/// Returns the value represented by the Imm12 attribute.
MLIR_CAPI_EXPORTED unsigned rtgtestImm12AttrGetValue(MlirAttribute attr);

/// If the attribute is an RTGTest Imm13Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsAImm13(MlirAttribute attr);

/// Creates an RTGTest Imm13 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestImm13AttrGet(MlirContext ctxt,
                                                     unsigned value);

/// Returns the value represented by the Imm13 attribute.
MLIR_CAPI_EXPORTED unsigned rtgtestImm13AttrGetValue(MlirAttribute attr);

/// If the attribute is an RTGTest Imm21Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsAImm21(MlirAttribute attr);

/// Creates an RTGTest Imm21 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestImm21AttrGet(MlirContext ctxt,
                                                     unsigned value);

/// Returns the value represented by the Imm21 attribute.
MLIR_CAPI_EXPORTED unsigned rtgtestImm21AttrGetValue(MlirAttribute attr);

/// If the attribute is an RTGTest Imm32Attr.
MLIR_CAPI_EXPORTED bool rtgtestAttrIsAImm32(MlirAttribute attr);

/// Creates an RTGTest Imm32 attribute in the context.
MLIR_CAPI_EXPORTED MlirAttribute rtgtestImm32AttrGet(MlirContext ctxt,
                                                     unsigned value);

/// Returns the value represented by the Imm32 attribute.
MLIR_CAPI_EXPORTED unsigned rtgtestImm32AttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_RTGTEST_H
