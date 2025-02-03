//===- RTGTest.cpp - C interface for the RTGTest dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/RTGTest.h"
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h"

#include "mlir/CAPI/Registration.h"

using namespace circt;
using namespace circt::rtgtest;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RTGTest, rtgtest, RTGTestDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool rtgtestTypeIsACPU(MlirType type) { return isa<CPUType>(unwrap(type)); }

MlirType rtgtestCPUTypeGet(MlirContext ctxt) {
  return wrap(CPUType::get(unwrap(ctxt)));
}

bool rtgtestTypeIsAIntegerRegister(MlirType type) {
  return isa<IntegerRegisterType>(unwrap(type));
}

MlirType rtgtestIntegerRegisterTypeGet(MlirContext ctxt) {
  return wrap(IntegerRegisterType::get(unwrap(ctxt)));
}

// Immediates.
//===----------------------------------------------------------------------===//

bool rtgtestTypeIsAImm12(MlirType type) { return isa<Imm12Type>(unwrap(type)); }

MlirType rtgtestImm12TypeGet(MlirContext ctxt) {
  return wrap(Imm12Type::get(unwrap(ctxt)));
}

bool rtgtestTypeIsAImm13(MlirType type) { return isa<Imm13Type>(unwrap(type)); }

MlirType rtgtestImm13TypeGet(MlirContext ctxt) {
  return wrap(Imm13Type::get(unwrap(ctxt)));
}

bool rtgtestTypeIsAImm21(MlirType type) { return isa<Imm21Type>(unwrap(type)); }

MlirType rtgtestImm21TypeGet(MlirContext ctxt) {
  return wrap(Imm21Type::get(unwrap(ctxt)));
}

bool rtgtestTypeIsAImm32(MlirType type) { return isa<Imm32Type>(unwrap(type)); }

MlirType rtgtestImm32TypeGet(MlirContext ctxt) {
  return wrap(Imm32Type::get(unwrap(ctxt)));
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

bool rtgtestAttrIsACPU(MlirAttribute attr) {
  return isa<CPUAttr>(unwrap(attr));
}

MlirAttribute rtgtestCPUAttrGet(MlirContext ctxt, unsigned id) {
  return wrap(CPUAttr::get(unwrap(ctxt), id));
}

unsigned rtgtestCPUAttrGetId(MlirAttribute attr) {
  return cast<CPUAttr>(unwrap(attr)).getId();
}

// Registers
//===----------------------------------------------------------------------===//

bool rtgtestAttrIsARegZero(MlirAttribute attr) {
  return isa<RegZeroAttr>(unwrap(attr));
}

MlirAttribute rtgtestRegZeroAttrGet(MlirContext ctxt) {
  return wrap(RegZeroAttr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegRa(MlirAttribute attr) {
  return isa<RegRaAttr>(unwrap(attr));
}

MlirAttribute rtgtestRegRaAttrGet(MlirContext ctxt) {
  return wrap(RegRaAttr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegSp(MlirAttribute attr) {
  return isa<RegSpAttr>(unwrap(attr));
}

MlirAttribute rtgtestRegSpAttrGet(MlirContext ctxt) {
  return wrap(RegSpAttr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegGp(MlirAttribute attr) {
  return isa<RegGpAttr>(unwrap(attr));
}

MlirAttribute rtgtestRegGpAttrGet(MlirContext ctxt) {
  return wrap(RegGpAttr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegTp(MlirAttribute attr) {
  return isa<RegTpAttr>(unwrap(attr));
}

MlirAttribute rtgtestRegTpAttrGet(MlirContext ctxt) {
  return wrap(RegTpAttr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegT0(MlirAttribute attr) {
  return isa<RegT0Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegT0AttrGet(MlirContext ctxt) {
  return wrap(RegT0Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegT1(MlirAttribute attr) {
  return isa<RegT1Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegT1AttrGet(MlirContext ctxt) {
  return wrap(RegT1Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegT2(MlirAttribute attr) {
  return isa<RegT2Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegT2AttrGet(MlirContext ctxt) {
  return wrap(RegT2Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS0(MlirAttribute attr) {
  return isa<RegS0Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS0AttrGet(MlirContext ctxt) {
  return wrap(RegS0Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS1(MlirAttribute attr) {
  return isa<RegS1Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS1AttrGet(MlirContext ctxt) {
  return wrap(RegS1Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA0(MlirAttribute attr) {
  return isa<RegA0Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA0AttrGet(MlirContext ctxt) {
  return wrap(RegA0Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA1(MlirAttribute attr) {
  return isa<RegA1Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA1AttrGet(MlirContext ctxt) {
  return wrap(RegA1Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA2(MlirAttribute attr) {
  return isa<RegA2Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA2AttrGet(MlirContext ctxt) {
  return wrap(RegA2Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA3(MlirAttribute attr) {
  return isa<RegA3Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA3AttrGet(MlirContext ctxt) {
  return wrap(RegA3Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA4(MlirAttribute attr) {
  return isa<RegA4Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA4AttrGet(MlirContext ctxt) {
  return wrap(RegA4Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA5(MlirAttribute attr) {
  return isa<RegA5Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA5AttrGet(MlirContext ctxt) {
  return wrap(RegA5Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA6(MlirAttribute attr) {
  return isa<RegA6Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA6AttrGet(MlirContext ctxt) {
  return wrap(RegA6Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegA7(MlirAttribute attr) {
  return isa<RegA7Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegA7AttrGet(MlirContext ctxt) {
  return wrap(RegA7Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS2(MlirAttribute attr) {
  return isa<RegS2Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS2AttrGet(MlirContext ctxt) {
  return wrap(RegS2Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS3(MlirAttribute attr) {
  return isa<RegS3Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS3AttrGet(MlirContext ctxt) {
  return wrap(RegS3Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS4(MlirAttribute attr) {
  return isa<RegS4Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS4AttrGet(MlirContext ctxt) {
  return wrap(RegS4Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS5(MlirAttribute attr) {
  return isa<RegS5Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS5AttrGet(MlirContext ctxt) {
  return wrap(RegS5Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS6(MlirAttribute attr) {
  return isa<RegS6Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS6AttrGet(MlirContext ctxt) {
  return wrap(RegS6Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS7(MlirAttribute attr) {
  return isa<RegS7Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS7AttrGet(MlirContext ctxt) {
  return wrap(RegS7Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS8(MlirAttribute attr) {
  return isa<RegS8Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS8AttrGet(MlirContext ctxt) {
  return wrap(RegS8Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS9(MlirAttribute attr) {
  return isa<RegS9Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS9AttrGet(MlirContext ctxt) {
  return wrap(RegS9Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS10(MlirAttribute attr) {
  return isa<RegS10Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS10AttrGet(MlirContext ctxt) {
  return wrap(RegS10Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegS11(MlirAttribute attr) {
  return isa<RegS11Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegS11AttrGet(MlirContext ctxt) {
  return wrap(RegS11Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegT3(MlirAttribute attr) {
  return isa<RegT3Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegT3AttrGet(MlirContext ctxt) {
  return wrap(RegT3Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegT4(MlirAttribute attr) {
  return isa<RegT4Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegT4AttrGet(MlirContext ctxt) {
  return wrap(RegT4Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegT5(MlirAttribute attr) {
  return isa<RegT5Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegT5AttrGet(MlirContext ctxt) {
  return wrap(RegT5Attr::get(unwrap(ctxt)));
}

bool rtgtestAttrIsARegT6(MlirAttribute attr) {
  return isa<RegT6Attr>(unwrap(attr));
}

MlirAttribute rtgtestRegT6AttrGet(MlirContext ctxt) {
  return wrap(RegT6Attr::get(unwrap(ctxt)));
}

// Immediates.
//===----------------------------------------------------------------------===//

bool rtgtestAttrIsAImm12(MlirAttribute attr) {
  return isa<Imm12Attr>(unwrap(attr));
}

MlirAttribute rtgtestImm12AttrGet(MlirContext ctxt, unsigned value) {
  return wrap(Imm12Attr::get(unwrap(ctxt), value));
}

unsigned rtgtestImm12AttrGetValue(MlirAttribute attr) {
  return cast<Imm12Attr>(unwrap(attr)).getValue();
}

bool rtgtestAttrIsAImm13(MlirAttribute attr) {
  return isa<Imm13Attr>(unwrap(attr));
}

MlirAttribute rtgtestImm13AttrGet(MlirContext ctxt, unsigned value) {
  return wrap(Imm13Attr::get(unwrap(ctxt), value));
}

unsigned rtgtestImm13AttrGetValue(MlirAttribute attr) {
  return cast<Imm13Attr>(unwrap(attr)).getValue();
}

bool rtgtestAttrIsAImm21(MlirAttribute attr) {
  return isa<Imm21Attr>(unwrap(attr));
}

MlirAttribute rtgtestImm21AttrGet(MlirContext ctxt, unsigned value) {
  return wrap(Imm21Attr::get(unwrap(ctxt), value));
}

unsigned rtgtestImm21AttrGetValue(MlirAttribute attr) {
  return cast<Imm21Attr>(unwrap(attr)).getValue();
}

bool rtgtestAttrIsAImm32(MlirAttribute attr) {
  return isa<Imm32Attr>(unwrap(attr));
}

MlirAttribute rtgtestImm32AttrGet(MlirContext ctxt, unsigned value) {
  return wrap(Imm32Attr::get(unwrap(ctxt), value));
}

unsigned rtgtestImm32AttrGetValue(MlirAttribute attr) {
  return cast<Imm32Attr>(unwrap(attr)).getValue();
}
