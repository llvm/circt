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
