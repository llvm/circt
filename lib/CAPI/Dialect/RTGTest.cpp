//===- RTGTest.cpp - C interface for the RTGTest dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/RTGTest.h"
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

bool rtgTypeIsIntegerReg(MlirType type) {
  return isa<IntegerRegisterType>(unwrap(type));
}

MlirType rtgIntegerRegTypeGet(MlirContext ctxt) {
  return wrap(IntegerRegisterType::get(unwrap(ctxt)));
}

bool rtgTypeIsFloatReg(MlirType type) {
  return isa<FloatRegisterType>(unwrap(type));
}

MlirType rtgFloatRegTypeGet(MlirContext ctxt) {
  return wrap(FloatRegisterType::get(unwrap(ctxt)));
}

bool rtgTypeIsVectorReg(MlirType type) {
  return isa<VectorRegisterType>(unwrap(type));
}

MlirType rtgVectorRegTypeGet(MlirContext ctxt) {
  return wrap(VectorRegisterType::get(unwrap(ctxt)));
}
