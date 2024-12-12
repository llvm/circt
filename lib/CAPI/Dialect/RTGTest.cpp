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

bool rtgtestTypeIsACPU(MlirType type) { return isa<CPUType>(unwrap(type)); }

MlirType rtgtestCPUTypeGet(MlirContext ctxt) {
  return wrap(CPUType::get(unwrap(ctxt)));
}
