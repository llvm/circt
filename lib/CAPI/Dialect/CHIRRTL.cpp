//===- CHIRRTL.cpp - C interface for the CHIRRTL dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/CHIRRTL.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace chirrtl;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CHIRRTL, chirrtl,
                                      circt::chirrtl::CHIRRTLDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MlirType chirrtlTypeGetCMemory(MlirContext ctx, MlirType elementType,
                               uint64_t numElements) {
  auto baseType = unwrap(elementType).cast<firrtl::FIRRTLBaseType>();
  assert(baseType && "element must be base type");

  return wrap(CMemoryType::get(unwrap(ctx), baseType, numElements));
}

MlirType chirrtlTypeGetCMemoryPort(MlirContext ctx) {
  return wrap(CMemoryPortType::get(unwrap(ctx)));
}
