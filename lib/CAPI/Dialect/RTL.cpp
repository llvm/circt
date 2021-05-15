//===- RTLDialect.cpp - C Interface for the RTL Dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements a C Interface for the RTL Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/HW.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace circt::rtl;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RTL, rtl, RTLDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

int64_t rtlGetBitWidth(MlirType type) { return getBitWidth(unwrap(type)); }

bool rtlTypeIsAValueType(MlirType type) { return isRTLValueType(unwrap(type)); }

bool rtlTypeIsAArrayType(MlirType type) {
  return unwrap(type).isa<ArrayType>();
}

MlirType rtlArrayTypeGet(MlirType element, size_t size) {
  return wrap(ArrayType::get(unwrap(element), size));
}

MlirType rtlArrayTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<ArrayType>().getElementType());
}

intptr_t rtlArrayTypeGetSize(MlirType type) {
  return unwrap(type).cast<ArrayType>().getSize();
}

MlirType rtlInOutTypeGet(MlirType element) {
  return wrap(InOutType::get(unwrap(element)));
}

MlirType rtlInOutTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<InOutType>().getElementType());
}

bool rtlTypeIsAInOut(MlirType type) { return unwrap(type).isa<InOutType>(); }

bool rtlTypeIsAStructType(MlirType type) {
  return unwrap(type).isa<StructType>();
}

MlirType rtlStructTypeGet(MlirContext ctx, intptr_t numElements,
                          RTLStructFieldInfo const *elements) {
  SmallVector<StructType::FieldInfo> fieldInfos;
  fieldInfos.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i) {
    auto typeAttr = unwrap(elements[i].attribute).dyn_cast<TypeAttr>();
    if (!typeAttr)
      return MlirType();
    fieldInfos.push_back(
        StructType::FieldInfo{unwrap(elements[i].name), typeAttr.getValue()});
  }
  return wrap(StructType::get(unwrap(ctx), fieldInfos));
}
