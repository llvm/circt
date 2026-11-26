//===- VerifDialect.cpp - Verif dialect implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace verif;

Operation *VerifDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto intType = dyn_cast<IntegerType>(type))
    if (auto attrValue = dyn_cast<IntegerAttr>(value))
      return hw::ConstantOp::create(builder, loc, type, attrValue);
  return nullptr;
}

#include "circt/Dialect/Verif/VerifDialect.cpp.inc"
#include "circt/Dialect/Verif/VerifEnums.cpp.inc"
