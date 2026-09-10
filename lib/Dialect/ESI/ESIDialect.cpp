//===- ESIDialect.cpp - ESI dialect code defs -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect definitions. Should be relatively standard boilerplate.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace circt::esi;

void ESIDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/ESI/ESI.cpp.inc"
      >();
}

Operation *ESIDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (isa<mlir::UnitAttr>(value))
    return hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
  return hw::materializeConstant(builder, value, type, loc);
}

// Provide implementations for the enums we use.
#include "circt/Dialect/ESI/ESIEnums.cpp.inc"

#include "circt/Dialect/ESI/ESIDialect.cpp.inc"
