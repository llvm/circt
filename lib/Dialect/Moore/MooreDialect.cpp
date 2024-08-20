//===- MooreDialect.cpp - Implement the Moore dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Moore/MooreOps.h"

using namespace circt;
using namespace circt::moore;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void MooreDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Moore/Moore.cpp.inc"
      >();
}

Operation *MooreDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto intType = dyn_cast<IntType>(type))
    if (auto intValue = dyn_cast<FVIntegerAttr>(value))
      return builder.create<ConstantOp>(loc, intType, intValue);
  return nullptr;
}

#include "circt/Dialect/Moore/MooreDialect.cpp.inc"
