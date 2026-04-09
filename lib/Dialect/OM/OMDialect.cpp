//===- OMDialect.cpp - Object Model dialect definition --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model dialect definition.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "mlir/IR/Builders.h"

#include "circt/Dialect/OM/OMDialect.cpp.inc"

void circt::om::OMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/OM/OM.cpp.inc"
      >();

  registerTypes();
  registerAttributes();
}

mlir::Operation *
circt::om::OMDialect::materializeConstant(mlir::OpBuilder &builder,
                                          mlir::Attribute value,
                                          mlir::Type type, mlir::Location loc) {
  if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(value))
    if (typedAttr.getType() == type)
      return ConstantOp::create(builder, loc, typedAttr);
  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/OM/OMEnums.cpp.inc"
