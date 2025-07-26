//===- SimDialect.cpp - Implement the Sim dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Sim dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace sim;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void SimDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Sim/Sim.cpp.inc"
      >();

  registerTypes();
}

#include "circt/Dialect/Sim/SimDialect.cpp.inc"

Operation *SimDialect::materializeConstant(::mlir::OpBuilder &builder,
                                           ::mlir::Attribute value,
                                           ::mlir::Type type,
                                           ::mlir::Location loc) {

  if (auto fmtStrType = llvm::dyn_cast<FormatStringType>(type))
    return FormatLitOp::create(builder, loc, fmtStrType,
                               llvm::cast<StringAttr>(value));
  return nullptr;
}
