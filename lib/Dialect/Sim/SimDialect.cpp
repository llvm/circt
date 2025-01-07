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
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Sim/SimAttributes.h"
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
  registerAttributes();
}

#include "circt/Dialect/Sim/SimDialect.cpp.inc"

Operation *SimDialect::materializeConstant(::mlir::OpBuilder &builder,
                                           ::mlir::Attribute value,
                                           ::mlir::Type type,
                                           ::mlir::Location loc) {

  // Delegate non 'sim' types to the HW dialect materializer.
  if (!isa<SimDialect>(type.getDialect()))
    return builder.getContext()
        ->getLoadedDialect<hw::HWDialect>()
        ->materializeConstant(builder, value, type, loc);

  if (auto fmtStrType = llvm::dyn_cast<FormatStringType>(type))
    return builder.create<FormatLitOp>(loc, fmtStrType,
                                       llvm::cast<StringAttr>(value));

  if (auto neverTriggerAttr = llvm::dyn_cast<NeverTriggerAttr>(value))
    return builder.create<NeverOp>(loc, neverTriggerAttr.getType());

  return nullptr;
}
