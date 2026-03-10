//===- LLHDDialect.cpp - Implement the LLHD dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/LLHDDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace circt;
using namespace circt::llhd;

//===----------------------------------------------------------------------===//
// LLHDDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with LLHD operations.
struct LLHDInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within LLHD can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *src, bool, IRMapping &) const final {
    return false;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LLHD Dialect
//===----------------------------------------------------------------------===//

void LLHDDialect::initialize() {
  registerTypes();
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LLHD/LLHD.cpp.inc"
      >();

  addInterfaces<LLHDInlinerInterface>();

  mlir::DialectRegistry registry;
  registerDestructableIntegerExternalModel(registry);
  getContext()->appendDialectRegistry(registry);
}

Operation *LLHDDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (auto timeAttr = dyn_cast<TimeAttr>(value))
    return llhd::ConstantTimeOp::create(builder, loc, type, timeAttr);

  if (auto intAttr = dyn_cast<IntegerAttr>(value))
    return hw::ConstantOp::create(builder, loc, type, intAttr);

  return nullptr;
}

#include "circt/Dialect/LLHD/LLHDDialect.cpp.inc"
