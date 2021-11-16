//===- StaticLogic.h - StaticLogic Definitions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an StaticLogic MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STATICLOGIC_OPS_H_
#define CIRCT_STATICLOGIC_OPS_H_

#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/StaticLogic/StaticLogicDialect.h.inc"

// Represents a recurrence-constrained minimum initiation interval. The ii
// represents the numerical value itself, and the iter arg def and use hold the
// definition and use of Values that constrain the ii.
struct RecMII {
  uint64_t ii;
  mlir::Value iterArgDef;
  mlir::OpOperand *iterArgUse;
};

#define GET_OP_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogic.h.inc"

#endif // CIRCT_STATICLOGIC_OPS_H_
