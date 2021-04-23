//===- FIRRTLDialect.h - FIRRTL dialect declaration ------------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_DIALECT_H
#define CIRCT_DIALECT_FIRRTL_DIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace firrtl {

class FIRRTLType;

/// If the specified attribute list has a firrtl.name attribute, return its
/// value.
StringAttr getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs);

} // namespace firrtl
} // namespace circt

// Pull in the dialect definition.
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/FIRRTL/FIRRTLEnums.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_IR_DIALECT_H
