//===- RTGTestDialect.h - RTG Test dialect declaration ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RTGTest MLIR dialect.  This dialect defines a minimal
// set of operations to use for testing the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTDIALECT_H
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// Pull in the Dialect definition.
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/RTGTest/IR/RTGTestEnums.h.inc"

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTDIALECT_H
