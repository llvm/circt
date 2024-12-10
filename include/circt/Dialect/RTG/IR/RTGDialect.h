//===- RTGDialect.h - RTG dialect declaration -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RTG MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGDIALECT_H
#define CIRCT_DIALECT_RTG_IR_RTGDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/RTG/IR/RTGEnums.h.inc"

// Pull in the Dialect definition.
#include "circt/Dialect/RTG/IR/RTGDialect.h.inc"

#endif // CIRCT_DIALECT_RTG_IR_RTGDIALECT_H
