//===- OFIRDialect.h - OFIR dialect declaration ------------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an MLIR dialect for the OFIR IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OFIR_DIALECT_H
#define CIRCT_DIALECT_OFIR_DIALECT_H

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/OFIR/OFIRDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
//#include "circt/Dialect/OFIR/OFIREnums.h.inc"

#endif // CIRCT_DIALECT_OFIR_IR_DIALECT_H
