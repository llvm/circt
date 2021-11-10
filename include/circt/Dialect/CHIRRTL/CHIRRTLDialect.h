//===- CHIRRTLDialect.h - CHIRRTL dialect declaration ----------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an MLIR dialect for the CHIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHIRRTL_DIALECT_H
#define CIRCT_DIALECT_CHIRRTL_DIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/CHIRRTL/CHIRRTLDialect.h.inc"

#endif // CIRCT_DIALECT_CHIRRTL_DIALECT_H