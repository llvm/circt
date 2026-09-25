//===- PIREnums.h - Enums for PIR dialect -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines PIR dialect specific enums.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIR_PIRENUMS_H
#define CIRCT_DIALECT_PIR_PIRENUMS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/PIR/PIREnums.h.inc"

#endif // CIRCT_DIALECT_PIR_PIRENUMS_H
