//===- APIntOps.h - Declare APInt dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the APInt dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_APINT_APINTOPS_H
#define CIRCT_DIALECT_APINT_APINTOPS_H

#include "circt/Dialect/APInt/APIntDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "circt/Dialect/APInt/APInt.h.inc"

#endif // CIRCT_DIALECT_APINT_APINTOPS_H
