//===- MooreOps.h - Declare Moore dialect operations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREOPS_H
#define CIRCT_DIALECT_MOORE_MOOREOPS_H

#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Moore/MooreEnums.h.inc"
// Clang format shouldn't reorder these headers.
#include "circt/Dialect/Moore/Moore.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOOREOPS_H
