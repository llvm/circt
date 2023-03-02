//===- Dialect.h - Arc dialect definition -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_DIALECT_H
#define CIRCT_DIALECT_ARC_DIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/Arc/ArcDialect.h.inc"

#endif // CIRCT_DIALECT_ARC_DIALECT_H
