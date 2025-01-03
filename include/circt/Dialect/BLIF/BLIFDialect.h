//===- BLIFDialect.h - BLIF dialect declaration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an BLIF MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_BLIF_BLIFDIALECT_H
#define CIRCT_DIALECT_BLIF_BLIFDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/BLIF/BLIFDialect.h.inc"
#include "circt/Dialect/BLIF/BLIFEnums.h.inc"

#endif // CIRCT_DIALECT_BLIF_BLIFDIALECT_H
