//===- SystemCOps.h - Declare SystemC dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCOPS_H
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCOPS_H

#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "circt/Dialect/SystemC/SystemCTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "circt/Dialect/SystemC/SystemCEnums.h.inc"
// Clang format shouldn't reorder these headers.
#include "circt/Dialect/SystemC/SystemC.h.inc"

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCOPS_H
