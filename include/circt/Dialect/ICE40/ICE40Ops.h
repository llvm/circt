//===- ICE40Ops.h - Declare ICE40 dialect operations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the ICE40 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ICE40_ICE40OPS_H
#define CIRCT_DIALECT_ICE40_ICE40OPS_H

#include "circt/Dialect/ICE40/ICE40Dialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "circt/Dialect/ICE40/ICE40.h.inc"

#endif // CIRCT_DIALECT_ICE40_ICE40OPS_H
