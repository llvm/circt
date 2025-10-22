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

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Moore/MooreAttributes.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

#define GET_OP_CLASSES
// Clang format shouldn't reorder these headers.
#include "circt/Dialect/Moore/Moore.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOOREOPS_H
