//===- DebugOps.h - Debug dialect operations ====----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGOPS_H
#define CIRCT_DIALECT_DEBUG_DEBUGOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Debug/DebugTypes.h"

// Operation definitions generated from `Debug.td`
#define GET_OP_CLASSES
#include "circt/Dialect/Debug/Debug.h.inc"

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_H
