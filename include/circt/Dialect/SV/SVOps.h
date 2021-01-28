//===- SVOps.h - Declare SV dialect operations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_OPS_H
#define CIRCT_DIALECT_SV_OPS_H

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// TODO: Remove this when OpAsmInterface qualfies this type. (To check, just
// delete this and see if it still compiles.)
namespace circt {
namespace sv {
using mlir::OpAsmOpInterface;
using mlir::OpAsmSetValueNameFn;
} // namespace sv
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/SV/SVEnums.h.inc"
// Clang format shouldn't reorder these headers.
#include "circt/Dialect/SV/SV.h.inc"
#include "circt/Dialect/SV/SVStructs.h.inc"

namespace circt {
namespace sv {

/// Return true if the specified operation is an expression.
bool isExpression(mlir::Operation *op);

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_OPS_H
