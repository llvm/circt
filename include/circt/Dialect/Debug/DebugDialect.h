//===- DebugDialect.h - Debug dialect definition ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGDIALECT_H
#define CIRCT_DIALECT_DEBUG_DEBUGDIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

// Attribute definitions generated from `DebugTypes.td`. The dialect's
// discardable attributes refer to them.
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Debug/DebugAttributes.h.inc"

// Dialect definition generated from `DebugDialect.td`
#include "circt/Dialect/Debug/DebugDialect.h.inc"

#endif // CIRCT_DIALECT_DEBUG_DEBUGDIALECT_H
