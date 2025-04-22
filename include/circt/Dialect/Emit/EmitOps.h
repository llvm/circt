//===- EmitOps.h - Declare Emit dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Emit dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMITOPS_H
#define CIRCT_DIALECT_EMIT_EMITOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Support/BuilderUtils.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Emit/Emit.h.inc"

namespace circt {
namespace emit {

/// Return the name of the fragments array attribute.
inline StringRef getFragmentsAttrName() { return "emit.fragments"; }

} // namespace emit
} // namespace circt

#endif // CIRCT_DIALECT_EMIT_EMITOPS_H
