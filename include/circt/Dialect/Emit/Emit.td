//===- Emit.td - Emit dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the `emit` dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMIT_TD
#define CIRCT_DIALECT_EMIT_EMIT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "circt/Dialect/Emit/EmitDialect.td"
include "circt/Dialect/Emit/EmitOps.td"

#endif // CIRCT_DIALECT_EMIT_EMIT_TD
