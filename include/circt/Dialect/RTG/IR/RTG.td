//===- RTG.td - RTG top-level definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTG_TD
#define CIRCT_DIALECT_RTG_IR_RTG_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"

include "circt/Dialect/RTG/IR/RTGDialect.td"
include "circt/Dialect/RTG/IR/RTGTypes.td"
include "circt/Dialect/RTG/IR/RTGOps.td"

#endif // CIRCT_DIALECT_RTG_IR_RTG_TD
