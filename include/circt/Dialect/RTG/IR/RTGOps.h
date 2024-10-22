//===- RTGOps.h - Declare RTG dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGOPS_H
#define CIRCT_DIALECT_RTG_IR_RTGOPS_H

#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "circt/Dialect/RTG/IR/RTGInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/RTG/IR/RTG.h.inc"

#endif // CIRCT_DIALECT_RTG_IR_RTGOPS_H
