//===- SimOps.h - Declare Sim dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Sim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPS_H
#define CIRCT_DIALECT_SIM_SIMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Support/BuilderUtils.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Sim/Sim.h.inc"

#endif // CIRCT_DIALECT_SIM_SIMOPS_H
