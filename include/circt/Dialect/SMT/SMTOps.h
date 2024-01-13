//===- SMTOps.h - SMT dialect operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTOPS_H
#define CIRCT_DIALECT_SMT_SMTOPS_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/SMT/SMTAttributes.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "circt/Dialect/SMT/SMTTypes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/SMT/SMT.h.inc"

#endif // CIRCT_DIALECT_SMT_SMTOPS_H
