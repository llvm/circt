//===- CHALKOps.h - Definition of CHALK dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKOPS_H
#define CIRCT_DIALECT_CHALK_CHALKOPS_H

#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Dialect/CHALK/CHALKTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/CHALK/CHALK.h.inc"

#endif // CIRCT_DIALECT_CHALK_CHALKOPS_H
