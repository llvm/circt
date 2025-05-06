//===- DatapathOps.h - Datapath dialect operations --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_DATAPATHOPS_H
#define CIRCT_DIALECT_DATAPATH_DATAPATHOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"

#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/HW/HWTypes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Datapath/Datapath.h.inc"

#endif // CIRCT_DIALECT_DATAPATH_DATAPATHOPS_H
