//===- BLIFOps.h - Declare BLIF dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the BLIF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_BLIF_BLIFOPS_H
#define CIRCT_DIALECT_BLIF_BLIFOPS_H

#include "circt/Dialect/BLIF/BLIFDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "circt/Dialect/BLIF/BLIF.h.inc"

#endif // CIRCT_DIALECT_BLIF_BLIFOPS_H

using namespace circt;
using namespace blif;
