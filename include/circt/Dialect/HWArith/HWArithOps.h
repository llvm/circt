//===- HWArithOps.h - Declare HWArith dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation class for the HWArith IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HWARITH_OPS_H
#define CIRCT_DIALECT_HWARITH_OPS_H

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "circt/Dialect/HWArith/HWArith.h.inc"

#endif // CIRCT_DIALECT_HWARITH_OPS_H
