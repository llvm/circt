//===- MSFTOps.h - Microsoft dialect operations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSFT dialect custom operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTOPS_H
#define CIRCT_DIALECT_MSFT_MSFTOPS_H

#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.h.inc"

#endif // CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H
