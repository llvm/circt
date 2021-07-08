//===- SeqOps.h - Declare Seq dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQOPS_H
#define CIRCT_DIALECT_SEQ_SEQOPS_H

#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Seq/SeqEnums.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Seq/Seq.h.inc"

#endif // CIRCT_DIALECT_SEQ_SEQOPS_H
