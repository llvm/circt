//===- SeqPass.h - Seq dialect pass declaration --------------0--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes of the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQPASSES_H
#define CIRCT_DIALECT_SEQ_SEQPASSES_H

namespace circt {
namespace seq {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Seq/SeqPasses.h.inc"

} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQPASSES_H
