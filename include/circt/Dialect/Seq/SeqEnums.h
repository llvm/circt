//===- SeqPasses.h - Seq pass enumeration -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQENUMS_H
#define CIRCT_DIALECT_SEQ_SEQENUMS_H

namespace circt {
namespace seq {

enum class ReadEnableMode { Zero, Ignore, Undefined };

} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQENUMS_H