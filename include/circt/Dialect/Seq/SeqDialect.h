//===- SeqDialect.h - Seq dialect declaration -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an Seq MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQDIALECT_H
#define CIRCT_DIALECT_SEQ_SEQDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace seq {

class SeqDialect : public Dialect {
public:
  explicit SeqDialect(MLIRContext *context);
  ~SeqDialect();

  static StringRef getDialectNamespace() { return "seq"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQDIALECT_H
