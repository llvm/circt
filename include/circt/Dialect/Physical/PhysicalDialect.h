//===- PhysicalDialect.h - PD dialect declaration ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Physical MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PHYSICAL_PHYSICALDIALECT_H
#define CIRCT_DIALECT_PHYSICAL_PHYSICALDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace physical {

class PhysicalDialect : public Dialect {
public:
  explicit PhysicalDialect(MLIRContext *context);
  ~PhysicalDialect();

  static StringRef getDialectNamespace() { return "physical"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

} // namespace physical
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQDIALECT_H
