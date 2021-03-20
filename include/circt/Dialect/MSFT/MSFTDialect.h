//===- MSFTDialect.h - Microsoft dialect declaration ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MSFT MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTDIALECT_H
#define CIRCT_DIALECT_MSFT_MSFTDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace msft {

class MSFTDialect : public Dialect {
public:
  explicit MSFTDialect(MLIRContext *context);
  ~MSFTDialect();

  static StringRef getDialectNamespace() { return "msft"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_MSFTDIALECT_H
