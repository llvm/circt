//===- EmissionPrinter.h - Provides printing utilites to ExportSystemC ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares the EmissionPrinter class to provide printing utilities to
// ExportSystemC.
//
//===----------------------------------------------------------------------===//

#ifndef EMISSION_DRIVER_H
#define EMISSION_DRIVER_H

#include "EmissionPattern.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace ExportSystemC {

class EmissionPrinter {
public:
  EmissionPrinter(raw_ostream &os, EmissionConfig config);

  void emitFileHeader(StringRef filename);
  void emitFileFooter(StringRef filename);

  LogicalResult emitOp(Operation *op);
  EmissionResult getExpression(Value value);
  LogicalResult emitRegion(Region &region, bool increaseIndent = true);
  void emitIndent();

  EmissionPatternSet getEmissionPatternSet() {
    return EmissionPatternSet(patterns);
  }

  EmissionPrinter &operator<<(StringRef str) {
    os << str;
    return *this;
  }

  EmissionPrinter &operator<<(int64_t num) {
    os << std::to_string(num);
    return *this;
  }

  void increaseIndent() { indentLevel++; }
  void decreaseIndent() { indentLevel--; }

private:
  std::vector<std::unique_ptr<EmissionPattern>> patterns;
  EmissionConfig config;
  raw_ostream &os;
  uint32_t indentLevel = 0;
};

} // namespace ExportSystemC
} // namespace circt

#endif // EMISSION_DRIVER_H
