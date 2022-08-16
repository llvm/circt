//===- EmitCEmissionPatterns.cpp - EmitC Dialect Emission Patterns --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the emitc dialect.
//
//===----------------------------------------------------------------------===//

#include "EmitCEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

using namespace mlir::emitc;
using namespace circt;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit emitc.include operations.
struct IncludeEmitter : OpEmissionPattern<IncludeOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(IncludeOp op, EmissionPrinter &p) override {
    p << "#include " << (op.getIsStandardInclude() ? "<" : "\"")
      << op.getInclude() << (op.getIsStandardInclude() ? ">" : "\"") << "\n";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateEmitCOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<IncludeEmitter>(context);
}
