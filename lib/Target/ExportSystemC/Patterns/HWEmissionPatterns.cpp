//===- HWEmissionPatterns.cpp - HW Dialect Emission Patterns --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the HW dialect.
//
//===----------------------------------------------------------------------===//

#include "HWEmissionPatterns.h"
#include "../EmissionPattern.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::ExportSystemC;

namespace {
struct ConstantEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<ConstantOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    auto op = value.getDefiningOp<ConstantOp>();

    if (op.getValue().getBitWidth() == 1) {
      if (op.getValue().getBoolValue())
        return EmissionResult("true", Precedence::LIT);

      return EmissionResult("false", Precedence::LIT);
    }

    SmallString<32> valueString;
    op.getValue().toStringUnsigned(valueString);
    return EmissionResult(valueString, Precedence::LIT);
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    return success();
  }
};
} // namespace

void circt::ExportSystemC::populateHWEmitters(EmissionPatternSet &patterns) {
  patterns.add<ConstantEmitter>();
}
