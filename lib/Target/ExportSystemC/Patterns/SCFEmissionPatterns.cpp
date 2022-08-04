//===- SCFEmissionPatterns.cpp - SCF Dialect Emission Patterns ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "SCFEmissionPatterns.h"
#include "../EmissionPattern.h"
#include "../EmissionPrinter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::ExportSystemC;
using namespace mlir::scf;

namespace {
template <typename Op>
struct EmptyEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<Op>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    return success();
  }
};

struct WhileEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<WhileOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto whileOp = cast<WhileOp>(op);
    auto condition = p.getExpression(whileOp.getConditionOp().getCondition());
    if (condition.failed())
      return failure();
    p.emitIndent();
    p << "while (" << condition.getExpressionString() << ") {\n";

    if (failed(p.emitRegion(whileOp.getAfter())))
      return failure();

    p.emitIndent();
    p << "}\n";
    return success();
  }
};
} // namespace

void circt::ExportSystemC::populateSCFEmitters(EmissionPatternSet &patterns) {
  patterns
      .add<EmptyEmitter<ConditionOp>, EmptyEmitter<YieldOp>, WhileEmitter>();
}
