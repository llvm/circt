//===- CombEmissionPatterns.cpp - Comb Dialect Emission Patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the comb dialect.
//
//===----------------------------------------------------------------------===//

#include "CombEmissionPatterns.h"
#include "EmissionPattern.h"
#include "EmissionPrinter.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::comb;
using namespace circt::ExportSystemC;

static StringRef exprString(Operation *op) {
  return TypeSwitch<Operation *, StringRef>(op)
      .Case<AddOp>([](auto op) { return " + "; })
      .Case<SubOp>([](auto op) { return " - "; })
      .Case<MulOp>([](auto op) { return " * "; })
      .Case<DivUOp>([](auto op) { return " / "; })
      .Case<ShlOp>([](auto op) { return " << "; })
      .Case<ShrUOp>([](auto op) { return " >> "; })
      .Case<ModUOp>([](auto op) { return " % "; })
      .Case<AndOp>([](auto op) { return " & "; })
      .Case<OrOp>([](auto op) { return " | "; })
      .Case<XorOp>([](auto op) { return " ^ "; });
  llvm_unreachable("unsupported operation!");
}

static Precedence exprPrecedence(Operation *op) {
  return TypeSwitch<Operation *, Precedence>(op)
      .Case<AddOp>([](auto op) { return Precedence::ADD; })
      .Case<SubOp>([](auto op) { return Precedence::SUB; })
      .Case<MulOp>([](auto op) { return Precedence::MUL; })
      .Case<DivUOp>([](auto op) { return Precedence::DIV; })
      .Case<ShlOp>([](auto op) { return Precedence::SHL; })
      .Case<ShrUOp>([](auto op) { return Precedence::SHR; })
      .Case<ModUOp>([](auto op) { return Precedence::MOD; })
      .Case<AndOp>([](auto op) { return Precedence::BITWISE_AND; })
      .Case<OrOp>([](auto op) { return Precedence::BITWISE_OR; })
      .Case<XorOp>([](auto op) { return Precedence::BITWISE_XOR; });
  llvm_unreachable("unsupported operation!");
}
namespace {
template <typename Op>
struct VariadicExpressionEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<Op>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    Op op = value.getDefiningOp<Op>();
    std::string expression = "";
    bool first = true;
    for (Value value : op.getInputs()) {
      if (!first)
        expression.append(exprString(op));
      first = false;
      EmissionResult operand = p.getExpression(value);
      bool insertParen = operand.getExpressionPrecedence() > exprPrecedence(op);
      if (insertParen)
        expression.append("(");
      expression.append(operand.getExpressionString());
      if (insertParen)
        expression.append(")");
    }
    return EmissionResult(expression, exprPrecedence(op));
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    return success();
  }
};

template <typename Op>
struct BinaryExpressionEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<Op>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    Op op = value.getDefiningOp<Op>();
    std::string expression = "";

    EmissionResult lhs = p.getExpression(op.getLhs());
    bool lhsParen = lhs.getExpressionPrecedence() > exprPrecedence(op);
    if (lhsParen)
      expression.append("(");
    expression.append(lhs.getExpressionString());
    if (lhsParen)
      expression.append(")");

    expression.append(exprString(op));

    EmissionResult rhs = p.getExpression(op.getRhs());
    bool rhsParen = rhs.getExpressionPrecedence() > exprPrecedence(op);
    if (rhsParen)
      expression.append("(");
    expression.append(rhs.getExpressionString());
    if (rhsParen)
      expression.append(")");

    return EmissionResult(expression, exprPrecedence(op));
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    return success();
  }
};
} // namespace

void circt::ExportSystemC::populateCombEmitters(EmissionPatternSet &patterns) {
  patterns
      .add<VariadicExpressionEmitter<AddOp>, VariadicExpressionEmitter<MulOp>,
           VariadicExpressionEmitter<AndOp>, VariadicExpressionEmitter<OrOp>,
           VariadicExpressionEmitter<XorOp>, BinaryExpressionEmitter<DivUOp>,
           BinaryExpressionEmitter<ModUOp>, BinaryExpressionEmitter<ShlOp>,
           BinaryExpressionEmitter<ShrUOp>, BinaryExpressionEmitter<SubOp>>();
}
