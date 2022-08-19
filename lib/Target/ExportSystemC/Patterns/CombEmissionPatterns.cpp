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
#include "../EmissionPattern.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::comb;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Map a comb operation to the corresponding operator string in C++.
static StringRef getExprString(Operation *op) {
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
}

/// Map a comb operation to the corresponding operator precedence in C++.
static Precedence getExprPrecedence(Operation *op) {
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
}

/// A convenience method that, if @param addParens is true, emits parentheses
/// before and after the expression emitted by the passed @param emitter. The
/// emitter is triggered independent of @param addParens.
static void parenthesize(bool addParens, const InlineEmitter &emitter,
                         EmissionPrinter &p) {
  if (addParens)
    p << "(";

  emitter.emit();

  if (addParens)
    p << ")";
}

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

///
namespace {
template <typename Op, bool castResult = false>
struct VariadicExpressionEmitter : OpEmissionPattern<Op> {
  explicit VariadicExpressionEmitter(MLIRContext *context)
      : OpEmissionPattern<Op>(context) {}

  MatchResult matchInlinable(Value value) override {
    // Bitwidth's bigger than 512 are emitted as bit-vectors which do not
    // directly support arithmetic operations, but are limited to
    if (value.getType().getIntOrFloatBitWidth() > 512)
      return {};

    if (auto op = value.getDefiningOp<Op>())
      return castResult ? Precedence::FUNCTIONAL_CAST : getExprPrecedence(op);
    return {};
  }
  void emitInlined(Value value, EmissionPrinter &p) override {
    Op op = value.getDefiningOp<Op>();

    if (castResult) {
      p.emitType(value.getType());
      p << "(";
    }

    bool first = true;
    for (Value value : op.getInputs()) {
      if (!first)
        p << getExprString(op);
      first = false;
      InlineEmitter operand = p.getInlinable(value);
      parenthesize(operand.getPrecedence() >= getExprPrecedence(op), operand,
                   p);
    }

    if (castResult)
      p << ")";
  }
};

///
template <typename Op, bool castResult = false>
struct BinaryExpressionEmitter : OpEmissionPattern<Op> {
  explicit BinaryExpressionEmitter(MLIRContext *context)
      : OpEmissionPattern<Op>(context) {}

  MatchResult matchInlinable(Value value) override {
    if (value.getType().getIntOrFloatBitWidth() > 512)
      return {};

    if (auto op = value.getDefiningOp<Op>())
      return castResult ? Precedence::FUNCTIONAL_CAST : getExprPrecedence(op);
    return {};
  }
  void emitInlined(Value value, EmissionPrinter &p) override {
    Op op = value.getDefiningOp<Op>();

    if (castResult) {
      p.emitType(value.getType());
      p << "(";
    }

    InlineEmitter lhs = p.getInlinable(op.getLhs());
    parenthesize(lhs.getPrecedence() >= getExprPrecedence(op), lhs, p);

    p << getExprString(op);

    InlineEmitter rhs = p.getInlinable(op.getRhs());
    parenthesize(rhs.getPrecedence() >= getExprPrecedence(op), rhs, p);

    if (castResult)
      p << ")";
  }
};

///
struct ConcatEmitter : OpEmissionPattern<ConcatOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ConcatOp>())
      return Precedence::FUNCTION_CALL;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    ConcatOp op = value.getDefiningOp<ConcatOp>();

    p.emitType(value.getType());
    p << "(";

    if (op.getInputs().size() == 1) {
      p.getInlinable(op.getInputs()[0]).emit();
      return;
    }

    for (size_t i = 0; i < op.getInputs().size() - 1; ++i) {
      p << "sc_dt::concat(";
      p.getInlinable(op.getInputs()[i]).emit();
      p << ", ";
    }

    p.getInlinable(op.getInputs()[op.getInputs().size() - 1]).emit();

    for (size_t i = 0; i < op.getInputs().size() - 1; ++i)
      p << ")";

    p << ")";
  }
};

///
struct ExtractEmitter : OpEmissionPattern<ExtractOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ExtractOp>())
      return Precedence::FUNCTION_CALL;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    ExtractOp op = value.getDefiningOp<ExtractOp>();

    p.emitType(value.getType());
    p << "(";

    InlineEmitter input = p.getInlinable(op.getInput());
    parenthesize(input.getPrecedence() > Precedence::VAR, input, p);

    p << ".range(" << op.getLowBit() << ", "
      << (op.getLowBit() + op.getResult().getType().getIntOrFloatBitWidth() - 1)
      << "))";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateCombOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns
      .add<VariadicExpressionEmitter<AddOp>, VariadicExpressionEmitter<MulOp>,
           VariadicExpressionEmitter<AndOp>, VariadicExpressionEmitter<OrOp>,
           VariadicExpressionEmitter<XorOp>, BinaryExpressionEmitter<DivUOp>,
           BinaryExpressionEmitter<ModUOp>, BinaryExpressionEmitter<ShlOp>,
           BinaryExpressionEmitter<ShrUOp>,
           BinaryExpressionEmitter<SubOp, true>>(context);
  patterns.add<ConcatEmitter, ExtractEmitter>(context);
}
