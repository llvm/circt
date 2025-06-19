//===- AssertionExpr.cpp - Slang assertion expression conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "slang/ast/expressions/AssertionExpr.h"
#include "ImportVerilogInternals.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include <optional>
#include <utility>

using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct AssertionExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  AssertionExprVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Helper to convert a range (min, optional max) to MLIR integer attributes
  std::pair<mlir::IntegerAttr, mlir::IntegerAttr>
  convertRangeToAttrs(uint32_t min,
                      std::optional<uint32_t> max = std::nullopt) {
    auto minAttr = builder.getI64IntegerAttr(min);
    mlir::IntegerAttr rangeAttr;
    if (max.has_value()) {
      rangeAttr = builder.getI64IntegerAttr(max.value() - min);
    }
    return {minAttr, rangeAttr};
  }

  /// Add repetition operation to a sequence
  Value createRepetition(Location loc,
                         const slang::ast::SequenceRepetition &repetition,
                         Value &inputSequence) {
    // Extract cycle range
    auto [minRepetitions, repetitionRange] =
        convertRangeToAttrs(repetition.range.min, repetition.range.max);

    using slang::ast::SequenceRepetition;

    // Check if repetition range is required
    if ((repetition.kind == SequenceRepetition::Nonconsecutive ||
         repetition.kind == SequenceRepetition::GoTo) &&
        !repetitionRange) {
      mlir::emitError(loc,
                      repetition.kind == SequenceRepetition::Nonconsecutive
                          ? "Nonconsecutive repetition requires a maximum value"
                          : "GoTo repetition requires a maximum value");
      return {};
    }

    switch (repetition.kind) {
    case SequenceRepetition::Consecutive:
      return builder.create<ltl::RepeatOp>(loc, inputSequence, minRepetitions,
                                           repetitionRange);
    case SequenceRepetition::Nonconsecutive:
      return builder.create<ltl::NonConsecutiveRepeatOp>(
          loc, inputSequence, minRepetitions, repetitionRange);
    case SequenceRepetition::GoTo:
      return builder.create<ltl::GoToRepeatOp>(loc, inputSequence,
                                               minRepetitions, repetitionRange);
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::SimpleAssertionExpr &expr) {
    // A Simple Assertion expression has the following members:
    // - Expression
    // - Optional repetition

    // The expression needs to have the type `i1` for the LTL operations.
    // Convert first to a moore type and then add a conversion to `i1`.
    auto value = context.convertRvalueExpression(expr.expr);
    auto loc = context.convertLocation(expr.expr.sourceRange);
    value = context.convertToI1(value);
    if (!value)
      return {};

    // The optional repetition is empty, return the converted expression
    if (!expr.repetition.has_value()) {
      return value;
    }

    // There is a repetition, embed the expression into the kind of given
    // repetition
    return createRepetition(loc, expr.repetition.value(), value);
  }

  Value visit(const slang::ast::SequenceConcatExpr &expr) {
    // Create a sequence of delayed operations, combined with a concat operation
    assert(!expr.elements.empty());

    SmallVector<Value> sequenceElements;

    for (const auto &concatElement : expr.elements) {
      Value sequenceValue =
          context.convertAssertionExpression(*concatElement.sequence, loc);
      if (!sequenceValue)
        return {};

      Type valueType = sequenceValue.getType();
      assert(valueType.isInteger(1) || mlir::isa<ltl::SequenceType>(valueType));

      auto [delayMin, delayRange] =
          convertRangeToAttrs(concatElement.delay.min, concatElement.delay.max);
      auto delayedSequence = builder.create<ltl::DelayOp>(loc, sequenceValue,
                                                          delayMin, delayRange);
      sequenceElements.push_back(delayedSequence);
    }

    return builder.createOrFold<ltl::ConcatOp>(loc, sequenceElements);
  }

  Value visit(const slang::ast::UnaryAssertionExpr &expr) {
    auto value = context.convertAssertionExpression(expr.expr, loc);
    if (!value)
      return {};
    using slang::ast::UnaryAssertionOperator;
    switch (expr.op) {
    case UnaryAssertionOperator::Not:
      return builder.create<ltl::NotOp>(loc, value);
    case UnaryAssertionOperator::SEventually:
      if (expr.range.has_value()) {
        mlir::emitError(loc, "Strong eventually with range not supported");
        return {};
      } else {
        return builder.create<ltl::EventuallyOp>(loc, value);
      }
    case UnaryAssertionOperator::Always: {
      std::pair<mlir::IntegerAttr, mlir::IntegerAttr> attr = {
          builder.getI64IntegerAttr(0), mlir::IntegerAttr{}};
      if (expr.range.has_value()) {
        attr =
            convertRangeToAttrs(expr.range.value().min, expr.range.value().max);
      }
      return builder.create<ltl::RepeatOp>(loc, value, attr.first, attr.second);
    }
    case UnaryAssertionOperator::NextTime: {
      auto minRepetitions = builder.getI64IntegerAttr(1);
      if (expr.range.has_value()) {
        minRepetitions = builder.getI64IntegerAttr(expr.range.value().min);
      }
      return builder.create<ltl::DelayOp>(loc, value, minRepetitions,
                                          builder.getI64IntegerAttr(0));
    }
    case UnaryAssertionOperator::Eventually:
    case UnaryAssertionOperator::SNextTime:
    case UnaryAssertionOperator::SAlways:
      mlir::emitError(loc, "unsupported unary operator: ")
          << slang::ast::toString(expr.op);
      return {};
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::BinaryAssertionExpr &expr) {
    auto lhs = context.convertAssertionExpression(expr.left, loc);
    auto rhs = context.convertAssertionExpression(expr.right, loc);
    if (!lhs || !rhs)
      return {};
    SmallVector<Value, 2> operands = {lhs, rhs};
    using slang::ast::BinaryAssertionOperator;
    switch (expr.op) {
    case BinaryAssertionOperator::And:
      return builder.create<ltl::AndOp>(loc, operands);
    case BinaryAssertionOperator::Or:
      return builder.create<ltl::OrOp>(loc, operands);
    case BinaryAssertionOperator::Intersect:
      return builder.create<ltl::IntersectOp>(loc, operands);
    case BinaryAssertionOperator::Throughout: {
      auto lhsRepeat = builder.create<ltl::RepeatOp>(
          loc, lhs, builder.getI64IntegerAttr(0), mlir::IntegerAttr{});
      return builder.create<ltl::IntersectOp>(
          loc, SmallVector<Value, 2>{lhsRepeat, rhs});
    }
    case BinaryAssertionOperator::Within: {
      auto constOne =
          builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
      auto oneRepeat = builder.create<ltl::RepeatOp>(
          loc, constOne, builder.getI64IntegerAttr(0), mlir::IntegerAttr{});
      auto repeatDelay = builder.create<ltl::DelayOp>(
          loc, oneRepeat, builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(0));
      auto lhsDelay = builder.create<ltl::DelayOp>(
          loc, lhs, builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(0));
      auto combined = builder.create<ltl::ConcatOp>(
          loc, SmallVector<Value, 3>{repeatDelay, lhsDelay, constOne});
      return builder.create<ltl::IntersectOp>(
          loc, SmallVector<Value, 2>{combined, rhs});
    }
    case BinaryAssertionOperator::Iff: {
      auto ored = builder.create<ltl::OrOp>(loc, operands);
      auto notOred = builder.create<ltl::NotOp>(loc, ored);
      auto anded = builder.create<ltl::AndOp>(loc, operands);
      return builder.create<ltl::OrOp>(loc,
                                       SmallVector<Value, 2>{notOred, anded});
    }
    case BinaryAssertionOperator::Until:
      return builder.create<ltl::UntilOp>(loc, operands);
    case BinaryAssertionOperator::UntilWith: {
      auto untilOp = builder.create<ltl::UntilOp>(loc, operands);
      auto andOp = builder.create<ltl::AndOp>(loc, operands);
      auto notUntil = builder.create<ltl::NotOp>(loc, untilOp);
      return builder.create<ltl::OrOp>(loc,
                                       SmallVector<Value, 2>{notUntil, andOp});
    }
    case BinaryAssertionOperator::Implies: {
      auto notLhs = builder.create<ltl::NotOp>(loc, lhs);
      return builder.create<ltl::OrOp>(loc, SmallVector<Value, 2>{notLhs, rhs});
    }
    case BinaryAssertionOperator::OverlappedImplication:
      return builder.create<ltl::ImplicationOp>(loc, operands);
    case BinaryAssertionOperator::NonOverlappedImplication: {
      auto constOne =
          builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
      auto lhsDelay = builder.create<ltl::DelayOp>(
          loc, lhs, builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(0));
      auto antecedent = builder.create<ltl::ConcatOp>(
          loc, SmallVector<Value, 2>{lhsDelay, constOne});
      return builder.create<ltl::ImplicationOp>(
          loc, SmallVector<Value, 2>{antecedent, rhs});
    }
    case BinaryAssertionOperator::OverlappedFollowedBy: {
      auto notRhs = builder.create<ltl::NotOp>(loc, rhs);
      auto implication = builder.create<ltl::ImplicationOp>(
          loc, SmallVector<Value, 2>{lhs, notRhs});
      return builder.create<ltl::NotOp>(loc, implication);
    }
    case BinaryAssertionOperator::NonOverlappedFollowedBy: {
      auto constOne =
          builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
      auto notRhs = builder.create<ltl::NotOp>(loc, rhs);
      auto lhsDelay = builder.create<ltl::DelayOp>(
          loc, lhs, builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(0));
      auto antecedent = builder.create<ltl::ConcatOp>(
          loc, SmallVector<Value, 2>{lhsDelay, constOne});
      auto implication = builder.create<ltl::ImplicationOp>(
          loc, SmallVector<Value, 2>{antecedent, notRhs});
      return builder.create<ltl::NotOp>(loc, implication);
    }
    case BinaryAssertionOperator::SUntil:
    case BinaryAssertionOperator::SUntilWith:
      mlir::emitError(loc, "unsupported binary operator: ")
          << slang::ast::toString(expr.op);
      return {};
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::ClockingAssertionExpr &expr) {
    auto assertionExpr = context.convertAssertionExpression(expr.expr, loc);
    if (!assertionExpr)
      return {};
    return context.convertLTLTimingControl(expr.clocking, assertionExpr);
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    mlir::emitError(loc, "unsupported expression: ")
        << slang::ast::toString(node.kind);
    return {};
  }

  Value visitInvalid(const slang::ast::AssertionExpr &expr) {
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

Value Context::convertAssertionExpression(const slang::ast::AssertionExpr &expr,
                                          Location loc) {
  AssertionExprVisitor visitor{*this, loc};
  return expr.visit(visitor);
}
// NOLINTEND(misc-no-recursion)

/// Helper function to convert a value to an i1 value.
Value Context::convertToI1(Value value) {
  if (!value)
    return {};
  auto type = dyn_cast<moore::IntType>(value.getType());
  if (!type || type.getBitSize() != 1) {
    mlir::emitError(value.getLoc(), "expected a 1-bit integer");
    return {};
  }

  return builder.create<moore::ConversionOp>(value.getLoc(),
                                             builder.getI1Type(), value);
}
