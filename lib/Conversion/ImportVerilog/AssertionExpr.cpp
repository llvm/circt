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

  /// Add repetition operation to a sequence
  Value createRepetition(Location loc,
                         const slang::ast::SequenceRepetition &repetition,
                         Value &inputSequence) {
    // Extract cycle range
    // Convert `min`/`max` of type `uint32_t` to the required 64 bit value for
    // `base`/`more`
    auto minRepetitions = mlir::IntegerAttr::get(builder.getIntegerType(64),
                                                 repetition.range.min);
    mlir::IntegerAttr repetitionRange;
    // In repetitions, the max value is optional.
    // The `more` value corresponds to the sequence range.
    if (repetition.range.max.has_value()) {
      repetitionRange = mlir::IntegerAttr::get(builder.getIntegerType(64),
                                               repetition.range.max.value() -
                                                   repetition.range.min);
    }

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
    if (!value) {
      mlir::emitError(loc,
                      "Failed to convert expression in SimpleAssertionExpr");
      return {};
    }
    value = context.convertToI1(value);
    if (!value) {
      mlir::emitError(
          loc,
          "Failed to convert expression to i1 type in SimpleAssertionExpr - "
          "expression may be too wide for boolean conversion");
      return {};
    }

    // The optional repetition is empty, return the converted expression
    if (!expr.repetition.has_value()) {
      return value;
    }

    // There is a repetition, embed the expression into the kind of given
    // repetition
    Value repeatedValue = createRepetition(loc, expr.repetition.value(), value);
    if (!repeatedValue) {
      mlir::emitError(loc,
                      "Failed to create repetition for SimpleAssertionExpr");
      return {};
    }
    return repeatedValue;
  }

  Value visit(const slang::ast::SequenceConcatExpr &expr) {
    // Create a sequence of delayed operations, combined with a concat operation
    if (expr.elements.empty())
      return {};

    SmallVector<Value> sequenceElements;

    for (const auto &concatElement : expr.elements) {
      Value sequenceValue =
          context.convertAssertionExpression(*concatElement.sequence, loc);
      if (!sequenceValue)
        continue;

      Type valueType = sequenceValue.getType();
      if (!valueType.isInteger(1) && !mlir::isa<ltl::SequenceType>(valueType)) {
        mlir::emitError(
            loc,
            "SequenceConcatExpr element must be of type i1 or ltl.sequence");
        return {};
      }

      mlir::IntegerAttr delayRange;
      if (concatElement.delay.max.has_value()) {
        delayRange = mlir::IntegerAttr::get(builder.getIntegerType(64),
                                            concatElement.delay.max.value() -
                                                concatElement.delay.min);
      }
      auto delayedSequence = builder.create<ltl::DelayOp>(
          loc, sequenceValue,
          builder.getI64IntegerAttr(concatElement.delay.min), delayRange);
      sequenceElements.push_back(delayedSequence);
    }

    if (!sequenceElements.empty())
      return builder.createOrFold<ltl::ConcatOp>(loc, sequenceElements);

    return {};
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
    case UnaryAssertionOperator::Always:
      if (expr.range.has_value()) {
        auto minRepetitions = mlir::IntegerAttr::get(builder.getIntegerType(64),
                                                     expr.range.value().min);
        mlir::IntegerAttr repetitionRange;
        if (expr.range.value().max.has_value()) {
          repetitionRange = mlir::IntegerAttr::get(
              builder.getIntegerType(64),
              expr.range.value().max.value() - expr.range.value().min);
        }
        return builder.create<ltl::RepeatOp>(loc, value, minRepetitions,
                                             repetitionRange);
      } else {
        mlir::IntegerAttr repetitionRange;
        return builder.create<ltl::RepeatOp>(
            loc, value, builder.getI64IntegerAttr(0), repetitionRange);
      }
    case UnaryAssertionOperator::NextTime:
      if (expr.range.has_value()) {
        auto minRepetitions = mlir::IntegerAttr::get(builder.getIntegerType(64),
                                                     expr.range.value().min);
        return builder.create<ltl::DelayOp>(loc, value, minRepetitions,
                                            builder.getI64IntegerAttr(0));

      } else {
        return builder.create<ltl::DelayOp>(loc, value,
                                            builder.getI64IntegerAttr(1),
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
      auto lhsDelay =
          builder.create<ltl::DelayOp>(loc, lhs, builder.getI64IntegerAttr(1),
                                       builder.getI64IntegerAttr(0));
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
      return builder.create<ltl::OrOp>(
          loc, SmallVector<Value, 2>{notUntil, andOp});
    }
    case BinaryAssertionOperator::Implies: {
      auto notLhs = builder.create<ltl::NotOp>(loc, lhs);
      return builder.create<ltl::OrOp>(loc,
                                       SmallVector<Value, 2>{notLhs, rhs});
    }
    case BinaryAssertionOperator::OverlappedImplication:
      return builder.create<ltl::ImplicationOp>(loc, operands);
    case BinaryAssertionOperator::NonOverlappedImplication: {
      auto constOne =
          builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
      auto lhsDelay =
          builder.create<ltl::DelayOp>(loc, lhs, builder.getI64IntegerAttr(1),
                                       builder.getI64IntegerAttr(0));
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
      auto lhsDelay =
          builder.create<ltl::DelayOp>(loc, lhs, builder.getI64IntegerAttr(1),
                                       builder.getI64IntegerAttr(0));
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
  if (auto type = dyn_cast_or_null<moore::IntType>(value.getType()))
    if (type.getBitSize() != 1) {
      mlir::emitError(value.getLoc(), "expected a 1-bit integer");
      return {};
    }

  return builder.create<moore::ConversionOp>(value.getLoc(),
                                             builder.getI1Type(), value);
}
