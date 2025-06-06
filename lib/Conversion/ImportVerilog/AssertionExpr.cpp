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
