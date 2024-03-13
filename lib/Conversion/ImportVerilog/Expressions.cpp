//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/syntax/AllSyntax.h"

using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct ExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  ExprVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Helper function to convert a value to its simple bit vector
  /// representation, if it has one. Otherwise returns null.
  Value convertToSimpleBitVector(Value value) {
    if (!value)
      return {};
    if (auto type = dyn_cast_or_null<moore::UnpackedType>(value.getType())) {
      if (type.isSimpleBitVector())
        return value;
      if (auto sbvt = type.castToSimpleBitVectorOrNull())
        return builder.create<moore::ConversionOp>(
            loc, sbvt.getType(builder.getContext()), value);
    }
    mlir::emitError(loc, "expression of type ")
        << value.getType() << " cannot be cast to a simple bit vector";
    return {};
  }

  /// Helper function to convert a value to its "truthy" boolean value.
  Value convertToBool(Value value) {
    if (!value)
      return {};
    if (auto type = dyn_cast_or_null<moore::IntType>(value.getType()))
      if (type.getBitSize() == 1)
        return value;
    if (auto type = dyn_cast_or_null<moore::UnpackedType>(value.getType()))
      return builder.create<moore::BoolCastOp>(loc, value);
    mlir::emitError(loc, "expression of type ")
        << value.getType() << " cannot be cast to a boolean";
    return {};
  }

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;
    auto d = mlir::emitError(loc, "unknown name `") << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no value generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle type conversions (explicit and implicit).
  Value visit(const slang::ast::ConversionExpression &expr) {
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};
    auto operand = context.convertExpression(expr.operand());
    if (!operand)
      return {};
    return builder.create<moore::ConversionOp>(loc, type, operand);
  }

  // Handle blocking and non-blocking assignments.
  Value visit(const slang::ast::AssignmentExpression &expr) {
    auto lhs = context.convertExpression(expr.left());
    auto rhs = context.convertExpression(expr.right());
    if (!lhs || !rhs)
      return {};

    if (lhs.getType() != rhs.getType())
      rhs = builder.create<moore::ConversionOp>(loc, lhs.getType(), rhs);

    if (expr.timingControl) {
      auto loc = context.convertLocation(expr.timingControl->sourceRange);
      mlir::emitError(loc, "delayed assignments not supported");
      return {};
    }

    if (expr.isNonBlocking())
      builder.create<moore::NonBlockingAssignOp>(loc, lhs, rhs);
    else
      builder.create<moore::BlockingAssignOp>(loc, lhs, rhs);
    return lhs;
  }

  // Helper function to convert an argument to a simple bit vector type, pass it
  // to a reduction op, and optionally invert the result.
  template <class ConcreteOp>
  Value createReduction(Value arg, bool invert) {
    arg = convertToSimpleBitVector(arg);
    if (!arg)
      return {};
    Value result = builder.create<ConcreteOp>(loc, arg);
    if (invert)
      result = builder.create<moore::NotOp>(loc, result);
    return result;
  }

  // Handle unary operators.
  Value visit(const slang::ast::UnaryExpression &expr) {
    auto arg = context.convertExpression(expr.operand());
    if (!arg)
      return {};

    using slang::ast::UnaryOperator;
    switch (expr.op) {
      // `+a` is simply `a`, but converted to a simple bit vector type since
      // this is technically an arithmetic operation.
    case UnaryOperator::Plus:
      return convertToSimpleBitVector(arg);

    case UnaryOperator::Minus:
      arg = convertToSimpleBitVector(arg);
      if (!arg)
        return {};
      return builder.create<moore::NegOp>(loc, arg);

    case UnaryOperator::BitwiseNot:
      arg = convertToSimpleBitVector(arg);
      if (!arg)
        return {};
      return builder.create<moore::NotOp>(loc, arg);

    case UnaryOperator::BitwiseAnd:
      return createReduction<moore::ReduceAndOp>(arg, false);
    case UnaryOperator::BitwiseOr:
      return createReduction<moore::ReduceOrOp>(arg, false);
    case UnaryOperator::BitwiseXor:
      return createReduction<moore::ReduceXorOp>(arg, false);
    case UnaryOperator::BitwiseNand:
      return createReduction<moore::ReduceAndOp>(arg, true);
    case UnaryOperator::BitwiseNor:
      return createReduction<moore::ReduceOrOp>(arg, true);
    case UnaryOperator::BitwiseXnor:
      return createReduction<moore::ReduceXorOp>(arg, true);

    case UnaryOperator::LogicalNot:
      arg = convertToBool(arg);
      if (!arg)
        return {};
      return builder.create<moore::NotOp>(loc, arg);

    case UnaryOperator::Preincrement:
    case UnaryOperator::Predecrement:
    case UnaryOperator::Postincrement:
    case UnaryOperator::Postdecrement:
      break;
    }

    mlir::emitError(loc, "unsupported unary operator");
    return {};
  }

  // Helper function to convert two arguments to a simple bit vector type and
  // pass them into a binary op.
  template <class ConcreteOp>
  Value createBinary(Value lhs, Value rhs) {
    lhs = convertToSimpleBitVector(lhs);
    rhs = convertToSimpleBitVector(rhs);
    if (!lhs || !rhs)
      return {};
    return builder.create<ConcreteOp>(loc, lhs, rhs);
  }

  // Handle binary operators.
  Value visit(const slang::ast::BinaryExpression &expr) {
    auto lhs = context.convertExpression(expr.left());
    auto rhs = context.convertExpression(expr.right());
    if (!lhs || !rhs)
      return {};

    using slang::ast::BinaryOperator;
    switch (expr.op) {
    case BinaryOperator::Add:
      return createBinary<moore::AddOp>(lhs, rhs);
    case BinaryOperator::Subtract:
      return createBinary<moore::SubOp>(lhs, rhs);
    case BinaryOperator::Multiply:
      return createBinary<moore::MulOp>(lhs, rhs);
    case BinaryOperator::Divide:
      return createBinary<moore::DivOp>(lhs, rhs);
    case BinaryOperator::Mod:
      return createBinary<moore::ModOp>(lhs, rhs);

    case BinaryOperator::BinaryAnd:
      return createBinary<moore::AndOp>(lhs, rhs);
    case BinaryOperator::BinaryOr:
      return createBinary<moore::OrOp>(lhs, rhs);
    case BinaryOperator::BinaryXor:
      return createBinary<moore::XorOp>(lhs, rhs);
    case BinaryOperator::BinaryXnor: {
      auto result = createBinary<moore::XorOp>(lhs, rhs);
      if (!result)
        return {};
      return builder.create<moore::NotOp>(loc, result);
    }

    case BinaryOperator::Equality:
      return createBinary<moore::EqOp>(lhs, rhs);
    case BinaryOperator::Inequality:
      return createBinary<moore::NeOp>(lhs, rhs);
    case BinaryOperator::CaseEquality:
      return createBinary<moore::CaseEqOp>(lhs, rhs);
    case BinaryOperator::CaseInequality:
      return createBinary<moore::CaseNeOp>(lhs, rhs);
    case BinaryOperator::WildcardEquality:
      return createBinary<moore::WildcardEqOp>(lhs, rhs);
    case BinaryOperator::WildcardInequality:
      return createBinary<moore::WildcardNeOp>(lhs, rhs);

    case BinaryOperator::GreaterThanEqual:
      return createBinary<moore::GeOp>(lhs, rhs);
    case BinaryOperator::GreaterThan:
      return createBinary<moore::GtOp>(lhs, rhs);
    case BinaryOperator::LessThanEqual:
      return createBinary<moore::LeOp>(lhs, rhs);
    case BinaryOperator::LessThan:
      return createBinary<moore::LtOp>(lhs, rhs);

    // See IEEE 1800-2017 § 11.4.7 "Logical operators".
    case BinaryOperator::LogicalAnd: {
      // TODO: This should short-circuit. Put the RHS code into an scf.if.
      lhs = convertToBool(lhs);
      rhs = convertToBool(rhs);
      if (!lhs || !rhs)
        return {};
      return builder.create<moore::AndOp>(loc, lhs, rhs);
    }
    case BinaryOperator::LogicalOr: {
      // TODO: This should short-circuit. Put the RHS code into an scf.if.
      lhs = convertToBool(lhs);
      rhs = convertToBool(rhs);
      if (!lhs || !rhs)
        return {};
      return builder.create<moore::OrOp>(loc, lhs, rhs);
    }
    case BinaryOperator::LogicalImplication: {
      // `(lhs -> rhs)` equivalent to `(!lhs || rhs)`.
      lhs = convertToBool(lhs);
      rhs = convertToBool(rhs);
      if (!lhs || !rhs)
        return {};
      auto notLHS = builder.create<moore::NotOp>(loc, lhs);
      return builder.create<moore::OrOp>(loc, notLHS, rhs);
    }
    case BinaryOperator::LogicalEquivalence: {
      // `(lhs <-> rhs)` equivalent to `(lhs && rhs) || (!lhs && !rhs)`.
      lhs = convertToBool(lhs);
      rhs = convertToBool(rhs);
      if (!lhs || !rhs)
        return {};
      auto notLHS = builder.create<moore::NotOp>(loc, lhs);
      auto notRHS = builder.create<moore::NotOp>(loc, rhs);
      auto both = builder.create<moore::AndOp>(loc, lhs, rhs);
      auto notBoth = builder.create<moore::AndOp>(loc, notLHS, notRHS);
      return builder.create<moore::OrOp>(loc, both, notBoth);
    }

    case BinaryOperator::LogicalShiftLeft:
      return createBinary<moore::ShlOp>(lhs, rhs);
    case BinaryOperator::LogicalShiftRight:
      return createBinary<moore::ShrOp>(lhs, rhs);
    case BinaryOperator::ArithmeticShiftLeft:
      return createBinary<moore::ShlOp>(lhs, rhs);
    case BinaryOperator::ArithmeticShiftRight: {
      // The `>>>` operator is an arithmetic right shift if the LHS operand is
      // signed, or a logical right shift if the operand is unsigned.
      lhs = convertToSimpleBitVector(lhs);
      rhs = convertToSimpleBitVector(rhs);
      if (!lhs || !rhs)
        return {};
      if (cast<moore::PackedType>(lhs.getType())
              .getSimpleBitVector()
              .isSigned())
        return builder.create<moore::AShrOp>(loc, lhs, rhs);
      return builder.create<moore::ShrOp>(loc, lhs, rhs);
    }

    case BinaryOperator::Power:
      break;
    }

    mlir::emitError(loc, "unsupported binary operator");
    return {};
  }

  // Materialize a Slang integer literal as a constant op.
  Value convertSVInt(const slang::SVInt &value, Type type) {
    if (value.hasUnknown()) {
      mlir::emitError(loc, "literals with X or Z bits not supported");
      return {};
    }
    if (value.getBitWidth() > 64) {
      mlir::emitError(loc, "unsupported bit width: literal is ")
          << value.getBitWidth() << " bits wide; only 64 supported";
      return {};
    }
    auto truncValue = value.as<uint64_t>().value();
    return builder.create<moore::ConstantOp>(loc, type, truncValue);
  }

  // Handle `'0`, `'1`, `'x`, and `'z` literals.
  Value visit(const slang::ast::UnbasedUnsizedIntegerLiteral &expr) {
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};
    return convertSVInt(expr.getValue(), type);
  }

  // Handle integer literals.
  Value visit(const slang::ast::IntegerLiteral &expr) {
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};
    return convertSVInt(expr.getValue(), type);
  }

  // Handle concatenations.
  Value visit(const slang::ast::ConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto *operand : expr.operands()) {
      auto value = context.convertExpression(*operand);
      if (!value)
        return {};
      operands.push_back(value);
    }
    return builder.create<moore::ConcatOp>(loc, operands);
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    mlir::emitError(loc, "unsupported expression: ")
        << slang::ast::toString(node.kind);
    return {};
  }

  Value visitInvalid(const slang::ast::Expression &expr) {
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

Value Context::convertExpression(const slang::ast::Expression &expr) {
  auto loc = convertLocation(expr.sourceRange);
  return expr.visit(ExprVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)
