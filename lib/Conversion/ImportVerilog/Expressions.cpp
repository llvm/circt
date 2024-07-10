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
struct RvalueExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  RvalueExprVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Helper function to convert a value to its simple bit vector
  /// representation, if it has one. Otherwise returns null.
  Value convertToSimpleBitVector(Value value) {
    if (!value)
      return {};
    if (isa<moore::IntType>(value.getType()) ||
        isa<moore::IntType>(
            dyn_cast<moore::RefType>(value.getType()).getNestedType()))
      return value;
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

  // Handle references to the left-hand side of a parent assignment.
  Value visit(const slang::ast::LValueReferenceExpression &expr) {
    assert(!context.lvalueStack.empty() && "parent assignments push lvalue");
    auto lvalue = context.lvalueStack.back();
    return builder.create<moore::ReadOp>(
        loc, cast<moore::RefType>(lvalue.getType()).getNestedType(), lvalue);
  }

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return isa<moore::NamedConstantOp>(value.getDefiningOp())
                 ? value
                 : builder.create<moore::ReadOp>(
                       loc,
                       cast<moore::RefType>(value.getType()).getNestedType(),
                       value);
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
    auto operand = context.convertRvalueExpression(expr.operand());
    if (!operand)
      return {};
    return builder.create<moore::ConversionOp>(loc, type, operand);
  }

  // Handle blocking and non-blocking assignments.
  Value visit(const slang::ast::AssignmentExpression &expr) {
    auto lhs = context.convertLvalueExpression(expr.left());
    context.lvalueStack.push_back(lhs);
    auto rhs = context.convertRvalueExpression(expr.right());
    context.lvalueStack.pop_back();
    if (!lhs || !rhs)
      return {};

    if (expr.timingControl) {
      auto loc = context.convertLocation(expr.timingControl->sourceRange);
      mlir::emitError(loc, "delayed assignments not supported");
      return {};
    }

    if (expr.isNonBlocking())
      builder.create<moore::NonBlockingAssignOp>(loc, lhs, rhs);
    else
      builder.create<moore::BlockingAssignOp>(loc, lhs, rhs);
    return rhs;
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

  // Helper function to create pre and post increments and decrements.
  Value createIncrement(Value arg, bool isInc, bool isPost) {
    auto preValue = convertToSimpleBitVector(arg);
    if (!preValue)
      return {};
    preValue = builder.create<moore::ReadOp>(
        loc, cast<moore::RefType>(preValue.getType()).getNestedType(),
        preValue);
    auto one = builder.create<moore::ConstantOp>(
        loc, cast<moore::IntType>(preValue.getType()), 1);
    auto postValue =
        isInc ? builder.create<moore::AddOp>(loc, preValue, one).getResult()
              : builder.create<moore::SubOp>(loc, preValue, one).getResult();
    builder.create<moore::BlockingAssignOp>(loc, arg, postValue);
    return isPost ? preValue : postValue;
  }

  // Handle unary operators.
  Value visit(const slang::ast::UnaryExpression &expr) {
    using slang::ast::UnaryOperator;
    Value arg;
    if (expr.op == UnaryOperator::Preincrement ||
        expr.op == UnaryOperator::Predecrement ||
        expr.op == UnaryOperator::Postincrement ||
        expr.op == UnaryOperator::Postdecrement)
      arg = context.convertLvalueExpression(expr.operand());
    else
      arg = context.convertRvalueExpression(expr.operand());
    if (!arg)
      return {};

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
      return createIncrement(arg, true, false);
    case UnaryOperator::Predecrement:
      return createIncrement(arg, false, false);
    case UnaryOperator::Postincrement:
      return createIncrement(arg, true, true);
    case UnaryOperator::Postdecrement:
      return createIncrement(arg, false, true);
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
    auto lhs = context.convertRvalueExpression(expr.left());
    auto rhs = context.convertRvalueExpression(expr.right());
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
      if (expr.type->isSigned())
        return createBinary<moore::DivSOp>(lhs, rhs);
      else
        return createBinary<moore::DivUOp>(lhs, rhs);
    case BinaryOperator::Mod:
      if (expr.type->isSigned())
        return createBinary<moore::ModSOp>(lhs, rhs);
      else
        return createBinary<moore::ModUOp>(lhs, rhs);

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
      if (expr.left().type->isSigned())
        return createBinary<moore::SgeOp>(lhs, rhs);
      else
        return createBinary<moore::UgeOp>(lhs, rhs);
    case BinaryOperator::GreaterThan:
      if (expr.left().type->isSigned())
        return createBinary<moore::SgtOp>(lhs, rhs);
      else
        return createBinary<moore::UgtOp>(lhs, rhs);
    case BinaryOperator::LessThanEqual:
      if (expr.left().type->isSigned())
        return createBinary<moore::SleOp>(lhs, rhs);
      else
        return createBinary<moore::UleOp>(lhs, rhs);
    case BinaryOperator::LessThan:
      if (expr.left().type->isSigned())
        return createBinary<moore::SltOp>(lhs, rhs);
      else
        return createBinary<moore::UltOp>(lhs, rhs);

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
      if (expr.type->isSigned())
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
    return builder.create<moore::ConstantOp>(loc, cast<moore::IntType>(type),
                                             truncValue);
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
      auto value = context.convertRvalueExpression(*operand);
      if (!value)
        continue;
      value = convertToSimpleBitVector(value);
      operands.push_back(value);
    }
    return builder.create<moore::ConcatOp>(loc, operands);
  }

  // Handle replications.
  Value visit(const slang::ast::ReplicationExpression &expr) {
    auto type = context.convertType(*expr.type);
    if (isa<moore::VoidType>(type))
      return {};

    auto value = context.convertRvalueExpression(expr.concat());
    if (!value)
      return {};
    return builder.create<moore::ReplicateOp>(loc, type, value);
  }

  // Handle single bit selections.
  Value visit(const slang::ast::ElementSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertRvalueExpression(expr.value());
    auto lowBit = context.convertRvalueExpression(expr.selector());

    if (!type || !value || !lowBit)
      return {};
    return builder.create<moore::ExtractOp>(loc, type, value, lowBit);
  }

  // Handle range bits selections.
  Value visit(const slang::ast::RangeSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertRvalueExpression(expr.value());
    Value lowBit;
    if (expr.getSelectionKind() == slang::ast::RangeSelectionKind::Simple) {
      if (expr.left().constant && expr.right().constant) {
        auto lhs = expr.left().constant->integer().as<uint64_t>().value();
        auto rhs = expr.right().constant->integer().as<uint64_t>().value();
        lowBit = lhs < rhs ? context.convertRvalueExpression(expr.left())
                           : context.convertRvalueExpression(expr.right());
      } else {
        mlir::emitError(loc, "unsupported a variable as the index in the")
            << slang::ast::toString(expr.getSelectionKind()) << "kind";
        return {};
      }
    } else if (expr.getSelectionKind() ==
               slang::ast::RangeSelectionKind::IndexedDown) {
      // IndexedDown: arr[7-:8]. It's equivalent to arr[7:0] or arr[0:7]
      // depending on little endian or bit endian. No matter which situation,
      // the low bit must be "0".
      auto minuend = context.convertRvalueExpression(expr.left());
      auto minuendType = cast<moore::UnpackedType>(minuend.getType());
      auto intType = moore::IntType::get(context.getContext(),
                                         minuendType.getBitSize().value(),
                                         minuendType.getDomain());
      auto sliceWidth =
          expr.right().constant->integer().as<uint64_t>().value() - 1;
      auto subtraction =
          builder.create<moore::ConstantOp>(loc, intType, sliceWidth);
      lowBit = builder.create<moore::SubOp>(loc, minuend, subtraction);
    } else
      // IndexedUp: arr[0+:8]. "0" is the low bit, "8" is the bits slice width.
      lowBit = context.convertRvalueExpression(expr.left());

    if (!type || !value || !lowBit)
      return {};
    return builder.create<moore::ExtractOp>(loc, type, value, lowBit);
  }

  Value visit(const slang::ast::MemberAccessExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto valueType = expr.value().type;
    auto value = context.convertRvalueExpression(expr.value());
    if (!type || !value)
      return {};
    if (valueType->isStruct()) {
      return builder.create<moore::StructExtractOp>(
          loc, type, builder.getStringAttr(expr.member.name), value);
    }
    if (valueType->isPackedUnion() || valueType->isUnpackedUnion()) {
      return builder.create<moore::UnionExtractOp>(
          loc, type, builder.getStringAttr(expr.member.name), value);
    }
    llvm_unreachable("unsupported symbol kind");
  }

  // Handle set membership operator.
  Value visit(const slang::ast::InsideExpression &expr) {
    auto lhs =
        convertToSimpleBitVector(context.convertRvalueExpression(expr.left()));
    if (!lhs)
      return {};
    // All conditions for determining whether it is inside.
    SmallVector<Value> conditions;

    // Traverse open range list.
    for (const auto *listExpr : expr.rangeList()) {
      Value cond;
      // The open range list on the right-hand side of the inside operator is a
      // comma-separated list of expressions or ranges.
      if (const auto *openRange =
              listExpr->as_if<slang::ast::OpenRangeExpression>()) {
        // Handle ranges.
        auto lowBound = convertToSimpleBitVector(
            context.convertRvalueExpression(openRange->left()));
        auto highBound = convertToSimpleBitVector(
            context.convertRvalueExpression(openRange->right()));
        if (!lowBound || !highBound)
          return {};
        Value leftValue, rightValue;
        // Determine if the expression on the left-hand side is inclusively
        // within the range.
        if (openRange->left().type->isSigned() ||
            expr.left().type->isSigned()) {
          leftValue = builder.create<moore::SgeOp>(loc, lhs, lowBound);
        } else {
          leftValue = builder.create<moore::UgeOp>(loc, lhs, lowBound);
        }
        if (openRange->right().type->isSigned() ||
            expr.left().type->isSigned()) {
          rightValue = builder.create<moore::SleOp>(loc, lhs, highBound);
        } else {
          rightValue = builder.create<moore::UleOp>(loc, lhs, highBound);
        }
        cond = builder.create<moore::AndOp>(loc, leftValue, rightValue);
      } else {
        // Handle expressions.
        if (!listExpr->type->isSimpleBitVector()) {
          if (listExpr->type->isUnpackedArray()) {
            mlir::emitError(
                loc, "unpacked arrays in 'inside' expressions not supported");
            return {};
          }
          mlir::emitError(
              loc, "only simple bit vectors supported in 'inside' expressions");
          return {};
        }
        auto value = convertToSimpleBitVector(
            context.convertRvalueExpression(*listExpr));
        if (!value)
          return {};
        cond = builder.create<moore::WildcardEqOp>(loc, lhs, value);
      }
      conditions.push_back(cond);
    }

    // Calculate the final result by `or` op.
    auto result = conditions.back();
    conditions.pop_back();
    while (!conditions.empty()) {
      result = builder.create<moore::OrOp>(loc, conditions.back(), result);
      conditions.pop_back();
    }
    return result;
  }

  // Handle conditional operator `?:`.
  Value visit(const slang::ast::ConditionalExpression &expr) {
    auto type = context.convertType(*expr.type);

    // Handle condition.
    Value cond = convertToSimpleBitVector(
        context.convertRvalueExpression(*expr.conditions.begin()->expr));
    cond = convertToBool(cond);
    if (!cond)
      return {};
    auto conditionalOp = builder.create<moore::ConditionalOp>(loc, type, cond);

    // Create blocks for true region and false region.
    conditionalOp.getTrueRegion().emplaceBlock();
    conditionalOp.getFalseRegion().emplaceBlock();

    OpBuilder::InsertionGuard g(builder);

    // Handle left expression.
    builder.setInsertionPointToStart(conditionalOp.getBody(0));
    auto trueValue = context.convertRvalueExpression(expr.left());
    if (!trueValue)
      return {};
    builder.create<moore::YieldOp>(loc, trueValue);

    // Handle right expression.
    builder.setInsertionPointToStart(conditionalOp.getBody(1));
    auto falseValue = context.convertRvalueExpression(expr.right());
    if (!falseValue)
      return {};
    builder.create<moore::YieldOp>(loc, falseValue);

    return conditionalOp.getResult();
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

namespace {
struct LvalueExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  LvalueExprVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Helper function to convert a value to its simple bit vector
  /// representation, if it has one. Otherwise returns null.
  Value convertToSimpleBitVector(Value value) {
    if (!value)
      return {};
    if (isa<moore::IntType>(
            cast<moore::RefType>(value.getType()).getNestedType()))
      return value;
    mlir::emitError(loc, "expression of type ")
        << value.getType() << " cannot be cast to a simple bit vector";
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

  // Handle concatenations.
  Value visit(const slang::ast::ConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto *operand : expr.operands()) {
      auto value = context.convertLvalueExpression(*operand);
      if (!value)
        continue;
      value = convertToSimpleBitVector(value);
      operands.push_back(value);
    }
    return builder.create<moore::ConcatRefOp>(loc, operands);
  }

  // Handle single bit selections.
  Value visit(const slang::ast::ElementSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertLvalueExpression(expr.value());
    auto lowBit = context.convertRvalueExpression(expr.selector());

    if (!type || !value || !lowBit)
      return {};
    return builder.create<moore::ExtractRefOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(type)), value,
        lowBit);
  }

  // Handle range bits selections.
  Value visit(const slang::ast::RangeSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertLvalueExpression(expr.value());
    Value lowBit;
    if (expr.getSelectionKind() == slang::ast::RangeSelectionKind::Simple) {
      if (expr.left().constant && expr.right().constant) {
        auto lhs = expr.left().constant->integer().as<uint64_t>().value();
        auto rhs = expr.right().constant->integer().as<uint64_t>().value();
        lowBit = lhs < rhs ? context.convertRvalueExpression(expr.left())
                           : context.convertRvalueExpression(expr.right());
      } else {
        mlir::emitError(loc, "unsupported a variable as the index in the")
            << slang::ast::toString(expr.getSelectionKind()) << "kind";
        return {};
      }
    } else if (expr.getSelectionKind() ==
               slang::ast::RangeSelectionKind::IndexedDown) {
      // IndexedDown: arr[7-:8]. It's equivalent to arr[7:0] or arr[0:7]
      // depending on little endian or bit endian. No matter which situation,
      // the low bit must be "0".
      auto minuend = context.convertRvalueExpression(expr.left());
      auto minuendType = cast<moore::UnpackedType>(minuend.getType());
      auto intType = moore::IntType::get(context.getContext(),
                                         minuendType.getBitSize().value(),
                                         minuendType.getDomain());
      auto sliceWidth =
          expr.right().constant->integer().as<uint64_t>().value() - 1;
      auto subtraction =
          builder.create<moore::ConstantOp>(loc, intType, sliceWidth);
      lowBit = builder.create<moore::SubOp>(loc, minuend, subtraction);
    } else
      // IndexedUp: arr[0+:8]. "0" is the low bit, "8" is the bits slice width.
      lowBit = context.convertRvalueExpression(expr.left());

    if (!type || !value || !lowBit)
      return {};
    return builder.create<moore::ExtractRefOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(type)), value,
        lowBit);
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    return context.convertRvalueExpression(node);
  }

  Value visitInvalid(const slang::ast::Expression &expr) {
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

Value Context::convertRvalueExpression(const slang::ast::Expression &expr) {
  auto loc = convertLocation(expr.sourceRange);
  return expr.visit(RvalueExprVisitor(*this, loc));
}

Value Context::convertLvalueExpression(const slang::ast::Expression &expr) {
  auto loc = convertLocation(expr.sourceRange);
  return expr.visit(LvalueExprVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)
