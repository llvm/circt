//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/syntax/AllSyntax.h"

using namespace circt;
using namespace ImportVerilog;
using moore::Domain;

/// Convert a Slang `SVInt` to a CIRCT `FVInt`.
static FVInt convertSVIntToFVInt(const slang::SVInt &svint) {
  if (svint.hasUnknown()) {
    unsigned numWords = svint.getNumWords() / 2;
    auto value = ArrayRef<uint64_t>(svint.getRawPtr(), numWords);
    auto unknown = ArrayRef<uint64_t>(svint.getRawPtr() + numWords, numWords);
    return FVInt(APInt(svint.getBitWidth(), value),
                 APInt(svint.getBitWidth(), unknown));
  }
  auto value = ArrayRef<uint64_t>(svint.getRawPtr(), svint.getNumWords());
  return FVInt(APInt(svint.getBitWidth(), value));
}

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct RvalueExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  RvalueExprVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Handle references to the left-hand side of a parent assignment.
  Value visit(const slang::ast::LValueReferenceExpression &expr) {
    assert(!context.lvalueStack.empty() && "parent assignments push lvalue");
    auto lvalue = context.lvalueStack.back();
    return builder.create<moore::ReadOp>(loc, lvalue);
  }

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol)) {
      if (isa<moore::RefType>(value.getType())) {
        auto readOp = builder.create<moore::ReadOp>(loc, value);
        if (context.rvalueReadCallback)
          context.rvalueReadCallback(readOp);
        value = readOp.getResult();
      }
      return value;
    }

    // Try to materialize constant values directly.
    auto constant = context.evaluateConstant(expr);
    if (auto value = context.materializeConstant(constant, *expr.type, loc))
      return value;

    // Otherwise some other part of ImportVerilog should have added an MLIR
    // value for this expression's symbol to the `context.valueSymbols` table.
    auto d = mlir::emitError(loc, "unknown name `") << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no rvalue generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle type conversions (explicit and implicit).
  Value visit(const slang::ast::ConversionExpression &expr) {
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};
    return context.convertRvalueExpression(expr.operand(), type);
  }

  // Handle blocking and non-blocking assignments.
  Value visit(const slang::ast::AssignmentExpression &expr) {
    auto lhs = context.convertLvalueExpression(expr.left());
    if (!lhs)
      return {};

    context.lvalueStack.push_back(lhs);
    auto rhs = context.convertRvalueExpression(
        expr.right(), cast<moore::RefType>(lhs.getType()).getNestedType());
    context.lvalueStack.pop_back();
    if (!rhs)
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
    arg = context.convertToSimpleBitVector(arg);
    if (!arg)
      return {};
    Value result = builder.create<ConcreteOp>(loc, arg);
    if (invert)
      result = builder.create<moore::NotOp>(loc, result);
    return result;
  }

  // Helper function to create pre and post increments and decrements.
  Value createIncrement(Value arg, bool isInc, bool isPost) {
    auto preValue = builder.create<moore::ReadOp>(loc, arg);
    auto one = builder.create<moore::ConstantOp>(
        loc, cast<moore::IntType>(preValue.getType()), 1);
    auto postValue =
        isInc ? builder.create<moore::AddOp>(loc, preValue, one).getResult()
              : builder.create<moore::SubOp>(loc, preValue, one).getResult();
    builder.create<moore::BlockingAssignOp>(loc, arg, postValue);
    if (isPost)
      return preValue;
    return postValue;
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
      return context.convertToSimpleBitVector(arg);

    case UnaryOperator::Minus:
      arg = context.convertToSimpleBitVector(arg);
      if (!arg)
        return {};
      return builder.create<moore::NegOp>(loc, arg);

    case UnaryOperator::BitwiseNot:
      arg = context.convertToSimpleBitVector(arg);
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
      arg = context.convertToBool(arg);
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
    lhs = context.convertToSimpleBitVector(lhs);
    if (!lhs)
      return {};
    rhs = context.convertToSimpleBitVector(rhs);
    if (!rhs)
      return {};
    return builder.create<ConcreteOp>(loc, lhs, rhs);
  }

  // Handle binary operators.
  Value visit(const slang::ast::BinaryExpression &expr) {
    auto lhs = context.convertRvalueExpression(expr.left());
    if (!lhs)
      return {};
    auto rhs = context.convertRvalueExpression(expr.right());
    if (!rhs)
      return {};

    // Determine the domain of the result.
    Domain domain = Domain::TwoValued;
    if (expr.type->isFourState() || expr.left().type->isFourState() ||
        expr.right().type->isFourState())
      domain = Domain::FourValued;

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
    case BinaryOperator::Power: {
      // Slang casts the LHS and result of the `**` operator to a four-valued
      // type, since the operator can return X even for two-valued inputs. To
      // maintain uniform types across operands and results, cast the RHS to
      // that four-valued type as well.
      auto rhsCast =
          builder.create<moore::ConversionOp>(loc, lhs.getType(), rhs);
      if (expr.type->isSigned())
        return createBinary<moore::PowSOp>(lhs, rhsCast);
      else
        return createBinary<moore::PowUOp>(lhs, rhsCast);
    }

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
      // TODO: This should short-circuit. Put the RHS code into a separate
      // block.
      lhs = context.convertToBool(lhs, domain);
      if (!lhs)
        return {};
      rhs = context.convertToBool(rhs, domain);
      if (!rhs)
        return {};
      return builder.create<moore::AndOp>(loc, lhs, rhs);
    }
    case BinaryOperator::LogicalOr: {
      // TODO: This should short-circuit. Put the RHS code into a separate
      // block.
      lhs = context.convertToBool(lhs, domain);
      if (!lhs)
        return {};
      rhs = context.convertToBool(rhs, domain);
      if (!rhs)
        return {};
      return builder.create<moore::OrOp>(loc, lhs, rhs);
    }
    case BinaryOperator::LogicalImplication: {
      // `(lhs -> rhs)` equivalent to `(!lhs || rhs)`.
      lhs = context.convertToBool(lhs, domain);
      if (!lhs)
        return {};
      rhs = context.convertToBool(rhs, domain);
      if (!rhs)
        return {};
      auto notLHS = builder.create<moore::NotOp>(loc, lhs);
      return builder.create<moore::OrOp>(loc, notLHS, rhs);
    }
    case BinaryOperator::LogicalEquivalence: {
      // `(lhs <-> rhs)` equivalent to `(lhs && rhs) || (!lhs && !rhs)`.
      lhs = context.convertToBool(lhs, domain);
      if (!lhs)
        return {};
      rhs = context.convertToBool(rhs, domain);
      if (!rhs)
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
      lhs = context.convertToSimpleBitVector(lhs);
      rhs = context.convertToSimpleBitVector(rhs);
      if (!lhs || !rhs)
        return {};
      if (expr.type->isSigned())
        return builder.create<moore::AShrOp>(loc, lhs, rhs);
      return builder.create<moore::ShrOp>(loc, lhs, rhs);
    }
    }

    mlir::emitError(loc, "unsupported binary operator");
    return {};
  }

  // Handle `'0`, `'1`, `'x`, and `'z` literals.
  Value visit(const slang::ast::UnbasedUnsizedIntegerLiteral &expr) {
    return context.materializeSVInt(expr.getValue(), *expr.type, loc);
  }

  // Handle integer literals.
  Value visit(const slang::ast::IntegerLiteral &expr) {
    return context.materializeSVInt(expr.getValue(), *expr.type, loc);
  }

  // Handle concatenations.
  Value visit(const slang::ast::ConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto *operand : expr.operands()) {
      auto value = context.convertRvalueExpression(*operand);
      if (!value)
        continue;
      value = context.convertToSimpleBitVector(value);
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
    if (!type || !value)
      return {};
    if (auto *constValue = expr.selector().constant) {
      assert(!constValue->hasUnknown());
      assert(constValue->size() <= 32);

      auto lowBit = constValue->integer().as<uint32_t>().value();
      return builder.create<moore::ExtractOp>(loc, type, value, lowBit);
    }
    auto lowBit = context.convertRvalueExpression(expr.selector());
    if (!lowBit)
      return {};
    return builder.create<moore::DynExtractOp>(loc, type, value, lowBit);
  }

  // Handle range bits selections.
  Value visit(const slang::ast::RangeSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertRvalueExpression(expr.value());
    if (!type || !value)
      return {};

    Value dynLowBit;
    uint32_t constLowBit;
    auto *leftConst = expr.left().constant;
    auto *rightConst = expr.right().constant;
    if (leftConst) {
      assert(!leftConst->hasUnknown());
      assert(leftConst->size() <= 32);
    }
    if (rightConst) {
      assert(!rightConst->hasUnknown());
      assert(rightConst->size() <= 32);
    }

    if (expr.getSelectionKind() == slang::ast::RangeSelectionKind::Simple) {
      if (leftConst && rightConst) {
        // Estimate whether is big endian or little endian.
        auto lhs = leftConst->integer().as<uint32_t>().value();
        auto rhs = rightConst->integer().as<uint32_t>().value();
        constLowBit = lhs < rhs ? lhs : rhs;
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
      if (leftConst) {
        auto subtrahend = leftConst->integer().as<uint32_t>().value();
        auto sliceWidth =
            expr.right().constant->integer().as<uint32_t>().value();
        constLowBit = subtrahend - sliceWidth - 1;
      } else {
        auto subtrahend = context.convertRvalueExpression(expr.left());
        auto subtrahendType = cast<moore::UnpackedType>(subtrahend.getType());
        auto intType = moore::IntType::get(context.getContext(),
                                           subtrahendType.getBitSize().value(),
                                           subtrahendType.getDomain());
        auto sliceWidth =
            expr.right().constant->integer().as<uint32_t>().value() - 1;
        auto minuend =
            builder.create<moore::ConstantOp>(loc, intType, sliceWidth);
        dynLowBit = builder.create<moore::SubOp>(loc, subtrahend, minuend);
      }
    } else {
      // IndexedUp: arr[0+:8]. "0" is the low bit, "8" is the bits slice width.
      if (leftConst)
        constLowBit = leftConst->integer().as<uint32_t>().value();
      else
        dynLowBit = context.convertRvalueExpression(expr.left());
    }
    if (leftConst && rightConst)
      return builder.create<moore::ExtractOp>(loc, type, value, constLowBit);
    return builder.create<moore::DynExtractOp>(loc, type, value, dynLowBit);
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
    mlir::emitError(loc, "expression of type ")
        << value.getType() << " cannot be accessed";
    return {};
  }

  // Handle set membership operator.
  Value visit(const slang::ast::InsideExpression &expr) {
    auto lhs = context.convertToSimpleBitVector(
        context.convertRvalueExpression(expr.left()));
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
        auto lowBound = context.convertToSimpleBitVector(
            context.convertRvalueExpression(openRange->left()));
        auto highBound = context.convertToSimpleBitVector(
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
        auto value = context.convertToSimpleBitVector(
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
    if (expr.conditions.size() > 1) {
      mlir::emitError(loc)
          << "unsupported conditional expression with more than one condition";
      return {};
    }
    const auto &cond = expr.conditions[0];
    if (cond.pattern) {
      mlir::emitError(loc) << "unsupported conditional expression with pattern";
      return {};
    }
    auto value =
        context.convertToBool(context.convertRvalueExpression(*cond.expr));
    if (!value)
      return {};
    auto conditionalOp = builder.create<moore::ConditionalOp>(loc, type, value);

    // Create blocks for true region and false region.
    auto &trueBlock = conditionalOp.getTrueRegion().emplaceBlock();
    auto &falseBlock = conditionalOp.getFalseRegion().emplaceBlock();

    OpBuilder::InsertionGuard g(builder);

    // Handle left expression.
    builder.setInsertionPointToStart(&trueBlock);
    auto trueValue = context.convertRvalueExpression(expr.left());
    if (!trueValue)
      return {};
    if (trueValue.getType() != type)
      trueValue = builder.create<moore::ConversionOp>(loc, type, trueValue);
    builder.create<moore::YieldOp>(loc, trueValue);

    // Handle right expression.
    builder.setInsertionPointToStart(&falseBlock);
    auto falseValue = context.convertRvalueExpression(expr.right());
    if (!falseValue)
      return {};
    if (falseValue.getType() != type)
      falseValue = builder.create<moore::ConversionOp>(loc, type, falseValue);
    builder.create<moore::YieldOp>(loc, falseValue);

    return conditionalOp.getResult();
  }

  /// Handle calls.
  Value visit(const slang::ast::CallExpression &expr) {
    // Class method calls are currently not supported.
    if (expr.thisClass()) {
      mlir::emitError(loc, "unsupported class method call");
      return {};
    }
    return std::visit(
        [&](auto &subroutine) { return visitCall(expr, subroutine); },
        expr.subroutine);
  }

  /// Handle subroutine calls.
  Value visitCall(const slang::ast::CallExpression &expr,
                  const slang::ast::SubroutineSymbol *subroutine) {
    auto *lowering = context.declareFunction(*subroutine);
    if (!lowering)
      return {};

    // Convert the call arguments. Input arguments are converted to an rvalue.
    // All other arguments are converted to lvalues and passed into the function
    // by reference.
    SmallVector<Value> arguments;
    for (auto [callArg, declArg] :
         llvm::zip(expr.arguments(), subroutine->getArguments())) {

      // Unpack the `<expr> = EmptyArgument` pattern emitted by Slang for output
      // and inout arguments.
      auto *expr = callArg;
      if (const auto *assign = expr->as_if<slang::ast::AssignmentExpression>())
        expr = &assign->left();

      Value value;
      if (declArg->direction == slang::ast::ArgumentDirection::In)
        value = context.convertRvalueExpression(*expr);
      else
        value = context.convertLvalueExpression(*expr);
      if (!value)
        return {};
      arguments.push_back(value);
    }

    // Create the call.
    auto callOp =
        builder.create<mlir::func::CallOp>(loc, lowering->op, arguments);

    // For calls to void functions we need to have a value to return from this
    // function. Create a dummy `unrealized_conversion_cast`, which will get
    // deleted again later on.
    if (callOp.getNumResults() == 0)
      return builder
          .create<mlir::UnrealizedConversionCastOp>(
              loc, moore::VoidType::get(context.getContext()), ValueRange{})
          .getResult(0);

    return callOp.getResult(0);
  }

  /// Handle system calls.
  Value visitCall(const slang::ast::CallExpression &expr,
                  const slang::ast::CallExpression::SystemCallInfo &info) {
    const auto &subroutine = *info.subroutine;
    auto args = expr.arguments();

    if (subroutine.name == "$signed" || subroutine.name == "$unsigned")
      return context.convertRvalueExpression(*args[0]);

    if (subroutine.name == "$clog2") {
      auto value = context.convertToSimpleBitVector(
          context.convertRvalueExpression(*args[0]));
      if (!value)
        return {};
      return builder.create<moore::Clog2BIOp>(loc, value);
    }

    mlir::emitError(loc) << "unsupported system call `" << subroutine.name
                         << "`";
    return {};
  }

  /// Handle string literals.
  Value visit(const slang::ast::StringLiteral &expr) {
    auto type = context.convertType(*expr.type);
    return builder.create<moore::StringConstantOp>(loc, type, expr.getValue());
  }

  /// Handle assignment patterns.
  Value visitAssignmentPattern(
      const slang::ast::AssignmentPatternExpressionBase &expr,
      unsigned replCount = 1) {
    auto type = context.convertType(*expr.type);

    // Convert the individual elements first.
    auto elementCount = expr.elements().size();
    SmallVector<Value> elements;
    elements.reserve(replCount * elementCount);
    for (auto elementExpr : expr.elements()) {
      auto value = context.convertRvalueExpression(*elementExpr);
      if (!value)
        return {};
      elements.push_back(value);
    }
    for (unsigned replIdx = 1; replIdx < replCount; ++replIdx)
      for (unsigned elementIdx = 0; elementIdx < elementCount; ++elementIdx)
        elements.push_back(elements[elementIdx]);

    // Handle packed structs.
    if (auto structType = dyn_cast<moore::StructType>(type)) {
      assert(structType.getMembers().size() == elements.size());
      return builder.create<moore::StructCreateOp>(loc, structType, elements);
    }

    // Handle unpacked structs.
    if (auto structType = dyn_cast<moore::UnpackedStructType>(type)) {
      assert(structType.getMembers().size() == elements.size());
      return builder.create<moore::StructCreateOp>(loc, structType, elements);
    }

    // Handle packed arrays.
    if (auto arrayType = dyn_cast<moore::ArrayType>(type)) {
      assert(arrayType.getSize() == elements.size());
      return builder.create<moore::ArrayCreateOp>(loc, arrayType, elements);
    }

    // Handle unpacked arrays.
    if (auto arrayType = dyn_cast<moore::UnpackedArrayType>(type)) {
      assert(arrayType.getSize() == elements.size());
      return builder.create<moore::ArrayCreateOp>(loc, arrayType, elements);
    }

    mlir::emitError(loc) << "unsupported assignment pattern with type " << type;
    return {};
  }

  Value visit(const slang::ast::SimpleAssignmentPatternExpression &expr) {
    return visitAssignmentPattern(expr);
  }

  Value visit(const slang::ast::StructuredAssignmentPatternExpression &expr) {
    return visitAssignmentPattern(expr);
  }

  Value visit(const slang::ast::ReplicatedAssignmentPatternExpression &expr) {
    auto count =
        context.evaluateConstant(expr.count()).integer().as<unsigned>();
    assert(count && "Slang guarantees constant non-zero replication count");
    return visitAssignmentPattern(expr, *count);
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

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;
    auto d = mlir::emitError(loc, "unknown name `") << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no lvalue generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle concatenations.
  Value visit(const slang::ast::ConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto *operand : expr.operands()) {
      auto value = context.convertLvalueExpression(*operand);
      if (!value)
        continue;
      operands.push_back(value);
    }
    return builder.create<moore::ConcatRefOp>(loc, operands);
  }

  // Handle single bit selections.
  Value visit(const slang::ast::ElementSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertLvalueExpression(expr.value());
    if (!type || !value)
      return {};
    if (auto *constValue = expr.selector().constant) {
      assert(!constValue->hasUnknown());
      assert(constValue->size() <= 32);

      auto lowBit = constValue->integer().as<uint32_t>().value();
      return builder.create<moore::ExtractRefOp>(
          loc, moore::RefType::get(cast<moore::UnpackedType>(type)), value,
          lowBit);
    }
    auto lowBit = context.convertRvalueExpression(expr.selector());
    if (!lowBit)
      return {};
    return builder.create<moore::DynExtractRefOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(type)), value,
        lowBit);
  }

  // Handle range bits selections.
  Value visit(const slang::ast::RangeSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertLvalueExpression(expr.value());
    if (!type || !value)
      return {};

    Value dynLowBit;
    uint32_t constLowBit;
    auto *leftConst = expr.left().constant;
    auto *rightConst = expr.right().constant;
    if (leftConst) {
      assert(!leftConst->hasUnknown());
      assert(leftConst->size() <= 32);
    }
    if (rightConst) {
      assert(!rightConst->hasUnknown());
      assert(rightConst->size() <= 32);
    }

    if (expr.getSelectionKind() == slang::ast::RangeSelectionKind::Simple) {
      if (leftConst && rightConst) {
        // Estimate whether is big endian or little endian.
        auto lhs = leftConst->integer().as<uint32_t>().value();
        auto rhs = rightConst->integer().as<uint32_t>().value();
        constLowBit = lhs < rhs ? lhs : rhs;
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
      if (leftConst) {
        auto subtrahend = leftConst->integer().as<uint32_t>().value();
        auto sliceWidth =
            expr.right().constant->integer().as<uint32_t>().value();
        constLowBit = subtrahend - sliceWidth - 1;
      } else {
        auto subtrahend = context.convertRvalueExpression(expr.left());
        auto subtrahendType = cast<moore::UnpackedType>(subtrahend.getType());
        auto intType = moore::IntType::get(context.getContext(),
                                           subtrahendType.getBitSize().value(),
                                           subtrahendType.getDomain());
        auto sliceWidth =
            expr.right().constant->integer().as<uint32_t>().value() - 1;
        auto minuend =
            builder.create<moore::ConstantOp>(loc, intType, sliceWidth);
        dynLowBit = builder.create<moore::SubOp>(loc, subtrahend, minuend);
      }
    } else {
      // IndexedUp: arr[0+:8]. "0" is the low bit, "8" is the bits slice width.
      if (leftConst)
        constLowBit = leftConst->integer().as<uint32_t>().value();
      else
        dynLowBit = context.convertRvalueExpression(expr.left());
    }
    if (leftConst && rightConst)
      return builder.create<moore::ExtractRefOp>(
          loc, moore::RefType::get(cast<moore::UnpackedType>(type)), value,
          constLowBit);
    return builder.create<moore::DynExtractRefOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(type)), value,
        dynLowBit);
  }

  Value visit(const slang::ast::MemberAccessExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto valueType = expr.value().type;
    auto value = context.convertLvalueExpression(expr.value());
    if (!type || !value)
      return {};
    if (valueType->isStruct()) {
      return builder.create<moore::StructExtractRefOp>(
          loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
          builder.getStringAttr(expr.member.name), value);
    }
    if (valueType->isPackedUnion() || valueType->isUnpackedUnion()) {
      return builder.create<moore::UnionExtractRefOp>(
          loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
          builder.getStringAttr(expr.member.name), value);
    }
    mlir::emitError(loc, "expression of type ")
        << value.getType() << " cannot be accessed";
    return {};
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

Value Context::convertRvalueExpression(const slang::ast::Expression &expr,
                                       Type requiredType) {
  auto loc = convertLocation(expr.sourceRange);
  auto value = expr.visit(RvalueExprVisitor(*this, loc));
  if (value && requiredType && value.getType() != requiredType)
    value = builder.create<moore::ConversionOp>(loc, requiredType, value);
  return value;
}

Value Context::convertLvalueExpression(const slang::ast::Expression &expr) {
  auto loc = convertLocation(expr.sourceRange);
  return expr.visit(LvalueExprVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)

/// Helper function to convert a value to its "truthy" boolean value.
Value Context::convertToBool(Value value) {
  if (!value)
    return {};
  if (auto type = dyn_cast_or_null<moore::IntType>(value.getType()))
    if (type.getBitSize() == 1)
      return value;
  if (auto type = dyn_cast_or_null<moore::UnpackedType>(value.getType()))
    return builder.create<moore::BoolCastOp>(value.getLoc(), value);
  mlir::emitError(value.getLoc(), "expression of type ")
      << value.getType() << " cannot be cast to a boolean";
  return {};
}

/// Materialize a Slang integer literal as a constant op.
Value Context::materializeSVInt(const slang::SVInt &svint,
                                const slang::ast::Type &astType, Location loc) {
  auto type = convertType(astType);
  if (!type)
    return {};

  bool typeIsFourValued = false;
  if (auto unpackedType = dyn_cast<moore::UnpackedType>(type))
    typeIsFourValued = unpackedType.getDomain() == moore::Domain::FourValued;

  auto fvint = convertSVIntToFVInt(svint);
  auto intType = moore::IntType::get(getContext(), fvint.getBitWidth(),
                                     fvint.hasUnknown() || typeIsFourValued
                                         ? moore::Domain::FourValued
                                         : moore::Domain::TwoValued);
  Value result = builder.create<moore::ConstantOp>(loc, intType, fvint);
  if (result.getType() != type)
    result = builder.create<moore::ConversionOp>(loc, type, result);
  return result;
}

Value Context::materializeConstant(const slang::ConstantValue &constant,
                                   const slang::ast::Type &type, Location loc) {
  if (constant.isInteger())
    return materializeSVInt(constant.integer(), type, loc);
  return {};
}

slang::ConstantValue
Context::evaluateConstant(const slang::ast::Expression &expr) {
  using slang::ast::EvalFlags;
  slang::ast::EvalContext evalContext(
      compilation, EvalFlags::CacheResults | EvalFlags::SpecparamsAllowed);
  return expr.eval(evalContext);
}

/// Helper function to convert a value to its "truthy" boolean value and
/// convert it to the given domain.
Value Context::convertToBool(Value value, Domain domain) {
  value = convertToBool(value);
  if (!value)
    return {};
  auto type = moore::IntType::get(getContext(), 1, domain);
  if (value.getType() == type)
    return value;
  return builder.create<moore::ConversionOp>(value.getLoc(), type, value);
}

Value Context::convertToSimpleBitVector(Value value) {
  if (!value)
    return {};
  if (isa<moore::IntType>(value.getType()))
    return value;

  // Some operations in Slang's AST, for example bitwise or `|`, don't cast
  // packed struct/array operands to simple bit vectors but directly operate
  // on the struct/array. Since the corresponding IR ops operate only on
  // simple bit vectors, insert a conversion in this case.
  if (auto packed = dyn_cast<moore::PackedType>(value.getType())) {
    if (auto bits = packed.getBitSize()) {
      auto sbvType =
          moore::IntType::get(value.getContext(), *bits, packed.getDomain());
      return builder.create<moore::ConversionOp>(value.getLoc(), sbvType,
                                                 value);
    }
  }

  mlir::emitError(value.getLoc()) << "expression of type " << value.getType()
                                  << " cannot be cast to a simple bit vector";
  return {};
}
