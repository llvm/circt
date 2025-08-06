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

/// Map an index into an array, with bounds `range`, to a bit offset of the
/// underlying bit storage. This is a dynamic version of
/// `slang::ConstantRange::translateIndex`.
static Value getSelectIndex(OpBuilder &builder, Location loc, Value index,
                            const slang::ConstantRange &range) {
  auto indexType = cast<moore::UnpackedType>(index.getType());
  auto bw = std::max(llvm::Log2_32_Ceil(std::max(std::abs(range.lower()),
                                                 std::abs(range.upper()))),
                     indexType.getBitSize().value());
  auto intType =
      moore::IntType::get(index.getContext(), bw, indexType.getDomain());

  if (range.isLittleEndian()) {
    if (range.lower() == 0)
      return index;

    Value newIndex =
        builder.createOrFold<moore::ConversionOp>(loc, intType, index);
    Value offset =
        moore::ConstantOp::create(builder, loc, intType, range.lower(),
                                  /*isSigned = */ range.lower() < 0);
    return builder.createOrFold<moore::SubOp>(loc, newIndex, offset);
  }

  if (range.upper() == 0)
    return builder.createOrFold<moore::NegOp>(loc, index);

  Value newIndex =
      builder.createOrFold<moore::ConversionOp>(loc, intType, index);
  Value offset = moore::ConstantOp::create(builder, loc, intType, range.upper(),
                                           /* isSigned = */ range.upper() < 0);
  return builder.createOrFold<moore::SubOp>(loc, offset, newIndex);
}

namespace {
/// A visitor handling expressions that can be lowered as lvalue and rvalue.
struct ExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;
  bool isLvalue;

  ExprVisitor(Context &context, Location loc, bool isLvalue)
      : context(context), loc(loc), builder(context.builder),
        isLvalue(isLvalue) {}

  /// Convert an expression either as an lvalue or rvalue, depending on whether
  /// this is an lvalue or rvalue visitor. This is useful for projections such
  /// as `a[i]`, where you want `a` as an lvalue if you want `a[i]` as an
  /// lvalue, or `a` as an rvalue if you want `a[i]` as an rvalue.
  Value convertLvalueOrRvalueExpression(const slang::ast::Expression &expr) {
    if (isLvalue)
      return context.convertLvalueExpression(expr);
    return context.convertRvalueExpression(expr);
  }

  /// Handle single bit selections.
  Value visit(const slang::ast::ElementSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = convertLvalueOrRvalueExpression(expr.value());
    if (!type || !value)
      return {};
    auto resultType =
        isLvalue ? moore::RefType::get(cast<moore::UnpackedType>(type)) : type;
    auto range = expr.value().type->getFixedRange();
    if (auto *constValue = expr.selector().constant) {
      assert(!constValue->hasUnknown());
      assert(constValue->size() <= 32);

      auto lowBit = constValue->integer().as<uint32_t>().value();
      if (isLvalue)
        return moore::ExtractRefOp::create(builder, loc, resultType, value,
                                           range.translateIndex(lowBit));
      else
        return moore::ExtractOp::create(builder, loc, resultType, value,
                                        range.translateIndex(lowBit));
    }
    auto lowBit = context.convertRvalueExpression(expr.selector());
    if (!lowBit)
      return {};
    lowBit = getSelectIndex(builder, loc, lowBit, range);
    if (isLvalue)
      return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                            lowBit);
    else
      return moore::DynExtractOp::create(builder, loc, resultType, value,
                                         lowBit);
  }

  /// Handle range bit selections.
  Value visit(const slang::ast::RangeSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = convertLvalueOrRvalueExpression(expr.value());
    if (!type || !value)
      return {};

    std::optional<int32_t> constLeft;
    std::optional<int32_t> constRight;
    if (auto *constant = expr.left().constant)
      constLeft = constant->integer().as<int32_t>();
    if (auto *constant = expr.right().constant)
      constRight = constant->integer().as<int32_t>();

    // We need to determine the right bound of the range. This is the address of
    // the least significant bit of the underlying bit storage, which is the
    // offset we want to pass to the extract op.
    //
    // The arrays [6:2] and [2:6] both have 5 bits worth of underlying storage.
    // The left and right bound of the range only determine the addressing
    // scheme of the storage bits:
    //
    // Storage bits:   4  3  2  1  0  <-- extract op works on storage bits
    // [6:2] indices:  6  5  4  3  2  ("little endian" in Slang terms)
    // [2:6] indices:  2  3  4  5  6  ("big endian" in Slang terms)
    //
    // Before we can extract, we need to map the range select left and right
    // bounds from these indices to actual bit positions in the storage.

    Value offsetDyn;
    int32_t offsetConst = 0;
    auto range = expr.value().type->getFixedRange();

    using slang::ast::RangeSelectionKind;
    if (expr.getSelectionKind() == RangeSelectionKind::Simple) {
      // For a constant range [a:b], we want the offset of the lowest storage
      // bit from which we are starting the extract. For a range [5:3] this is
      // bit index 3; for a range [3:5] this is bit index 5. Both of these are
      // later translated map to bit offset 1 (see bit indices above).
      assert(constRight && "constness checked in slang");
      offsetConst = *constRight;
    } else {
      // For an indexed range [a+:b] or [a-:b], determining the lowest storage
      // bit is a bit more complicated. We start out with the base index `a`.
      // This is the lower *index* of the range, but not the lower *storage bit
      // position*.
      //
      // The range [a+:b] expands to [a+b-1:a] for a [6:2] range, or [a:a+b-1]
      // for a [2:6] range. The range [a-:b] expands to [a:a-b+1] for a [6:2]
      // range, or [a-b+1:a] for a [2:6] range.
      if (constLeft) {
        offsetConst = *constLeft;
      } else {
        offsetDyn = context.convertRvalueExpression(expr.left());
        if (!offsetDyn)
          return {};
      }

      // For a [a-:b] select on [2:6] and a [a+:b] select on [6:2], the range
      // expands to [a-b+1:a] and [a+b-1:a]. In this case, the right bound which
      // corresponds to the lower *storage bit offset*, is just `a` and there's
      // no further tweaking to do.
      int32_t offsetAdd = 0;

      // For a [a-:b] select on [6:2], the range expands to [a:a-b+1]. We
      // therefore have to take the `a` from above and adjust it by `-b+1` to
      // arrive at the right bound.
      if (expr.getSelectionKind() == RangeSelectionKind::IndexedDown &&
          range.isLittleEndian()) {
        assert(constRight && "constness checked in slang");
        offsetAdd = 1 - *constRight;
      }

      // For a [a+:b] select on [2:6], the range expands to [a:a+b-1]. We
      // therefore have to take the `a` from above and adjust it by `+b-1` to
      // arrive at the right bound.
      if (expr.getSelectionKind() == RangeSelectionKind::IndexedUp &&
          !range.isLittleEndian()) {
        assert(constRight && "constness checked in slang");
        offsetAdd = *constRight - 1;
      }

      // Adjust the offset such that it matches the right bound of the range.
      if (offsetAdd != 0) {
        if (offsetDyn)
          offsetDyn = moore::AddOp::create(
              builder, loc, offsetDyn,
              moore::ConstantOp::create(
                  builder, loc, cast<moore::IntType>(offsetDyn.getType()),
                  offsetAdd,
                  /*isSigned=*/offsetAdd < 0));
        else
          offsetConst += offsetAdd;
      }
    }

    // Create a dynamic or constant extract. Use `getSelectIndex` and
    // `ConstantRange::translateIndex` to map from the bit indices provided by
    // the user to the actual storage bit position. Since `offset*` corresponds
    // to the right bound of the range, which provides the index of the least
    // significant selected storage bit, we get the bit offset at which we want
    // to start extracting.
    auto resultType =
        isLvalue ? moore::RefType::get(cast<moore::UnpackedType>(type)) : type;

    if (offsetDyn) {
      offsetDyn = getSelectIndex(builder, loc, offsetDyn, range);
      if (isLvalue) {
        return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                              offsetDyn);
      } else {
        return moore::DynExtractOp::create(builder, loc, resultType, value,
                                           offsetDyn);
      }
    } else {
      offsetConst = range.translateIndex(offsetConst);
      if (isLvalue) {
        return moore::ExtractRefOp::create(builder, loc, resultType, value,
                                           offsetConst);
      } else {
        return moore::ExtractOp::create(builder, loc, resultType, value,
                                        offsetConst);
      }
    }
  }

  /// Handle concatenations.
  Value visit(const slang::ast::ConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto *operand : expr.operands()) {
      // Handle empty replications like `{0{...}}` which may occur within
      // concatenations. Slang assigns them a `void` type which we can check for
      // here.
      if (operand->type->isVoid())
        continue;
      auto value = convertLvalueOrRvalueExpression(*operand);
      if (!value)
        return {};
      if (!isLvalue)
        value = context.convertToSimpleBitVector(value);
      if (!value)
        return {};
      operands.push_back(value);
    }
    if (isLvalue)
      return moore::ConcatRefOp::create(builder, loc, operands);
    else
      return moore::ConcatOp::create(builder, loc, operands);
  }

  /// Handle member accesses.
  Value visit(const slang::ast::MemberAccessExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto valueType = expr.value().type;
    auto value = convertLvalueOrRvalueExpression(expr.value());
    if (!type || !value)
      return {};

    auto resultType =
        isLvalue ? moore::RefType::get(cast<moore::UnpackedType>(type)) : type;
    auto memberName = builder.getStringAttr(expr.member.name);

    // Handle structs.
    if (valueType->isStruct()) {
      if (isLvalue)
        return moore::StructExtractRefOp::create(builder, loc, resultType,
                                                 memberName, value);
      else
        return moore::StructExtractOp::create(builder, loc, resultType,
                                              memberName, value);
    }

    // Handle unions.
    if (valueType->isPackedUnion() || valueType->isUnpackedUnion()) {
      if (isLvalue)
        return moore::UnionExtractRefOp::create(builder, loc, resultType,
                                                memberName, value);
      else
        return moore::UnionExtractOp::create(builder, loc, type, memberName,
                                             value);
    }

    mlir::emitError(loc, "expression of type ")
        << value.getType() << " has no member fields";
    return {};
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Rvalue Conversion
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct RvalueExprVisitor : public ExprVisitor {
  RvalueExprVisitor(Context &context, Location loc)
      : ExprVisitor(context, loc, /*isLvalue=*/false) {}
  using ExprVisitor::visit;

  // Handle references to the left-hand side of a parent assignment.
  Value visit(const slang::ast::LValueReferenceExpression &expr) {
    assert(!context.lvalueStack.empty() && "parent assignments push lvalue");
    auto lvalue = context.lvalueStack.back();
    return moore::ReadOp::create(builder, loc, lvalue);
  }

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol)) {
      if (isa<moore::RefType>(value.getType())) {
        auto readOp = moore::ReadOp::create(builder, loc, value);
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

  // Handle hierarchical values, such as `x = Top.sub.var`.
  Value visit(const slang::ast::HierarchicalValueExpression &expr) {
    auto hierLoc = context.convertLocation(expr.symbol.location);
    if (auto value = context.valueSymbols.lookup(&expr.symbol)) {
      if (isa<moore::RefType>(value.getType())) {
        auto readOp = moore::ReadOp::create(builder, hierLoc, value);
        if (context.rvalueReadCallback)
          context.rvalueReadCallback(readOp);
        value = readOp.getResult();
      }
      return value;
    }

    // Emit an error for those hierarchical values not recorded in the
    // `valueSymbols`.
    auto d = mlir::emitError(loc, "unknown hierarchical name `")
             << expr.symbol.name << "`";
    d.attachNote(hierLoc) << "no rvalue generated for "
                          << slang::ast::toString(expr.symbol.kind);
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
      moore::NonBlockingAssignOp::create(builder, loc, lhs, rhs);
    else
      moore::BlockingAssignOp::create(builder, loc, lhs, rhs);
    return rhs;
  }

  // Helper function to convert an argument to a simple bit vector type, pass it
  // to a reduction op, and optionally invert the result.
  template <class ConcreteOp>
  Value createReduction(Value arg, bool invert) {
    arg = context.convertToSimpleBitVector(arg);
    if (!arg)
      return {};
    Value result = ConcreteOp::create(builder, loc, arg);
    if (invert)
      result = moore::NotOp::create(builder, loc, result);
    return result;
  }

  // Helper function to create pre and post increments and decrements.
  Value createIncrement(Value arg, bool isInc, bool isPost) {
    auto preValue = moore::ReadOp::create(builder, loc, arg);
    auto one = moore::ConstantOp::create(
        builder, loc, cast<moore::IntType>(preValue.getType()), 1);
    auto postValue =
        isInc ? moore::AddOp::create(builder, loc, preValue, one).getResult()
              : moore::SubOp::create(builder, loc, preValue, one).getResult();
    moore::BlockingAssignOp::create(builder, loc, arg, postValue);
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
      return moore::NegOp::create(builder, loc, arg);

    case UnaryOperator::BitwiseNot:
      arg = context.convertToSimpleBitVector(arg);
      if (!arg)
        return {};
      return moore::NotOp::create(builder, loc, arg);

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
      return moore::NotOp::create(builder, loc, arg);

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
    return ConcreteOp::create(builder, loc, lhs, rhs);
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
          moore::ConversionOp::create(builder, loc, lhs.getType(), rhs);
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
      return moore::NotOp::create(builder, loc, result);
    }

    case BinaryOperator::Equality:
      if (isa<moore::UnpackedArrayType>(lhs.getType()))
        return moore::UArrayCmpOp::create(
            builder, loc, moore::UArrayCmpPredicate::eq, lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::eq, lhs, rhs);
      else
        return createBinary<moore::EqOp>(lhs, rhs);
    case BinaryOperator::Inequality:
      if (isa<moore::UnpackedArrayType>(lhs.getType()))
        return moore::UArrayCmpOp::create(
            builder, loc, moore::UArrayCmpPredicate::ne, lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::ne, lhs, rhs);
      else
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
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::ge, lhs, rhs);
      else
        return createBinary<moore::UgeOp>(lhs, rhs);
    case BinaryOperator::GreaterThan:
      if (expr.left().type->isSigned())
        return createBinary<moore::SgtOp>(lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::gt, lhs, rhs);
      else
        return createBinary<moore::UgtOp>(lhs, rhs);
    case BinaryOperator::LessThanEqual:
      if (expr.left().type->isSigned())
        return createBinary<moore::SleOp>(lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::le, lhs, rhs);
      else
        return createBinary<moore::UleOp>(lhs, rhs);
    case BinaryOperator::LessThan:
      if (expr.left().type->isSigned())
        return createBinary<moore::SltOp>(lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::lt, lhs, rhs);
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
      return moore::AndOp::create(builder, loc, lhs, rhs);
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
      return moore::OrOp::create(builder, loc, lhs, rhs);
    }
    case BinaryOperator::LogicalImplication: {
      // `(lhs -> rhs)` equivalent to `(!lhs || rhs)`.
      lhs = context.convertToBool(lhs, domain);
      if (!lhs)
        return {};
      rhs = context.convertToBool(rhs, domain);
      if (!rhs)
        return {};
      auto notLHS = moore::NotOp::create(builder, loc, lhs);
      return moore::OrOp::create(builder, loc, notLHS, rhs);
    }
    case BinaryOperator::LogicalEquivalence: {
      // `(lhs <-> rhs)` equivalent to `(lhs && rhs) || (!lhs && !rhs)`.
      lhs = context.convertToBool(lhs, domain);
      if (!lhs)
        return {};
      rhs = context.convertToBool(rhs, domain);
      if (!rhs)
        return {};
      auto notLHS = moore::NotOp::create(builder, loc, lhs);
      auto notRHS = moore::NotOp::create(builder, loc, rhs);
      auto both = moore::AndOp::create(builder, loc, lhs, rhs);
      auto notBoth = moore::AndOp::create(builder, loc, notLHS, notRHS);
      return moore::OrOp::create(builder, loc, both, notBoth);
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
        return moore::AShrOp::create(builder, loc, lhs, rhs);
      return moore::ShrOp::create(builder, loc, lhs, rhs);
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

  // Handle time literals.
  Value visit(const slang::ast::TimeLiteral &expr) {
    // The time literal is expressed in the current time scale. Determine the
    // conversion factor to convert the literal from the current time scale into
    // femtoseconds.
    static_assert(int(slang::TimeUnit::Seconds) == 0);
    static_assert(int(slang::TimeUnit::Femtoseconds) == 5);
    static_assert(int(slang::TimeScaleMagnitude::One) == 1);
    static_assert(int(slang::TimeScaleMagnitude::Ten) == 10);
    static_assert(int(slang::TimeScaleMagnitude::Hundred) == 100);
    static constexpr double units[] = {1e15, 1e12, 1e9, 1e6, 1e3, 1e0};
    auto base = context.timeScale.base;
    double scale = units[int(base.unit)] * int(base.magnitude);

    // Convert and round the value to femtoseconds.
    double value = std::round(expr.getValue() * scale);
    assert(value >= 0.0);

    // Check that the value does not exceed what we can represent in the IR.
    // Casting the maximum uint64 value to double changes its value from
    // 18446744073709551615 to 18446744073709551616, which makes the comparison
    // overestimate the largest number we can represent. To avoid this, round
    // the maximum value down to the closest number that only has the front 53
    // bits set. This matches the mantissa of a double, plus the implicit
    // leading 1, ensuring that we can accurately represent the limit.
    static constexpr uint64_t limit =
        (std::numeric_limits<uint64_t>::max() >> 11) << 11;
    if (value > limit) {
      mlir::emitError(loc) << "time value is larger than " << limit << " fs";
      return {};
    }

    return moore::ConstantTimeOp::create(builder, loc,
                                         static_cast<uint64_t>(value));
  }

  // Handle replications.
  Value visit(const slang::ast::ReplicationExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertRvalueExpression(expr.concat());
    if (!value)
      return {};
    return moore::ReplicateOp::create(builder, loc, type, value);
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
          leftValue = moore::SgeOp::create(builder, loc, lhs, lowBound);
        } else {
          leftValue = moore::UgeOp::create(builder, loc, lhs, lowBound);
        }
        if (openRange->right().type->isSigned() ||
            expr.left().type->isSigned()) {
          rightValue = moore::SleOp::create(builder, loc, lhs, highBound);
        } else {
          rightValue = moore::UleOp::create(builder, loc, lhs, highBound);
        }
        cond = moore::AndOp::create(builder, loc, leftValue, rightValue);
      } else {
        // Handle expressions.
        if (!listExpr->type->isIntegral()) {
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
        cond = moore::WildcardEqOp::create(builder, loc, lhs, value);
      }
      conditions.push_back(cond);
    }

    // Calculate the final result by `or` op.
    auto result = conditions.back();
    conditions.pop_back();
    while (!conditions.empty()) {
      result = moore::OrOp::create(builder, loc, conditions.back(), result);
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
    auto conditionalOp =
        moore::ConditionalOp::create(builder, loc, type, value);

    // Create blocks for true region and false region.
    auto &trueBlock = conditionalOp.getTrueRegion().emplaceBlock();
    auto &falseBlock = conditionalOp.getFalseRegion().emplaceBlock();

    OpBuilder::InsertionGuard g(builder);

    // Handle left expression.
    builder.setInsertionPointToStart(&trueBlock);
    auto trueValue = context.convertRvalueExpression(expr.left(), type);
    if (!trueValue)
      return {};
    moore::YieldOp::create(builder, loc, trueValue);

    // Handle right expression.
    builder.setInsertionPointToStart(&falseBlock);
    auto falseValue = context.convertRvalueExpression(expr.right(), type);
    if (!falseValue)
      return {};
    moore::YieldOp::create(builder, loc, falseValue);

    return conditionalOp.getResult();
  }

  /// Handle calls.
  Value visit(const slang::ast::CallExpression &expr) {
    // Class method calls are currently not supported.
    if (expr.thisClass()) {
      mlir::emitError(loc, "unsupported class method call");
      return {};
    }

    // Try to materialize constant values directly.
    auto constant = context.evaluateConstant(expr);
    if (auto value = context.materializeConstant(constant, *expr.type, loc))
      return value;

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
        mlir::func::CallOp::create(builder, loc, lowering->op, arguments);

    // For calls to void functions we need to have a value to return from this
    // function. Create a dummy `unrealized_conversion_cast`, which will get
    // deleted again later on.
    if (callOp.getNumResults() == 0)
      return mlir::UnrealizedConversionCastOp::create(
                 builder, loc, moore::VoidType::get(context.getContext()),
                 ValueRange{})
          .getResult(0);

    return callOp.getResult(0);
  }

  /// Handle system calls.
  Value visitCall(const slang::ast::CallExpression &expr,
                  const slang::ast::CallExpression::SystemCallInfo &info) {
    const auto &subroutine = *info.subroutine;
    auto args = expr.arguments();

    if (args.size() == 1) {
      auto value = context.convertRvalueExpression(*args[0]);
      if (!value)
        return {};
      auto result = context.convertSystemCallArity1(subroutine, loc, value);
      if (failed(result))
        return {};
      if (*result)
        return *result;
    }

    mlir::emitError(loc) << "unsupported system call `" << subroutine.name
                         << "`";
    return {};
  }

  /// Handle string literals.
  Value visit(const slang::ast::StringLiteral &expr) {
    auto type = context.convertType(*expr.type);
    return moore::StringConstantOp::create(builder, loc, type, expr.getValue());
  }

  /// Handle real literals.
  Value visit(const slang::ast::RealLiteral &expr) {
    return moore::RealLiteralOp::create(
        builder, loc, builder.getF64FloatAttr(expr.getValue()));
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

    // Handle integers.
    if (auto intType = dyn_cast<moore::IntType>(type)) {
      assert(intType.getWidth() == elements.size());
      std::reverse(elements.begin(), elements.end());
      return moore::ConcatOp::create(builder, loc, intType, elements);
    }

    // Handle packed structs.
    if (auto structType = dyn_cast<moore::StructType>(type)) {
      assert(structType.getMembers().size() == elements.size());
      return moore::StructCreateOp::create(builder, loc, structType, elements);
    }

    // Handle unpacked structs.
    if (auto structType = dyn_cast<moore::UnpackedStructType>(type)) {
      assert(structType.getMembers().size() == elements.size());
      return moore::StructCreateOp::create(builder, loc, structType, elements);
    }

    // Handle packed arrays.
    if (auto arrayType = dyn_cast<moore::ArrayType>(type)) {
      assert(arrayType.getSize() == elements.size());
      return moore::ArrayCreateOp::create(builder, loc, arrayType, elements);
    }

    // Handle unpacked arrays.
    if (auto arrayType = dyn_cast<moore::UnpackedArrayType>(type)) {
      assert(arrayType.getSize() == elements.size());
      return moore::ArrayCreateOp::create(builder, loc, arrayType, elements);
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

  Value visit(const slang::ast::StreamingConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto stream : expr.streams()) {
      auto operandLoc = context.convertLocation(stream.operand->sourceRange);
      if (!stream.constantWithWidth.has_value() && stream.withExpr) {
        mlir::emitError(operandLoc)
            << "Moore only support streaming "
               "concatenation with fixed size 'with expression'";
        return {};
      }
      Value value;
      if (stream.constantWithWidth.has_value()) {
        value = context.convertRvalueExpression(*stream.withExpr);
        auto type = cast<moore::UnpackedType>(value.getType());
        auto intType = moore::IntType::get(
            context.getContext(), type.getBitSize().value(), type.getDomain());
        // Do not care if it's signed, because we will not do expansion.
        value = context.materializeConversion(intType, value, false, loc);
      } else {
        value = context.convertRvalueExpression(*stream.operand);
      }

      value = context.convertToSimpleBitVector(value);
      if (!value)
        return {};
      operands.push_back(value);
    }
    Value value;

    if (operands.size() == 1) {
      // There must be at least one element, otherwise slang will report an
      // error.
      value = operands.front();
    } else {
      value = moore::ConcatOp::create(builder, loc, operands).getResult();
    }

    if (expr.sliceSize == 0) {
      return value;
    }

    auto type = cast<moore::IntType>(value.getType());
    SmallVector<Value> slicedOperands;
    auto iterMax = type.getWidth() / expr.sliceSize;
    auto remainSize = type.getWidth() % expr.sliceSize;

    for (size_t i = 0; i < iterMax; i++) {
      auto extractResultType = moore::IntType::get(
          context.getContext(), expr.sliceSize, type.getDomain());

      auto extracted = moore::ExtractOp::create(builder, loc, extractResultType,
                                                value, i * expr.sliceSize);
      slicedOperands.push_back(extracted);
    }
    // Handle other wire
    if (remainSize) {
      auto extractResultType = moore::IntType::get(
          context.getContext(), remainSize, type.getDomain());

      auto extracted = moore::ExtractOp::create(
          builder, loc, extractResultType, value, iterMax * expr.sliceSize);
      slicedOperands.push_back(extracted);
    }

    return moore::ConcatOp::create(builder, loc, slicedOperands);
  }

  Value visit(const slang::ast::AssertionInstanceExpression &expr) {
    return context.convertAssertionExpression(expr.body, loc);
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

//===----------------------------------------------------------------------===//
// Lvalue Conversion
//===----------------------------------------------------------------------===//

namespace {
struct LvalueExprVisitor : public ExprVisitor {
  LvalueExprVisitor(Context &context, Location loc)
      : ExprVisitor(context, loc, /*isLvalue=*/true) {}
  using ExprVisitor::visit;

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;
    auto d = mlir::emitError(loc, "unknown name `") << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no lvalue generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle hierarchical values, such as `Top.sub.var = x`.
  Value visit(const slang::ast::HierarchicalValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;

    // Emit an error for those hierarchical values not recorded in the
    // `valueSymbols`.
    auto d = mlir::emitError(loc, "unknown hierarchical name `")
             << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no lvalue generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  Value visit(const slang::ast::StreamingConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto stream : expr.streams()) {
      auto operandLoc = context.convertLocation(stream.operand->sourceRange);
      if (!stream.constantWithWidth.has_value() && stream.withExpr) {
        mlir::emitError(operandLoc)
            << "Moore only support streaming "
               "concatenation with fixed size 'with expression'";
        return {};
      }
      Value value;
      if (stream.constantWithWidth.has_value()) {
        value = context.convertLvalueExpression(*stream.withExpr);
        auto type = cast<moore::UnpackedType>(
            cast<moore::RefType>(value.getType()).getNestedType());
        auto intType = moore::RefType::get(moore::IntType::get(
            context.getContext(), type.getBitSize().value(), type.getDomain()));
        // Do not care if it's signed, because we will not do expansion.
        value = context.materializeConversion(intType, value, false, loc);
      } else {
        value = context.convertLvalueExpression(*stream.operand);
      }

      if (!value)
        return {};
      operands.push_back(value);
    }
    Value value;
    if (operands.size() == 1) {
      // There must be at least one element, otherwise slang will report an
      // error.
      value = operands.front();
    } else {
      value = moore::ConcatRefOp::create(builder, loc, operands).getResult();
    }

    if (expr.sliceSize == 0) {
      return value;
    }

    auto type = cast<moore::IntType>(
        cast<moore::RefType>(value.getType()).getNestedType());
    SmallVector<Value> slicedOperands;
    auto widthSum = type.getWidth();
    auto domain = type.getDomain();
    auto iterMax = widthSum / expr.sliceSize;
    auto remainSize = widthSum % expr.sliceSize;

    for (size_t i = 0; i < iterMax; i++) {
      auto extractResultType = moore::RefType::get(
          moore::IntType::get(context.getContext(), expr.sliceSize, domain));

      auto extracted = moore::ExtractRefOp::create(
          builder, loc, extractResultType, value, i * expr.sliceSize);
      slicedOperands.push_back(extracted);
    }
    // Handle other wire
    if (remainSize) {
      auto extractResultType = moore::RefType::get(
          moore::IntType::get(context.getContext(), remainSize, domain));

      auto extracted = moore::ExtractRefOp::create(
          builder, loc, extractResultType, value, iterMax * expr.sliceSize);
      slicedOperands.push_back(extracted);
    }

    return moore::ConcatRefOp::create(builder, loc, slicedOperands);
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

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

Value Context::convertRvalueExpression(const slang::ast::Expression &expr,
                                       Type requiredType) {
  auto loc = convertLocation(expr.sourceRange);
  auto value = expr.visit(RvalueExprVisitor(*this, loc));
  if (value && requiredType)
    value =
        materializeConversion(requiredType, value, expr.type->isSigned(), loc);
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
    return moore::BoolCastOp::create(builder, value.getLoc(), value);
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
  Value result = moore::ConstantOp::create(builder, loc, intType, fvint);
  if (result.getType() != type)
    result = moore::ConversionOp::create(builder, loc, type, result);
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
  return moore::ConversionOp::create(builder, value.getLoc(), type, value);
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
      return moore::ConversionOp::create(builder, value.getLoc(), sbvType,
                                         value);
    }
  }

  mlir::emitError(value.getLoc()) << "expression of type " << value.getType()
                                  << " cannot be cast to a simple bit vector";
  return {};
}

Value Context::materializeConversion(Type type, Value value, bool isSigned,
                                     Location loc) {
  if (type == value.getType())
    return value;
  auto dstPacked = dyn_cast<moore::PackedType>(type);
  auto srcPacked = dyn_cast<moore::PackedType>(value.getType());

  // Resize the value if needed.
  if (dstPacked && srcPacked && dstPacked.getBitSize() &&
      srcPacked.getBitSize() &&
      *dstPacked.getBitSize() != *srcPacked.getBitSize()) {
    auto dstWidth = *dstPacked.getBitSize();
    auto srcWidth = *srcPacked.getBitSize();

    // Convert the value to a simple bit vector which we can extend or truncate.
    auto srcWidthType = moore::IntType::get(value.getContext(), srcWidth,
                                            srcPacked.getDomain());
    if (value.getType() != srcWidthType)
      value = moore::ConversionOp::create(builder, value.getLoc(), srcWidthType,
                                          value);

    // Create truncation or sign/zero extension ops depending on the source and
    // destination width.
    auto dstWidthType = moore::IntType::get(value.getContext(), dstWidth,
                                            srcPacked.getDomain());
    if (dstWidth < srcWidth) {
      value = moore::TruncOp::create(builder, loc, dstWidthType, value);
    } else if (dstWidth > srcWidth) {
      if (isSigned)
        value = moore::SExtOp::create(builder, loc, dstWidthType, value);
      else
        value = moore::ZExtOp::create(builder, loc, dstWidthType, value);
    }
  }

  if (value.getType() != type)
    value = moore::ConversionOp::create(builder, loc, type, value);
  return value;
}

FailureOr<Value>
Context::convertSystemCallArity1(const slang::ast::SystemSubroutine &subroutine,
                                 Location loc, Value value) {
  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          // Signed and unsigned system functions.
          .Case("$signed", [&]() { return value; })
          .Case("$unsigned", [&]() { return value; })

          // Math functions in SystemVerilog.
          .Case("$clog2",
                [&]() -> FailureOr<Value> {
                  value = convertToSimpleBitVector(value);
                  if (!value)
                    return failure();
                  return (Value)moore::Clog2BIOp::create(builder, loc, value);
                })
          .Case("$ln",
                [&]() -> Value {
                  return moore::LnBIOp::create(builder, loc, value);
                })
          .Case("$log10",
                [&]() -> Value {
                  return moore::Log10BIOp::create(builder, loc, value);
                })
          .Case("$sin",
                [&]() -> Value {
                  return moore::SinBIOp::create(builder, loc, value);
                })
          .Case("$cos",
                [&]() -> Value {
                  return moore::CosBIOp::create(builder, loc, value);
                })
          .Case("$tan",
                [&]() -> Value {
                  return moore::TanBIOp::create(builder, loc, value);
                })
          .Case("$exp",
                [&]() -> Value {
                  return moore::ExpBIOp::create(builder, loc, value);
                })
          .Case("$sqrt",
                [&]() -> Value {
                  return moore::SqrtBIOp::create(builder, loc, value);
                })
          .Case("$floor",
                [&]() -> Value {
                  return moore::FloorBIOp::create(builder, loc, value);
                })
          .Case("$ceil",
                [&]() -> Value {
                  return moore::CeilBIOp::create(builder, loc, value);
                })
          .Case("$asin",
                [&]() -> Value {
                  return moore::AsinBIOp::create(builder, loc, value);
                })
          .Case("$acos",
                [&]() -> Value {
                  return moore::AcosBIOp::create(builder, loc, value);
                })
          .Case("$atan",
                [&]() -> Value {
                  return moore::AtanBIOp::create(builder, loc, value);
                })
          .Case("$sinh",
                [&]() -> Value {
                  return moore::SinhBIOp::create(builder, loc, value);
                })
          .Case("$cosh",
                [&]() -> Value {
                  return moore::CoshBIOp::create(builder, loc, value);
                })
          .Case("$tanh",
                [&]() -> Value {
                  return moore::TanhBIOp::create(builder, loc, value);
                })
          .Case("$asinh",
                [&]() -> Value {
                  return moore::AsinhBIOp::create(builder, loc, value);
                })
          .Case("$acosh",
                [&]() -> Value {
                  return moore::AcoshBIOp::create(builder, loc, value);
                })
          .Case("$atanh",
                [&]() -> Value {
                  return moore::AtanhBIOp::create(builder, loc, value);
                })
          .Default([&]() -> Value { return {}; });
  return systemCallRes();
}
