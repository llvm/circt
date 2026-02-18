//===- AssertionExpr.cpp - Slang assertion expression conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "slang/ast/expressions/AssertionExpr.h"
#include "ImportVerilogInternals.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "slang/ast/SystemSubroutine.h"

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
      return ltl::RepeatOp::create(builder, loc, inputSequence, minRepetitions,
                                   repetitionRange);
    case SequenceRepetition::Nonconsecutive:
      return ltl::NonConsecutiveRepeatOp::create(
          builder, loc, inputSequence, minRepetitions, repetitionRange);
    case SequenceRepetition::GoTo:
      return ltl::GoToRepeatOp::create(builder, loc, inputSequence,
                                       minRepetitions, repetitionRange);
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::SimpleAssertionExpr &expr) {
    // Handle expression
    auto value = context.convertRvalueExpression(expr.expr);
    if (!value)
      return {};
    auto loc = context.convertLocation(expr.expr.sourceRange);
    auto valueType = value.getType();
    // For assertion instances the value is already the expected type, convert
    // boolean value
    if (!mlir::isa<ltl::SequenceType, ltl::PropertyType, mlir::IntegerType>(
            valueType)) {
      value = context.convertToI1(value);
    }
    if (!value)
      return {};

    // Handle repetition
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

      [[maybe_unused]] Type valueType = sequenceValue.getType();
      assert(valueType.isInteger(1) || mlir::isa<ltl::SequenceType>(valueType));

      auto [delayMin, delayRange] =
          convertRangeToAttrs(concatElement.delay.min, concatElement.delay.max);
      auto delayedSequence = ltl::DelayOp::create(builder, loc, sequenceValue,
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
      return ltl::NotOp::create(builder, loc, value);
    case UnaryAssertionOperator::SEventually:
      if (expr.range.has_value()) {
        mlir::emitError(loc, "Strong eventually with range not supported");
        return {};
      } else {
        return ltl::EventuallyOp::create(builder, loc, value);
      }
    case UnaryAssertionOperator::Always: {
      std::pair<mlir::IntegerAttr, mlir::IntegerAttr> attr = {
          builder.getI64IntegerAttr(0), mlir::IntegerAttr{}};
      if (expr.range.has_value()) {
        attr =
            convertRangeToAttrs(expr.range.value().min, expr.range.value().max);
      }
      return ltl::RepeatOp::create(builder, loc, value, attr.first,
                                   attr.second);
    }
    case UnaryAssertionOperator::NextTime: {
      auto minRepetitions = builder.getI64IntegerAttr(1);
      if (expr.range.has_value()) {
        minRepetitions = builder.getI64IntegerAttr(expr.range.value().min);
      }
      return ltl::DelayOp::create(builder, loc, value, minRepetitions,
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
      return ltl::AndOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Or:
      return ltl::OrOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Intersect:
      return ltl::IntersectOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Throughout: {
      auto lhsRepeat = ltl::RepeatOp::create(
          builder, loc, lhs, builder.getI64IntegerAttr(0), mlir::IntegerAttr{});
      return ltl::IntersectOp::create(builder, loc,
                                      SmallVector<Value, 2>{lhsRepeat, rhs});
    }
    case BinaryAssertionOperator::Within: {
      auto constOne =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      auto oneRepeat = ltl::RepeatOp::create(builder, loc, constOne,
                                             builder.getI64IntegerAttr(0),
                                             mlir::IntegerAttr{});
      auto repeatDelay = ltl::DelayOp::create(builder, loc, oneRepeat,
                                              builder.getI64IntegerAttr(1),
                                              builder.getI64IntegerAttr(0));
      auto lhsDelay =
          ltl::DelayOp::create(builder, loc, lhs, builder.getI64IntegerAttr(1),
                               builder.getI64IntegerAttr(0));
      auto combined = ltl::ConcatOp::create(
          builder, loc, SmallVector<Value, 3>{repeatDelay, lhsDelay, constOne});
      return ltl::IntersectOp::create(builder, loc,
                                      SmallVector<Value, 2>{combined, rhs});
    }
    case BinaryAssertionOperator::Iff: {
      auto ored = ltl::OrOp::create(builder, loc, operands);
      auto notOred = ltl::NotOp::create(builder, loc, ored);
      auto anded = ltl::AndOp::create(builder, loc, operands);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notOred, anded});
    }
    case BinaryAssertionOperator::Until:
      return ltl::UntilOp::create(builder, loc, operands);
    case BinaryAssertionOperator::UntilWith: {
      auto untilOp = ltl::UntilOp::create(builder, loc, operands);
      auto andOp = ltl::AndOp::create(builder, loc, operands);
      auto notUntil = ltl::NotOp::create(builder, loc, untilOp);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notUntil, andOp});
    }
    case BinaryAssertionOperator::Implies: {
      auto notLhs = ltl::NotOp::create(builder, loc, lhs);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notLhs, rhs});
    }
    case BinaryAssertionOperator::OverlappedImplication:
      return ltl::ImplicationOp::create(builder, loc, operands);
    case BinaryAssertionOperator::NonOverlappedImplication: {
      auto constOne =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      auto lhsDelay =
          ltl::DelayOp::create(builder, loc, lhs, builder.getI64IntegerAttr(1),
                               builder.getI64IntegerAttr(0));
      auto antecedent = ltl::ConcatOp::create(
          builder, loc, SmallVector<Value, 2>{lhsDelay, constOne});
      return ltl::ImplicationOp::create(builder, loc,
                                        SmallVector<Value, 2>{antecedent, rhs});
    }
    case BinaryAssertionOperator::OverlappedFollowedBy: {
      auto notRhs = ltl::NotOp::create(builder, loc, rhs);
      auto implication = ltl::ImplicationOp::create(
          builder, loc, SmallVector<Value, 2>{lhs, notRhs});
      return ltl::NotOp::create(builder, loc, implication);
    }
    case BinaryAssertionOperator::NonOverlappedFollowedBy: {
      auto constOne =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      auto notRhs = ltl::NotOp::create(builder, loc, rhs);
      auto lhsDelay =
          ltl::DelayOp::create(builder, loc, lhs, builder.getI64IntegerAttr(1),
                               builder.getI64IntegerAttr(0));
      auto antecedent = ltl::ConcatOp::create(
          builder, loc, SmallVector<Value, 2>{lhsDelay, constOne});
      auto implication = ltl::ImplicationOp::create(
          builder, loc, SmallVector<Value, 2>{antecedent, notRhs});
      return ltl::NotOp::create(builder, loc, implication);
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

FailureOr<Value> Context::convertAssertionSystemCallArity1(
    const slang::ast::SystemSubroutine &subroutine, Location loc, Value value,
    Type originalType) {

  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          // Translate $fell to ¬x[0] ∧ x[-1]
          .Case("$fell",
                [&]() -> Value {
                  auto current = value;
                  auto past =
                      ltl::PastOp::create(builder, loc, value, 1, Value{})
                          .getResult();
                  auto fell = comb::ICmpOp::create(builder, loc,
                                                   comb::ICmpPredicate::ugt,
                                                   past, current, false)
                                  .getResult();
                  return fell;
                })
          // Translate $rose to x[0] ∧ ¬x[-1]
          .Case("$rose",
                [&]() -> Value {
                  auto past =
                      ltl::PastOp::create(builder, loc, value, 1, Value{})
                          .getResult();
                  auto current = value;
                  auto rose = comb::ICmpOp::create(builder, loc,
                                                   comb::ICmpPredicate::ult,
                                                   past, current, false)
                                  .getResult();
                  return rose;
                })
          // Translate $changed to ( ¬x[0] ∧ x[-1] ) ⋁ ( x[0] ∧ ¬x[-1] )
          .Case("$changed",
                [&]() -> Value {
                  auto past =
                      ltl::PastOp::create(builder, loc, value, 1, Value{})
                          .getResult();
                  auto current = value;
                  auto changed = comb::ICmpOp::create(builder, loc,
                                                      comb::ICmpPredicate::ne,
                                                      past, current, false)
                                     .getResult();
                  return changed;
                })
          // Translate $stable to ( x[0] ∧ x[-1] ) ⋁ ( ¬x[0] ∧ ¬x[-1] )
          .Case("$stable",
                [&]() -> Value {
                  auto past =
                      ltl::PastOp::create(builder, loc, value, 1, Value{})
                          .getResult();
                  auto current = value;
                  auto stable = comb::ICmpOp::create(builder, loc,
                                                     comb::ICmpPredicate::eq,
                                                     past, current, false)
                                    .getResult();
                  return stable;
                })
          .Case("$past",
                [&]() -> Value {
                  Value past =
                      ltl::PastOp::create(builder, loc, value, 1, Value{});
                  // Cast back to Moore integers so Moore ops can use the result
                  // if needed
                  if (auto ty = dyn_cast<moore::IntType>(originalType)) {
                    past = moore::FromBuiltinIntOp::create(builder, loc, past);
                    if (ty.getDomain() == Domain::FourValued)
                      past = moore::IntToLogicOp::create(builder, loc, past);
                  }
                  return past;
                })
          .Default([&]() -> Value { return {}; });
  return systemCallRes();
}

Value Context::convertAssertionCallExpression(
    const slang::ast::CallExpression &expr,
    const slang::ast::CallExpression::SystemCallInfo &info, Location loc) {

  const auto &subroutine = *info.subroutine;
  auto args = expr.arguments();

  FailureOr<Value> result;
  Value value;
  Value intVal;
  Type originalType;
  moore::IntType valTy;

  switch (args.size()) {
  case (1):
    value = this->convertRvalueExpression(*args[0]);
    originalType = value.getType();
    valTy = dyn_cast<moore::IntType>(value.getType());
    if (!valTy) {
      mlir::emitError(loc) << "expected integer argument for system call `"
                           << subroutine.name << "`";
      return {};
    }
    // If the value is four-valued, we need to map it to two-valued before we
    // cast it to a builtin int
    if (valTy.getDomain() == Domain::FourValued) {
      value = builder.createOrFold<moore::LogicToIntOp>(loc, value);
    }
    intVal = builder.createOrFold<moore::ToBuiltinIntOp>(loc, value);
    if (!intVal)
      return {};
    result = this->convertAssertionSystemCallArity1(subroutine, loc, intVal,
                                                    originalType);
    break;

  default:
    break;
  }

  if (failed(result))
    return {};
  if (*result)
    return *result;

  mlir::emitError(loc) << "unsupported system call `" << subroutine.name << "`";
  return {};
}

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
  auto loc = value.getLoc();
  auto type = dyn_cast<moore::IntType>(value.getType());
  if (!type || type.getBitSize() != 1) {
    mlir::emitError(loc, "expected a 1-bit integer");
    return {};
  }
  if (type.getDomain() == Domain::FourValued) {
    value = moore::LogicToIntOp::create(builder, loc, value);
  }
  return moore::ToBuiltinIntOp::create(builder, loc, value);
}
