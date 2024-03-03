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

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;
    auto d = mlir::emitError(loc, "unknown name `") << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no value generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle blocking and non-blocking assignments.
  Value visit(const slang::ast::AssignmentExpression &expr) {
    auto lhs = context.convertExpression(expr.left());
    auto rhs = context.convertExpression(expr.right());
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
    return lhs;
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
