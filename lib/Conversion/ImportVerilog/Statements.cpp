//===- Statements.cpp - Slang statement conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"

using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct StmtVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  StmtVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Skip empty statements (stray semicolons).
  LogicalResult visit(const slang::ast::EmptyStatement &) { return success(); }

  // Convert every statement in a statement list. The Verilog syntax follows a
  // similar philosophy as C/C++, where things like `if` and `for` accept a
  // single statement as body. But then a `{...}` block is a valid statement,
  // which allows for the `if {...}` syntax. In Verilog, things like `final`
  // accept a single body statement, but that can be a `begin ... end` block,
  // which in turn has a single body statement, which then commonly is a list of
  // statements.
  LogicalResult visit(const slang::ast::StatementList &stmts) {
    for (auto *stmt : stmts.list)
      if (failed(context.convertStatement(*stmt)))
        return failure();
    return success();
  }

  // Inline `begin ... end` blocks into the parent.
  LogicalResult visit(const slang::ast::BlockStatement &stmt) {
    return context.convertStatement(stmt.body);
  }

  // Handle expression statements.
  LogicalResult visit(const slang::ast::ExpressionStatement &stmt) {
    return success(context.convertExpression(stmt.expr));
  }

  // Handle variable declarations.
  LogicalResult visit(const slang::ast::VariableDeclStatement &stmt) {
    const auto &var = stmt.symbol;
    auto type = context.convertType(*var.getDeclaredType());
    if (!type)
      return failure();

    Value initial;
    if (const auto *init = var.getInitializer()) {
      initial = context.convertExpression(*init);
      if (!initial)
        return failure();
    }

    builder.create<moore::VariableOp>(loc, type,
                                      builder.getStringAttr(var.name), initial);
    return success();
  }

  // Ignore timing control for now.
  LogicalResult visit(const slang::ast::TimedStatement &stmt) {
    return success();
  }

  /// Emit an error for all other statements.
  template <typename T>
  LogicalResult visit(T &&stmt) {
    mlir::emitError(loc, "unsupported statement: ")
        << slang::ast::toString(stmt.kind);
    return mlir::failure();
  }

  LogicalResult visitInvalid(const slang::ast::Statement &stmt) {
    mlir::emitError(loc, "invalid statement: ")
        << slang::ast::toString(stmt.kind);
    return mlir::failure();
  }
};
} // namespace

LogicalResult Context::convertStatement(const slang::ast::Statement &stmt) {
  auto loc = convertLocation(stmt.sourceRange);
  return stmt.visit(StmtVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)
