//===- Statements.cpp - Slang statement conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"

using namespace mlir;
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
    return failure(!context.convertRvalueExpression(stmt.expr));
  }

  // Handle variable declarations.
  LogicalResult visit(const slang::ast::VariableDeclStatement &stmt) {
    const auto &var = stmt.symbol;
    auto type = context.convertType(*var.getDeclaredType());
    if (!type)
      return failure();

    Value initial;
    if (const auto *init = var.getInitializer()) {
      initial = context.convertRvalueExpression(*init);
      if (!initial)
        return failure();
    }

    // Collect local temporary variables.
    auto varOp = builder.create<moore::VariableOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
        builder.getStringAttr(var.name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         &var, varOp);
    return success();
  }

  // Handle if statements.
  LogicalResult visit(const slang::ast::ConditionalStatement &stmt) {
    // Generate the condition. There may be multiple conditions linked with the
    // `&&&` operator.
    Value allConds;
    for (const auto &condition : stmt.conditions) {
      if (condition.pattern)
        return mlir::emitError(loc,
                               "match patterns in if conditions not supported");
      auto cond = context.convertRvalueExpression(*condition.expr);
      if (!cond)
        return failure();
      cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
      if (allConds)
        allConds = builder.create<moore::AndOp>(loc, allConds, cond);
      else
        allConds = cond;
    }
    assert(allConds && "slang guarantees at least one condition");
    allConds =
        builder.create<moore::ConversionOp>(loc, builder.getI1Type(), allConds);

    // Generate the if operation.
    auto ifOp =
        builder.create<scf::IfOp>(loc, allConds, stmt.ifFalse != nullptr);
    OpBuilder::InsertionGuard guard(builder);

    // Generate the "then" body.
    builder.setInsertionPoint(ifOp.thenYield());
    if (failed(context.convertStatement(stmt.ifTrue)))
      return failure();

    // Generate the "else" body if present.
    if (stmt.ifFalse) {
      builder.setInsertionPoint(ifOp.elseYield());
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
    }

    return success();
  }

  // Handle case statements.
  LogicalResult visit(const slang::ast::CaseStatement &caseStmt) {
    auto caseExpr = context.convertRvalueExpression(caseStmt.expr);
    if (!caseExpr)
      return failure();

    auto items = caseStmt.items;
    // Used to generate the condition of the default case statement.
    SmallVector<Value> defaultConds;
    // Traverse the case items.
    for (auto item : items) {
      // One statement will be matched with multi-conditions.
      // Like case(cond) 0, 1 : y = x; endcase.
      SmallVector<Value> allConds;
      for (const auto *expr : item.expressions) {
        auto itemExpr = context.convertRvalueExpression(*expr);
        if (!itemExpr)
          return failure();

        auto newEqOp = builder.create<moore::EqOp>(loc, caseExpr, itemExpr);
        allConds.push_back(newEqOp);
      }
      // Bound all conditions of an item into one.
      auto cond = allConds.back();
      allConds.pop_back();
      while (!allConds.empty()) {
        cond = builder.create<moore::OrOp>(loc, allConds.back(), cond);
        allConds.pop_back();
      }
      // Gather all items' conditions.
      defaultConds.push_back(cond);
      cond =
          builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
      auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(ifOp.thenYield());
      if (failed(context.convertStatement(*item.stmt)))
        return failure();
    }
    // Handle the 'default case' statement if it exists.
    if (caseStmt.defaultCase) {
      auto cond = defaultConds.back();
      defaultConds.pop_back();
      while (!defaultConds.empty()) {
        cond = builder.create<moore::OrOp>(loc, defaultConds.back(), cond);
        defaultConds.pop_back();
      }
      cond = builder.create<moore::NotOp>(loc, cond);
      cond =
          builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
      auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(ifOp.thenYield());
      if (failed(context.convertStatement(*caseStmt.defaultCase)))
        return failure();
    }
    return success();
  }

  // Handle `for` loops.
  LogicalResult visit(const slang::ast::ForLoopStatement &stmt) {
    if (!stmt.loopVars.empty())
      return mlir::emitError(loc,
                             "variables in for loop initializer not supported");

    // Generate the initializers.
    for (auto *initExpr : stmt.initializers)
      if (!context.convertRvalueExpression(*initExpr))
        return failure();

    // Create the while op.
    auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    OpBuilder::InsertionGuard guard(builder);

    // In the "before" region, check that the condition holds.
    builder.createBlock(&whileOp.getBefore());
    auto cond = context.convertRvalueExpression(*stmt.stopExpr);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<mlir::scf::ConditionOp>(loc, cond, ValueRange{});

    // In the "after" region, generate the loop body and step expressions.
    builder.createBlock(&whileOp.getAfter());
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    for (auto *stepExpr : stmt.steps)
      if (!context.convertRvalueExpression(*stepExpr))
        return failure();
    builder.create<mlir::scf::YieldOp>(loc);

    return success();
  }

  // Handle `repeat` loops.
  LogicalResult visit(const slang::ast::RepeatLoopStatement &stmt) {
    // Create the while op and feed in the repeat count as the initial counter
    // value.
    auto count = context.convertRvalueExpression(stmt.count);
    if (!count)
      return failure();
    auto type = cast<moore::IntType>(count.getType());
    auto whileOp = builder.create<scf::WhileOp>(loc, type, count);
    OpBuilder::InsertionGuard guard(builder);

    // In the "before" region, check that the counter is non-zero.
    auto *block = builder.createBlock(&whileOp.getBefore(), {}, type, loc);
    auto counterArg = block->getArgument(0);
    auto cond = builder.createOrFold<moore::BoolCastOp>(loc, counterArg);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<scf::ConditionOp>(loc, cond, counterArg);

    // In the "after" region, generate the loop body and decrement the counter.
    block = builder.createBlock(&whileOp.getAfter(), {}, type, loc);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    counterArg = block->getArgument(0);
    auto constOne = builder.create<moore::ConstantOp>(loc, type, 1);
    auto subOp = builder.create<moore::SubOp>(loc, counterArg, constOne);
    builder.create<scf::YieldOp>(loc, ValueRange{subOp});

    return success();
  }

  // Handle `while` loops.
  LogicalResult visit(const slang::ast::WhileLoopStatement &stmt) {
    // Create the while op.
    auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    OpBuilder::InsertionGuard guard(builder);

    // In the "before" region, check that the condition holds.
    builder.createBlock(&whileOp.getBefore());
    auto cond = context.convertRvalueExpression(stmt.cond);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<mlir::scf::ConditionOp>(loc, cond, ValueRange{});

    // In the "after" region, generate the loop body.
    builder.createBlock(&whileOp.getAfter());
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    builder.create<mlir::scf::YieldOp>(loc);

    return success();
  }

  // Handle `do ... while` loops.
  LogicalResult visit(const slang::ast::DoWhileLoopStatement &stmt) {
    // Create the while op.
    auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    OpBuilder::InsertionGuard guard(builder);

    // In the "before" region, generate the loop body and check that the
    // condition holds.
    builder.createBlock(&whileOp.getBefore());
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    auto cond = context.convertRvalueExpression(stmt.cond);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<mlir::scf::ConditionOp>(loc, cond, ValueRange{});

    // Generate an empty "after" region.
    builder.createBlock(&whileOp.getAfter());
    builder.create<mlir::scf::YieldOp>(loc);

    return success();
  }

  // Handle `forever` loops.
  LogicalResult visit(const slang::ast::ForeverLoopStatement &stmt) {
    // Create the while op.
    auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    OpBuilder::InsertionGuard guard(builder);

    // In the "before" region, return true for the condition.
    builder.createBlock(&whileOp.getBefore());
    auto cond = builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
    builder.create<mlir::scf::ConditionOp>(loc, cond, ValueRange{});

    // In the "after" region, generate the loop body.
    builder.createBlock(&whileOp.getAfter());
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    builder.create<mlir::scf::YieldOp>(loc);

    return success();
  }

  // Handle timing control.
  LogicalResult visit(const slang::ast::TimedStatement &stmt) {
    if (failed(context.convertTimingControl(stmt.timing)))
      return failure();
    if (failed(context.convertStatement(stmt.stmt)))
      return failure();
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
