//===- Statements.cpp - Slang statement conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "llvm/ADT/ScopeExit.h"

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

  bool isTerminated() const { return !builder.getInsertionBlock(); }
  void setTerminated() { builder.clearInsertionPoint(); }

  Block &createBlock() {
    assert(builder.getInsertionBlock());
    auto block = std::make_unique<Block>();
    block->insertAfter(builder.getInsertionBlock());
    return *block.release();
  }

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
    for (auto *stmt : stmts.list) {
      if (isTerminated()) {
        auto loc = context.convertLocation(stmt->sourceRange);
        mlir::emitWarning(loc, "unreachable code");
        break;
      }
      if (failed(context.convertStatement(*stmt)))
        return failure();
    }
    return success();
  }

  // Inline `begin ... end` blocks into the parent.
  LogicalResult visit(const slang::ast::BlockStatement &stmt) {
    return context.convertStatement(stmt.body);
  }

  // Handle expression statements.
  LogicalResult visit(const slang::ast::ExpressionStatement &stmt) {
    auto value = context.convertRvalueExpression(stmt.expr);
    if (!value)
      return failure();

    // Expressions like calls to void functions return a dummy value that has no
    // uses. If the returned value is trivially dead, remove it.
    if (auto *defOp = value.getDefiningOp())
      if (isOpTriviallyDead(defOp))
        defOp->erase();

    return success();
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

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    builder.create<cf::CondBranchOp>(loc, allConds, &trueBlock,
                                     falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (failed(context.convertStatement(stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &exitBlock);

    // Generate the false branch if present.
    if (stmt.ifFalse) {
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      if (!isTerminated())
        builder.create<cf::BranchOp>(loc, &exitBlock);
    }

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle case statements.
  LogicalResult visit(const slang::ast::CaseStatement &caseStmt) {
    using slang::ast::CaseStatementCondition;
    auto caseExpr = context.convertRvalueExpression(caseStmt.expr);
    if (!caseExpr)
      return failure();

    // Check each case individually. This currently ignores the `unique`,
    // `unique0`, and `priority` modifiers which would allow for additional
    // optimizations.
    auto &exitBlock = createBlock();

    for (const auto &item : caseStmt.items) {
      // Create the block that will contain the main body of the expression.
      // This is where any of the comparisons will branch to if they match.
      auto &matchBlock = createBlock();

      // The SV standard requires expressions to be checked in the order
      // specified by the user, and for the evaluation to stop as soon as the
      // first matching expression is encountered.
      for (const auto *expr : item.expressions) {
        auto value = context.convertRvalueExpression(*expr);
        if (!value)
          return failure();
        auto itemLoc = value.getLoc();

        // Generate the appropriate equality operator.
        Value cond;
        switch (caseStmt.condition) {
        case CaseStatementCondition::Normal:
          cond = builder.create<moore::CaseEqOp>(itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::WildcardXOrZ:
          cond = builder.create<moore::CaseXZEqOp>(itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::WildcardJustZ:
          cond = builder.create<moore::CaseZEqOp>(itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::Inside:
          mlir::emitError(loc, "unsupported set membership case statement");
          return failure();
        }
        cond = builder.create<moore::ConversionOp>(itemLoc, builder.getI1Type(),
                                                   cond);

        // If the condition matches, branch to the match block. Otherwise
        // continue checking the next expression in a new block.
        auto &nextBlock = createBlock();
        builder.create<mlir::cf::CondBranchOp>(itemLoc, cond, &matchBlock,
                                               &nextBlock);
        builder.setInsertionPointToEnd(&nextBlock);
      }

      // The current block is the fall-through after all conditions have been
      // checked and nothing matched. Move the match block up before this point
      // to make the IR easier to read.
      matchBlock.moveBefore(builder.getInsertionBlock());

      // Generate the code for this item's statement in the match block.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&matchBlock);
      if (failed(context.convertStatement(*item.stmt)))
        return failure();
      if (!isTerminated()) {
        auto loc = context.convertLocation(item.stmt->sourceRange);
        builder.create<mlir::cf::BranchOp>(loc, &exitBlock);
      }
    }

    // Generate the default case if present.
    if (caseStmt.defaultCase)
      if (failed(context.convertStatement(*caseStmt.defaultCase)))
        return failure();
    if (!isTerminated())
      builder.create<mlir::cf::BranchOp>(loc, &exitBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle `for` loops.
  LogicalResult visit(const slang::ast::ForLoopStatement &stmt) {
    // Generate the initializers.
    for (auto *initExpr : stmt.initializers)
      if (!context.convertRvalueExpression(*initExpr))
        return failure();

    // Create the blocks for the loop condition, body, step, and exit.
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    builder.create<cf::BranchOp>(loc, &checkBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertRvalueExpression(*stmt.stopExpr);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<cf::CondBranchOp>(loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &stepBlock);

    // Generate the step expressions.
    builder.setInsertionPointToEnd(&stepBlock);
    for (auto *stepExpr : stmt.steps)
      if (!context.convertRvalueExpression(*stepExpr))
        return failure();
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &checkBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle `repeat` loops.
  LogicalResult visit(const slang::ast::RepeatLoopStatement &stmt) {
    auto count = context.convertRvalueExpression(stmt.count);
    if (!count)
      return failure();

    // Create the blocks for the loop condition, body, step, and exit.
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    auto currentCount = checkBlock.addArgument(count.getType(), count.getLoc());
    builder.create<cf::BranchOp>(loc, &checkBlock, count);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = builder.createOrFold<moore::BoolCastOp>(loc, currentCount);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<cf::CondBranchOp>(loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &stepBlock);

    // Decrement the current count and branch back to the check block.
    builder.setInsertionPointToEnd(&stepBlock);
    auto one = builder.create<moore::ConstantOp>(
        count.getLoc(), cast<moore::IntType>(count.getType()), 1);
    Value nextCount =
        builder.create<moore::SubOp>(count.getLoc(), currentCount, one);
    builder.create<cf::BranchOp>(loc, &checkBlock, nextCount);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle `while` and `do-while` loops.
  LogicalResult createWhileLoop(const slang::ast::Expression &condExpr,
                                const slang::ast::Statement &bodyStmt,
                                bool atLeastOnce) {
    // Create the blocks for the loop condition, body, and exit.
    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    builder.create<cf::BranchOp>(loc, atLeastOnce ? &bodyBlock : &checkBlock);
    if (atLeastOnce)
      bodyBlock.moveBefore(&checkBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&checkBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertRvalueExpression(condExpr);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<cf::CondBranchOp>(loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(bodyStmt)))
      return failure();
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &checkBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult visit(const slang::ast::WhileLoopStatement &stmt) {
    return createWhileLoop(stmt.cond, stmt.body, false);
  }

  LogicalResult visit(const slang::ast::DoWhileLoopStatement &stmt) {
    return createWhileLoop(stmt.cond, stmt.body, true);
  }

  // Handle `forever` loops.
  LogicalResult visit(const slang::ast::ForeverLoopStatement &stmt) {
    // Create the blocks for the loop body and exit.
    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    builder.create<cf::BranchOp>(loc, &bodyBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&bodyBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &bodyBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle timing control.
  LogicalResult visit(const slang::ast::TimedStatement &stmt) {
    return context.convertTimingControl(stmt.timing, stmt.stmt);
  }

  // Handle return statements.
  LogicalResult visit(const slang::ast::ReturnStatement &stmt) {
    if (stmt.expr) {
      auto expr = context.convertRvalueExpression(*stmt.expr);
      if (!expr)
        return failure();
      builder.create<mlir::func::ReturnOp>(loc, expr);
    } else {
      builder.create<mlir::func::ReturnOp>(loc);
    }
    setTerminated();
    return success();
  }

  // Handle continue statements.
  LogicalResult visit(const slang::ast::ContinueStatement &stmt) {
    if (context.loopStack.empty())
      return mlir::emitError(loc,
                             "cannot `continue` without a surrounding loop");
    builder.create<cf::BranchOp>(loc, context.loopStack.back().continueBlock);
    setTerminated();
    return success();
  }

  // Handle break statements.
  LogicalResult visit(const slang::ast::BreakStatement &stmt) {
    if (context.loopStack.empty())
      return mlir::emitError(loc, "cannot `break` without a surrounding loop");
    builder.create<cf::BranchOp>(loc, context.loopStack.back().breakBlock);
    setTerminated();
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
  assert(builder.getInsertionBlock());
  auto loc = convertLocation(stmt.sourceRange);
  return stmt.visit(StmtVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)
