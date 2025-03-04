//===- Statements.cpp - Slang statement conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/SystemSubroutine.h"
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

  LogicalResult recursiveForeach(const slang::ast::ForeachLoopStatement &stmt,
                                 uint32_t level) {
    // find current dimension we are operate.
    const auto &loopDim = stmt.loopDims[level];
    if (!loopDim.range.has_value()) {
      emitError(loc) << "dynamic loop variable is unsupported";
    }
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    const auto &iter = loopDim.loopVar;
    auto type = context.convertType(*iter->getDeclaredType());
    if (!type)
      return failure();

    Value initial = builder.create<moore::ConstantOp>(
        loc, cast<moore::IntType>(type), loopDim.range->lower());

    // Create loop varirable in this dimension
    Value varOp = builder.create<moore::VariableOp>(
        loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
        builder.getStringAttr(iter->name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         iter, varOp);

    builder.create<cf::BranchOp>(loc, &checkBlock);
    builder.setInsertionPointToEnd(&checkBlock);

    // When the loop variable is greater than the upper bound, goto exit
    auto upperBound = builder.create<moore::ConstantOp>(
        loc, cast<moore::IntType>(type), loopDim.range->upper());

    auto var = builder.create<moore::ReadOp>(loc, varOp);
    Value cond = builder.create<moore::SleOp>(loc, var, upperBound);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
    builder.create<cf::CondBranchOp>(loc, cond, &bodyBlock, &exitBlock);

    builder.setInsertionPointToEnd(&bodyBlock);

    // find next dimension in this foreach statement, it finded then recuersive
    // resolve, else perform body statement
    bool hasNext = false;
    for (uint32_t nextLevel = level + 1; nextLevel < stmt.loopDims.size();
         nextLevel++) {
      if (stmt.loopDims[nextLevel].loopVar) {
        if (failed(recursiveForeach(stmt, nextLevel)))
          return failure();
        hasNext = true;
        break;
      }
    }

    if (!hasNext) {
      if (failed(context.convertStatement(stmt.body)))
        return failure();
    }
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &stepBlock);

    builder.setInsertionPointToEnd(&stepBlock);

    // add one to loop variable
    var = builder.create<moore::ReadOp>(loc, varOp);
    auto one =
        builder.create<moore::ConstantOp>(loc, cast<moore::IntType>(type), 1);
    auto postValue = builder.create<moore::AddOp>(loc, var, one).getResult();
    builder.create<moore::BlockingAssignOp>(loc, varOp, postValue);
    builder.create<cf::BranchOp>(loc, &checkBlock);

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
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
    // Special handling for calls to system tasks that return no result value.
    if (const auto *call = stmt.expr.as_if<slang::ast::CallExpression>()) {
      if (const auto *info =
              std::get_if<slang::ast::CallExpression::SystemCallInfo>(
                  &call->subroutine)) {
        auto handled = visitSystemCall(stmt, *call, *info);
        if (failed(handled))
          return failure();
        if (handled == true)
          return success();
      }
    }

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
      initial = context.convertRvalueExpression(*init, {}, true);
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

  LogicalResult visit(const slang::ast::ForeachLoopStatement &stmt) {
    for (uint32_t level = 0; level < stmt.loopDims.size(); level++) {
      if (stmt.loopDims[level].loopVar)
        return recursiveForeach(stmt, level);
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

  // Handle immediate assertion statements.
  LogicalResult visit(const slang::ast::ImmediateAssertionStatement &stmt) {
    auto cond = context.convertRvalueExpression(stmt.cond);
    cond = context.convertToBool(cond);
    if (!cond)
      return failure();

    // Handle assertion statements that don't have an action block.
    if (stmt.ifTrue && stmt.ifTrue->as_if<slang::ast::EmptyStatement>()) {
      auto defer = moore::DeferAssert::Immediate;
      if (stmt.isFinal)
        defer = moore::DeferAssert::Final;
      else if (stmt.isDeferred)
        defer = moore::DeferAssert::Observed;

      switch (stmt.assertionKind) {
      case slang::ast::AssertionKind::Assert:
        builder.create<moore::AssertOp>(loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::Assume:
        builder.create<moore::AssumeOp>(loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::CoverProperty:
        builder.create<moore::CoverOp>(loc, defer, cond, StringAttr{});
        return success();
      default:
        break;
      }
      mlir::emitError(loc) << "unsupported immediate assertion kind: "
                           << slang::ast::toString(stmt.assertionKind);
      return failure();
    }

    // Regard assertion statements with an action block as the "if-else".
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    builder.create<cf::CondBranchOp>(loc, cond, &trueBlock,
                                     falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (stmt.ifTrue && failed(context.convertStatement(*stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      builder.create<cf::BranchOp>(loc, &exitBlock);

    if (stmt.ifFalse) {
      // Generate the false branch if present.
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

  /// Handle the subset of system calls that return no result value. Return
  /// true if the called system task could be handled, false otherwise. Return
  /// failure if an error occurred.
  FailureOr<bool>
  visitSystemCall(const slang::ast::ExpressionStatement &stmt,
                  const slang::ast::CallExpression &expr,
                  const slang::ast::CallExpression::SystemCallInfo &info) {
    const auto &subroutine = *info.subroutine;
    auto args = expr.arguments();

    // Simulation Control Tasks

    if (subroutine.name == "$stop") {
      createFinishMessage(args.size() >= 1 ? args[0] : nullptr);
      builder.create<moore::StopBIOp>(loc);
      return true;
    }

    if (subroutine.name == "$finish") {
      createFinishMessage(args.size() >= 1 ? args[0] : nullptr);
      builder.create<moore::FinishBIOp>(loc, 0);
      builder.create<moore::UnreachableOp>(loc);
      setTerminated();
      return true;
    }

    if (subroutine.name == "$exit") {
      // Calls to `$exit` from outside a `program` are ignored. Since we don't
      // yet support programs, there is nothing to do here.
      // TODO: Fix this once we support programs.
      return true;
    }

    // Display and Write Tasks (`$display[boh]?` or `$write[boh]?`)

    // Check for a `$display` or `$write` prefix.
    bool isDisplay = false;     // display or write
    bool appendNewline = false; // display
    StringRef remainingName = subroutine.name;
    if (remainingName.consume_front("$display")) {
      isDisplay = true;
      appendNewline = true;
    } else if (remainingName.consume_front("$write")) {
      isDisplay = true;
    }

    // Check for optional `b`, `o`, or `h` suffix indicating default format.
    using moore::IntFormat;
    IntFormat defaultFormat = IntFormat::Decimal;
    if (isDisplay && !remainingName.empty()) {
      if (remainingName == "b")
        defaultFormat = IntFormat::Binary;
      else if (remainingName == "o")
        defaultFormat = IntFormat::Octal;
      else if (remainingName == "h")
        defaultFormat = IntFormat::HexLower;
      else
        isDisplay = false;
    }

    if (isDisplay) {
      auto message =
          context.convertFormatString(args, loc, defaultFormat, appendNewline);
      if (failed(message))
        return failure();
      if (*message == Value{})
        return true;
      builder.create<moore::DisplayBIOp>(loc, *message);
      return true;
    }

    // Severity Tasks
    using moore::Severity;
    std::optional<Severity> severity;
    if (subroutine.name == "$info")
      severity = Severity::Info;
    else if (subroutine.name == "$warning")
      severity = Severity::Warning;
    else if (subroutine.name == "$error")
      severity = Severity::Error;
    else if (subroutine.name == "$fatal")
      severity = Severity::Fatal;

    if (severity) {
      // The `$fatal` task has an optional leading verbosity argument.
      const slang::ast::Expression *verbosityExpr = nullptr;
      if (severity == Severity::Fatal && args.size() >= 1) {
        verbosityExpr = args[0];
        args = args.subspan(1);
      }

      // Handle the string formatting.
      auto message = context.convertFormatString(args, loc);
      if (failed(message))
        return failure();
      if (*message == Value{})
        *message = builder.create<moore::FormatLiteralOp>(loc, "");

      builder.create<moore::SeverityBIOp>(loc, *severity, *message);

      // Handle the `$fatal` case which behaves like a `$finish`.
      if (severity == Severity::Fatal) {
        createFinishMessage(verbosityExpr);
        builder.create<moore::FinishBIOp>(loc, 1);
        builder.create<moore::UnreachableOp>(loc);
        setTerminated();
      }
      return true;
    }

    // Give up on any other system tasks. These will be tried again as an
    // expression later.
    return false;
  }

  /// Create the optional diagnostic message print for finish-like ops.
  void createFinishMessage(const slang::ast::Expression *verbosityExpr) {
    unsigned verbosity = 1;
    if (verbosityExpr) {
      auto value =
          context.evaluateConstant(*verbosityExpr).integer().as<unsigned>();
      assert(value && "Slang guarantees constant verbosity parameter");
      verbosity = *value;
    }
    if (verbosity == 0)
      return;
    builder.create<moore::FinishMessageBIOp>(loc, verbosity > 1);
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
