//===- Statements.cpp - Slang statement conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/syntax/AllSyntax.h"
#include "llvm/ADT/ScopeExit.h"
#include <slang/ast/SFormat.h>

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
    if (!loopDim.range.has_value())
      return mlir::emitError(loc) << "dynamic loop variable is unsupported";
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

    Value initial = moore::ConstantOp::create(
        builder, loc, cast<moore::IntType>(type), loopDim.range->lower());

    // Create loop varirable in this dimension
    Value varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
        builder.getStringAttr(iter->name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         iter, varOp);

    cf::BranchOp::create(builder, loc, &checkBlock);
    builder.setInsertionPointToEnd(&checkBlock);

    // When the loop variable is greater than the upper bound, goto exit
    auto upperBound = moore::ConstantOp::create(
        builder, loc, cast<moore::IntType>(type), loopDim.range->upper());

    auto var = moore::ReadOp::create(builder, loc, varOp);
    Value cond = moore::SleOp::create(builder, loc, var, upperBound);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

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
      cf::BranchOp::create(builder, loc, &stepBlock);

    builder.setInsertionPointToEnd(&stepBlock);

    // add one to loop variable
    var = moore::ReadOp::create(builder, loc, varOp);
    auto one =
        moore::ConstantOp::create(builder, loc, cast<moore::IntType>(type), 1);
    auto postValue = moore::AddOp::create(builder, loc, var, one).getResult();
    moore::BlockingAssignOp::create(builder, loc, varOp, postValue);
    cf::BranchOp::create(builder, loc, &checkBlock);

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
      if (failed(context.convertStatement(*stmt))) {
        context.dbgs(stmt->syntax->sourceRange().start())
            << "Failed to convert statement " << stmt->syntax->toString();
        return failure();
      }
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
        if (failed(handled)) {
          context.dbgs(stmt.sourceRange.start())
              << "Failed to convert system call " << stmt.syntax->toString();
          return failure();
        }
        if (handled == true)
          return success();
      }
      context.dbgs(stmt.sourceRange.start())
          << "Assuming statement " << stmt.syntax->toString()
          << " is not a system task";
    }

    auto value = context.convertRvalueExpression(stmt.expr);
    if (!value) {
      context.dbgs(stmt.sourceRange.start())
          << "Failed to convert expression statement as RValue "
          << stmt.expr.syntax->toString();
      return failure();
    }
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
      initial = context.convertRvalueExpression(*init, type);
      if (!initial)
        return failure();
    }

    // Collect local temporary variables.
    auto varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
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
        allConds = moore::AndOp::create(builder, loc, allConds, cond);
      else
        allConds = cond;
    }
    assert(allConds && "slang guarantees at least one condition");
    allConds = moore::ToBuiltinBoolOp::create(builder, loc, allConds);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, allConds, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (failed(context.convertStatement(stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    // Generate the false branch if present.
    if (stmt.ifFalse) {
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
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

  /// Handle case statements.
  LogicalResult visit(const slang::ast::CaseStatement &caseStmt) {
    using slang::ast::AttributeSymbol;
    using slang::ast::CaseStatementCondition;
    auto caseExpr = context.convertRvalueExpression(caseStmt.expr);
    if (!caseExpr)
      return failure();

    // Check each case individually. This currently ignores the `unique`,
    // `unique0`, and `priority` modifiers which would allow for additional
    // optimizations.
    auto &exitBlock = createBlock();
    Block *lastMatchBlock = nullptr;
    SmallVector<moore::FVIntegerAttr> itemConsts;

    for (const auto &item : caseStmt.items) {
      // Create the block that will contain the main body of the expression.
      // This is where any of the comparisons will branch to if they match.
      auto &matchBlock = createBlock();
      lastMatchBlock = &matchBlock;

      // The SV standard requires expressions to be checked in the order
      // specified by the user, and for the evaluation to stop as soon as the
      // first matching expression is encountered.
      for (const auto *expr : item.expressions) {
        auto value = context.convertRvalueExpression(*expr);
        if (!value)
          return failure();
        auto itemLoc = value.getLoc();

        // Take note if the expression is a constant.
        auto maybeConst = value;
        while (isa_and_nonnull<moore::ConversionOp, moore::IntToLogicOp,
                               moore::LogicToIntOp>(maybeConst.getDefiningOp()))
          maybeConst = maybeConst.getDefiningOp()->getOperand(0);
        if (auto defOp = maybeConst.getDefiningOp<moore::ConstantOp>())
          itemConsts.push_back(defOp.getValueAttr());

        // Generate the appropriate equality operator.
        Value cond;
        switch (caseStmt.condition) {
        case CaseStatementCondition::Normal:
          cond = moore::CaseEqOp::create(builder, itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::WildcardXOrZ:
          cond = moore::CaseXZEqOp::create(builder, itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::WildcardJustZ:
          cond = moore::CaseZEqOp::create(builder, itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::Inside:
          mlir::emitError(loc, "unsupported set membership case statement");
          return failure();
        }
        cond = moore::ToBuiltinBoolOp::create(builder, itemLoc, cond);

        // If the condition matches, branch to the match block. Otherwise
        // continue checking the next expression in a new block.
        auto &nextBlock = createBlock();
        mlir::cf::CondBranchOp::create(builder, itemLoc, cond, &matchBlock,
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
        mlir::cf::BranchOp::create(builder, loc, &exitBlock);
      }
    }

    const auto caseStmtAttrs = context.compilation.getAttributes(caseStmt);
    const bool hasFullCaseAttr =
        llvm::find_if(caseStmtAttrs, [](const AttributeSymbol *attr) {
          return attr->name == "full_case";
        }) != caseStmtAttrs.end();

    // Check if the case statement looks exhaustive assuming two-state values.
    // We use this information to work around a common bug in input Verilog
    // where a case statement enumerates all possible two-state values of the
    // case expression, but forgets to deal with cases involving X and Z bits in
    // the input.
    //
    // Once the core dialects start supporting four-state values we may want to
    // tuck this behind an import option that is on by default, since it does
    // not preserve semantics.
    auto twoStateExhaustive = false;
    if (auto intType = dyn_cast<moore::IntType>(caseExpr.getType());
        intType && intType.getWidth() < 32 &&
        itemConsts.size() == (1 << intType.getWidth())) {
      // Sort the constants by value.
      llvm::sort(itemConsts, [](auto a, auto b) {
        return a.getValue().getRawValue().ult(b.getValue().getRawValue());
      });

      // Ensure that every possible value of the case expression is present. Do
      // this by starting at 0 and iterating over all sorted items. Each item
      // must be the previous item + 1. At the end, the addition must exactly
      // overflow and take us back to zero.
      auto nextValue = FVInt::getZero(intType.getWidth());
      for (auto value : itemConsts) {
        if (value.getValue() != nextValue)
          break;
        nextValue += 1;
      }
      twoStateExhaustive = nextValue.isZero();
    }

    // If the case statement is exhaustive assuming two-state values, don't
    // generate the default case. Instead, branch to the last match block. This
    // will essentially make the last case item the "default".
    //
    // Alternatively, if the case statement has an (* full_case *) attribute
    // but no default case, it indicates that the developer has intentionally
    // covered all known possible values. Hence, the last match block is
    // treated as the implicit "default" case.
    if ((twoStateExhaustive || (hasFullCaseAttr && !caseStmt.defaultCase)) &&
        lastMatchBlock &&
        caseStmt.condition == CaseStatementCondition::Normal) {
      mlir::cf::BranchOp::create(builder, loc, lastMatchBlock);
    } else {
      // Generate the default case if present.
      if (caseStmt.defaultCase)
        if (failed(context.convertStatement(*caseStmt.defaultCase)))
          return failure();
      if (!isTerminated())
        mlir::cf::BranchOp::create(builder, loc, &exitBlock);
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
    cf::BranchOp::create(builder, loc, &checkBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertRvalueExpression(*stmt.stopExpr);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    // Generate the step expressions.
    builder.setInsertionPointToEnd(&stepBlock);
    for (auto *stepExpr : stmt.steps)
      if (!context.convertRvalueExpression(*stepExpr))
        return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &checkBlock);

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
    cf::BranchOp::create(builder, loc, &checkBlock, count);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = builder.createOrFold<moore::BoolCastOp>(loc, currentCount);
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    // Decrement the current count and branch back to the check block.
    builder.setInsertionPointToEnd(&stepBlock);
    auto one = moore::ConstantOp::create(
        builder, count.getLoc(), cast<moore::IntType>(count.getType()), 1);
    Value nextCount =
        moore::SubOp::create(builder, count.getLoc(), currentCount, one);
    cf::BranchOp::create(builder, loc, &checkBlock, nextCount);

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
    cf::BranchOp::create(builder, loc, atLeastOnce ? &bodyBlock : &checkBlock);
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
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(bodyStmt)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &checkBlock);

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
    cf::BranchOp::create(builder, loc, &bodyBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&bodyBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &bodyBlock);

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
      mlir::func::ReturnOp::create(builder, loc, expr);
    } else {
      mlir::func::ReturnOp::create(builder, loc);
    }
    setTerminated();
    return success();
  }

  // Handle continue statements.
  LogicalResult visit(const slang::ast::ContinueStatement &stmt) {
    if (context.loopStack.empty())
      return mlir::emitError(loc,
                             "cannot `continue` without a surrounding loop");
    cf::BranchOp::create(builder, loc, context.loopStack.back().continueBlock);
    setTerminated();
    return success();
  }

  // Handle break statements.
  LogicalResult visit(const slang::ast::BreakStatement &stmt) {
    if (context.loopStack.empty())
      return mlir::emitError(loc, "cannot `break` without a surrounding loop");
    cf::BranchOp::create(builder, loc, context.loopStack.back().breakBlock);
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
        moore::AssertOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::Assume:
        moore::AssumeOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::CoverProperty:
        moore::CoverOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      default:
        break;
      }
      mlir::emitError(loc) << "unsupported immediate assertion kind: "
                           << slang::ast::toString(stmt.assertionKind);
      return failure();
    }

    // Regard assertion statements with an action block as the "if-else".
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, cond, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (stmt.ifTrue && failed(context.convertStatement(*stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    if (stmt.ifFalse) {
      // Generate the false branch if present.
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
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

  // Handle concurrent assertion statements.
  LogicalResult visit(const slang::ast::ConcurrentAssertionStatement &stmt) {
    auto loc = context.convertLocation(stmt.sourceRange);
    auto property = context.convertAssertionExpression(stmt.propertySpec, loc);
    if (!property)
      return failure();

    // Handle assertion statements that don't have an action block.
    if (stmt.ifTrue && stmt.ifTrue->as_if<slang::ast::EmptyStatement>()) {
      switch (stmt.assertionKind) {
      case slang::ast::AssertionKind::Assert:
        verif::AssertOp::create(builder, loc, property, Value(), StringAttr{});
        return success();
      case slang::ast::AssertionKind::Assume:
        verif::AssumeOp::create(builder, loc, property, Value(), StringAttr{});
        return success();
      default:
        break;
      }
      mlir::emitError(loc) << "unsupported concurrent assertion kind: "
                           << slang::ast::toString(stmt.assertionKind);
      return failure();
    }

    mlir::emitError(loc)
        << "concurrent assertion statements with action blocks "
           "are not supported yet";
    return failure();
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
      moore::StopBIOp::create(builder, loc);
      return true;
    }

    if (subroutine.name == "$finish") {
      createFinishMessage(args.size() >= 1 ? args[0] : nullptr);
      moore::FinishBIOp::create(builder, loc, 0);
      moore::UnreachableOp::create(builder, loc);
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
      moore::DisplayBIOp::create(builder, loc, *message);
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
        *message = moore::FormatLiteralOp::create(builder, loc, "");

      moore::SeverityBIOp::create(builder, loc, *severity, *message);

      // Handle the `$fatal` case which behaves like a `$finish`.
      if (severity == Severity::Fatal) {
        createFinishMessage(verbosityExpr);
        moore::FinishBIOp::create(builder, loc, 1);
        moore::UnreachableOp::create(builder, loc);
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
    moore::FinishMessageBIOp::create(builder, loc, verbosity > 1);
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
