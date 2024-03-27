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

  // Handle if statements.
  LogicalResult visit(const slang::ast::ConditionalStatement &stmt) {
    // Generate the condition. There may be multiple conditions linked with the
    // `&&&` operator.
    Value allConds;
    for (const auto &condition : stmt.conditions) {
      if (condition.pattern)
        return mlir::emitError(loc,
                               "match patterns in if conditions not supported");
      auto cond = context.convertExpression(*condition.expr);
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

  // Handle `for` loops.
  LogicalResult visit(const slang::ast::ForLoopStatement &stmt) {
    if (!stmt.loopVars.empty())
      return mlir::emitError(loc,
                             "variables in for loop initializer not supported");

    // Generate the initializers.
    for (auto *initExpr : stmt.initializers)
      if (!context.convertExpression(*initExpr))
        return failure();

    // Create the while op.
    auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    OpBuilder::InsertionGuard guard(builder);

    // In the "before" region, check that the condition holds.
    builder.createBlock(&whileOp.getBefore());
    auto cond = context.convertExpression(*stmt.stopExpr);
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
      if (!context.convertExpression(*stepExpr))
        return failure();
    builder.create<mlir::scf::YieldOp>(loc);

    return success();
  }

  // Handle `repeat` loops.
  LogicalResult visit(const slang::ast::RepeatLoopStatement &stmt) {
    // Create the while op and feed in the repeat count as the initial counter
    // value.
    auto count = context.convertExpression(stmt.count);
    if (!count)
      return failure();
    auto type = count.getType();
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
    auto cond = context.convertExpression(stmt.cond);
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
    auto cond = context.convertExpression(stmt.cond);
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

  // Unroll ForeachLoop into nested for loops, parse the body in the innermost
  // layer, and break out to the outermost layer.
  LogicalResult visit(const slang::ast::ForeachLoopStatement &foreachStmt) {

    // Store unrolled loops in Dimension order
    SmallVector<mlir::scf::WhileOp> loops;
    auto type = moore::IntType::get(context.getContext(), moore::IntType::Int);
    auto step = builder.create<moore::ConstantOp>(loc, type, 1);
    for (auto &dimension : foreachStmt.loopDims) {
      // Skip null dimension loopVar between i,j in foreach(array[i, ,j,k])
      if (!dimension.loopVar)
        continue;

      // lower bound
      builder.create<moore::ConstantOp>(loc, type, dimension.range->lower());
      // uppper bound

      auto ub = builder.create<moore::ConstantOp>(loc, type,
                                                  dimension.range->upper());
      auto index = builder.create<moore::ConstantOp>(loc, type,
                                                     dimension.range->lower());

      // insert nested whileOp in after region
      if (!loops.empty())
        builder.setInsertionPointToEnd(loops.back().getAfterBody());
      auto whileOp = builder.create<mlir::scf::WhileOp>(loc, TypeRange{type},
                                                        ValueRange{index});
      // OpBuilder::InsertionGuard guard(builder);

      // The before-region of the WhileOp.
      Block *before = builder.createBlock(&whileOp.getBefore(), {}, type, loc);
      builder.setInsertionPointToEnd(before);

      // Check if index overflows
      Value cond;
      if (dimension.range->lower() <= dimension.range->upper()) {
        cond = builder.create<moore::LtOp>(loc, index, ub);
      } else {
        cond = builder.create<moore::GeOp>(loc, index, ub);
      }

      cond =
          builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);
      builder.create<mlir::scf::ConditionOp>(loc, cond, before->getArguments());

      // Remember the iterator variable in each loops
      context.valueSymbols.insert(dimension.loopVar, before->getArgument(0));

      // The after-region of the WhileOp.
      Block *after = builder.createBlock(&whileOp.getAfter(), {}, type, loc);
      builder.setInsertionPointToStart(after);
      loops.push_back(whileOp);
    }

    // gen body in innermost block
    if (failed(context.convertStatement(foreachStmt.body)))
      return failure();

    // gen index iteration in the end
    for (auto it = foreachStmt.loopDims.rbegin();
         it != foreachStmt.loopDims.rend(); ++it) {
      if (!it->loopVar)
        continue;
      auto whileOp = loops.back();
      if (!whileOp.getAfter().hasOneBlock()) {
        mlir::emitError(loc, "no block in while after region");
        return failure();
      }

      builder.setInsertionPointToEnd(whileOp.getAfterBody());
      auto index = whileOp.getAfterArguments().back();
      Value afterIndex;
      if (it->range->lower() <= it->range->upper()) {
        // step ++
        afterIndex = builder.create<moore::AddOp>(loc, index, step);
      } else {
        // step --
        afterIndex = builder.create<moore::SubOp>(loc, index, step);
      }

      builder.create<mlir::scf::YieldOp>(
          loc, mlir::SmallVector<Value, 1>{afterIndex});
      builder.setInsertionPointAfter(whileOp);
      loops.pop_back();
    }

    return success();
  }

  // Handle timing control statements.
  LogicalResult visit(const slang::ast::TimedStatement &stmt) {
    if (failed(context.visitTimingControl(&stmt.timing)))
      return failure();
    return context.convertStatement(stmt.stmt);
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
