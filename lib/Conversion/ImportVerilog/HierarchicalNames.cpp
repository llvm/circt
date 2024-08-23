//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"

using namespace circt;
using namespace ImportVerilog;

namespace {
struct HierPathValueExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  // Such as `sub.a`, the `sub` is the outermost module for the hierarchical
  // variable `a`.
  const slang::ast::Symbol &outermostModule;

  HierPathValueExprVisitor(Context &context, Location loc,
                           const slang::ast::Symbol &outermostModule)
      : context(context), loc(loc), builder(context.builder),
        outermostModule(outermostModule) {}

  // Handle hierarchical values
  LogicalResult visit(const slang::ast::HierarchicalValueExpression &expr) {
    auto *currentInstBody =
        expr.symbol.getParentScope()->getContainingInstance();

    // Like module Foo; int a; "Foo.a;" endmodule.
    // Ignore "Foo.a" invoked by this module itself.
    if (currentInstBody ==
        outermostModule.as_if<slang::ast::InstanceBodySymbol>())
      return success();

    auto hierName = builder.getStringAttr(expr.symbol.name);
    const slang::ast::InstanceBodySymbol *parentInstBody = nullptr;

    // Collect the hierarchical names that are added to the port list.
    std::function<void(const slang::ast::InstanceBodySymbol *)>
        collectHierarchicalPaths = [&](auto sym) {
          // Here we use "sameHierPaths" to avoid collecting the repeat
          // hierarchical names on the same path.
          if (!context.sameHierPaths.contains(hierName) ||
              !context.hierPaths.contains(sym)) {
            context.hierPaths[sym].push_back(
                HierPathInfo{hierName, {}, &expr.symbol});
            context.sameHierPaths.insert(hierName);
          }

          // The instance body must have instance header. So only estimate the
          // parent scope of the instance header whether exists.
          auto *scope = sym->parentInstance->getParentScope();
          if (scope)
            parentInstBody = scope->getContainingInstance();

          // Such as `Top.sub.a`, we will record`sub.a` for the port list,
          if (currentInstBody !=
              outermostModule.as_if<slang::ast::InstanceBodySymbol>()) {
            hierName =
                builder.getStringAttr(sym->parentInstance->name +
                                      llvm::Twine(".") + hierName.getValue());

            if (parentInstBody)
              collectHierarchicalPaths(parentInstBody);
          }
        };
    collectHierarchicalPaths(currentInstBody);
    return success();
  }

  /// TODO:Skip all others.
  /// But we should output a warning to display which symbol had been skipped.
  /// However, to ensure we can test smoothly, we didn't do that.
  template <typename T>
  LogicalResult visit(T &&node) {
    return success();
  }

  LogicalResult visitInvalid(const slang::ast::Expression &expr) {
    mlir::emitError(loc, "invalid expression");
    return failure();
  }
};
} // namespace

LogicalResult
Context::collectHierarchicalValues(const slang::ast::Expression &expr,
                                   const slang::ast::Symbol &outermostModule) {
  auto loc = convertLocation(expr.sourceRange);
  return expr.visit(HierPathValueExprVisitor(*this, loc, outermostModule));
}

/// Traverse the instance body.
namespace {
struct InstBodyVisitor {
  Context &context;
  Location loc;

  InstBodyVisitor(Context &context, Location loc)
      : context(context), loc(loc) {}

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    return context.traverseInstanceBody(instNode.body);
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto &outermostModule = varNode.getParentScope()->asSymbol();
    if (const auto *init = varNode.getInitializer())
      if (failed(context.collectHierarchicalValues(*init, outermostModule)))
        return failure();
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto &outermostModule = netNode.getParentScope()->asSymbol();
    if (const auto *init = netNode.getInitializer())
      if (failed(context.collectHierarchicalValues(*init, outermostModule)))
        return failure();
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    const auto &expr =
        assignNode.getAssignment().as<slang::ast::AssignmentExpression>();

    // Such as `sub.a`, the `sub` is the outermost module for the hierarchical
    // variable `a`.
    auto &outermostModule = assignNode.getParentScope()->asSymbol();
    if (expr.left().hasHierarchicalReference())
      if (failed(
              context.collectHierarchicalValues(expr.left(), outermostModule)))
        return failure();

    if (expr.right().hasHierarchicalReference())
      if (failed(
              context.collectHierarchicalValues(expr.right(), outermostModule)))
        return failure();

    return success();
  }

  /// TODO:Skip all others.
  /// But we should output a warning to display which symbol had been skipped.
  /// However, to ensure we can test smoothly, we didn't do that.
  template <typename T>
  LogicalResult visit(T &&node) {
    return success();
  }
};
}; // namespace

LogicalResult Context::traverseInstanceBody(const slang::ast::Symbol &symbol) {
  if (auto *instBodySymbol = symbol.as_if<slang::ast::InstanceBodySymbol>())
    for (auto &member : instBodySymbol->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(InstBodyVisitor(*this, loc))))
        return failure();
    }
  return success();
}
