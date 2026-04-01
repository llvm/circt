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
struct HierPathValueExprVisitor
    : public slang::ast::ASTVisitor<HierPathValueExprVisitor,
                                    /*VisitStatements=*/false,
                                    /*VisitExpressions=*/true> {
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
  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    auto *currentInstBody =
        expr.symbol.getParentScope()->getContainingInstance();
    auto *outermostInstBody =
        outermostModule.as_if<slang::ast::InstanceBodySymbol>();

    // Like module Foo; int a; Foo.a; endmodule.
    // Ignore "Foo.a" invoked by this module itself.
    if (currentInstBody == outermostInstBody)
      return;

    auto hierName = builder.getStringAttr(expr.symbol.name);
    const slang::ast::InstanceBodySymbol *parentInstBody = nullptr;

    // Collect hierarchical names that are added to the port list.
    std::function<void(const slang::ast::InstanceBodySymbol *, bool)>
        collectHierarchicalPaths = [&](auto sym, bool isUpward) {
          // Here we use "sameHierPaths" to avoid collecting the repeat
          // hierarchical names on the same path.
          if (!context.sameHierPaths.contains(hierName) ||
              !context.hierPaths.contains(sym)) {
            context.hierPaths[sym].push_back(
                HierPathInfo{hierName,
                             {},
                             isUpward ? slang::ast::ArgumentDirection::Out
                                      : slang::ast::ArgumentDirection::In,
                             &expr.symbol});
            context.sameHierPaths.insert(hierName);
          }

          // Iterate up from the current instance body symbol until meeting the
          // outermost module.
          parentInstBody =
              sym->parentInstance->getParentScope()->getContainingInstance();
          if (!parentInstBody)
            return;

          if (isUpward) {
            // Avoid collecting hierarchical names into the outermost module.
            if (parentInstBody && parentInstBody != outermostInstBody) {
              hierName =
                  builder.getStringAttr(sym->parentInstance->name +
                                        llvm::Twine(".") + hierName.getValue());
              collectHierarchicalPaths(parentInstBody, isUpward);
            }
          } else {
            if (parentInstBody && parentInstBody != currentInstBody)
              collectHierarchicalPaths(parentInstBody, isUpward);
          }
        };

    // Determine whether hierarchical names are upward or downward.
    auto *tempInstBody = currentInstBody;
    while (tempInstBody) {
      tempInstBody = tempInstBody->parentInstance->getParentScope()
                         ->getContainingInstance();
      if (tempInstBody == outermostInstBody) {
        collectHierarchicalPaths(currentInstBody, true);
        return;
      }
    }

    hierName = builder.getStringAttr(currentInstBody->parentInstance->name +
                                     llvm::Twine(".") + hierName.getValue());
    collectHierarchicalPaths(outermostInstBody, false);
  }
};
} // namespace

void Context::collectHierarchicalValues(
    const slang::ast::Expression &expr,
    const slang::ast::Symbol &outermostModule) {
  auto loc = convertLocation(expr.sourceRange);
  HierPathValueExprVisitor visitor(*this, loc, outermostModule);
  expr.visit(visitor);
}

/// Traverse the instance body.
namespace {
struct InstBodyVisitor
    : public slang::ast::ASTVisitor<InstBodyVisitor,
                                    /*VisitStatements=*/true,
                                    /*VisitExpressions=*/true> {

  InstBodyVisitor(Context &context, const slang::ast::Symbol &outermostModule)
      : context(context), outermostModule(outermostModule) {}

  void handle(const slang::ast::InstanceSymbol &instNode) {
    context.traverseInstanceBody(instNode.body);
  }

  void handle(const slang::ast::Expression &expr) {
    context.collectHierarchicalValues(expr, outermostModule);
  }

  Context &context;
  const slang::ast::Symbol &outermostModule;
};

} // namespace

void Context::traverseInstanceBody(const slang::ast::Symbol &symbol) {
  if (auto *instBodySymbol = symbol.as_if<slang::ast::InstanceBodySymbol>())
    for (auto &member : instBodySymbol->members()) {
      auto &outermostModule = member.getParentScope()->asSymbol();
      InstBodyVisitor visitor(*this, outermostModule);
      member.visit(visitor);
    }
}
