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

  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    auto builder = context.builder;
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
