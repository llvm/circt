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
                                    slang::ast::VisitFlags::AllGood> {
  InstBodyVisitor(
      Context &context, const slang::ast::Symbol &outermostModule,
      DenseSet<const slang::ast::InstanceBodySymbol *> &visitedBodies)
      : context(context), outermostModule(outermostModule),
        visitedBodies(visitedBodies) {}

  void handle(const slang::ast::InstanceSymbol &instNode) {
    traverseInstanceBody(context, instNode, visitedBodies);
    // Also visit port connection expressions to find hier refs used as
    // port arguments (e.g., .in_val(b_inst.local_val)).
    for (auto *conn : instNode.getPortConnections())
      if (auto *connExpr = conn->getExpression())
        connExpr->visit(*this);
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

    // References resolved via an interface port (e.g. `bus.member` where `bus`
    // is an `Iface.modport` port) are not cross-instance hierarchical accesses;
    // they are handled by the interface port lowering machinery in
    // Structure.cpp. Recording them here would add a spurious hierPath input
    // to the module signature that nothing fills in at the instance site.
    if (expr.ref.isViaIfacePort())
      return;

    auto hierName = builder.getStringAttr(expr.symbol.name);
    const slang::ast::InstanceBodySymbol *parentInstBody = nullptr;

    // Collect hierarchical names that are added to the port list.
    std::function<void(const slang::ast::InstanceBodySymbol *, bool)>
        collectHierarchicalPaths = [&](auto sym, bool isUpward) {
          // Check if this path already exists globally for this module
          HierPathInfo *existing = nullptr;
          if (context.hierPaths.contains(sym)) {
            for (auto &path : context.hierPaths[sym]) {
              if (path.hierName == hierName) {
                existing = &path;
                break;
              }
            }
          }

          if (!existing) {
            context.hierPaths[sym].push_back(
                HierPathInfo{hierName,
                             {},
                             isUpward ? slang::ast::ArgumentDirection::Out
                                      : slang::ast::ArgumentDirection::In,
                             {&expr.symbol}});
          } else {
            // The path already exists, but this may be a different instance
            // resolving to a different symbol object. Add as an alias.
            if (!llvm::is_contained(existing->valueSyms, &expr.symbol))
              existing->valueSyms.push_back(&expr.symbol);
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
  DenseSet<const slang::ast::InstanceBodySymbol *> &visitedBodies;

  static void traverseInstanceBody(
      Context &context, const slang::ast::InstanceSymbol &symbol,
      DenseSet<const slang::ast::InstanceBodySymbol *> &visitedBodies) {
    const slang::ast::InstanceBodySymbol *body = getCanonicalBody(symbol);
    if (visitedBodies.insert(body).second) {
      for (auto &member : body->members()) {
        auto &outermostModule = member.getParentScope()->asSymbol();
        InstBodyVisitor visitor(context, outermostModule, visitedBodies);
        member.visit(visitor);
      }
    }
  }
};

} // namespace

void Context::traverseInstanceBody(const slang::ast::InstanceSymbol &symbol) {
  // Top-level entry point: create a fresh visitedBodies set to prevent
  // infinite recursion and to skip identical module bodies.
  DenseSet<const slang::ast::InstanceBodySymbol *> visitedBodies;
  InstBodyVisitor::traverseInstanceBody(*this, symbol, visitedBodies);
}
