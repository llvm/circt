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
  InstBodyVisitor(Context &context, const slang::ast::Symbol &outermostModule,
                  DenseSet<StringAttr> &sameHierPaths)
      : context(context), outermostModule(outermostModule),
        sameHierPaths(sameHierPaths) {}

  void handle(const slang::ast::InstanceSymbol &instNode) {
    context.traverseInstanceBody(instNode.body, sameHierPaths);
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
          // Here we use "sameHierPaths" to avoid collecting the repeat
          // hierarchical names on the same path.
          if (!sameHierPaths.contains(hierName) ||
              !context.hierPaths.contains(sym)) {
            context.hierPaths[sym].push_back(
                HierPathInfo{hierName,
                             {},
                             isUpward ? slang::ast::ArgumentDirection::Out
                                      : slang::ast::ArgumentDirection::In,
                             {&expr.symbol}});
            sameHierPaths.insert(hierName);
          } else {
            // The path already exists (dedup hit), but this may be a
            // different instance resolving to a different symbol object
            // for the same logical variable. Add it as an alias.
            for (auto &existing : context.hierPaths[sym]) {
              if (existing.hierName == hierName) {
                if (!llvm::is_contained(existing.valueSyms, &expr.symbol))
                  existing.valueSyms.push_back(&expr.symbol);
                break;
              }
            }
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

  // Deduplication set shared across the entire traversal tree rooted at
  // the top-level traverseInstanceBody call.
  DenseSet<StringAttr> &sameHierPaths;
};

} // namespace

void Context::traverseInstanceBody(const slang::ast::Symbol &symbol) {
  // Top-level entry point: create a fresh deduplication set that is shared
  // across all recursive traversals of this instance tree. This prevents
  // cross-hierarchy contamination between independent top-level instances.
  DenseSet<StringAttr> sameHierPaths;
  traverseInstanceBody(symbol, sameHierPaths);
}

void Context::traverseInstanceBody(const slang::ast::Symbol &symbol,
                                   DenseSet<StringAttr> &sameHierPaths) {
  if (auto *instBodySymbol = symbol.as_if<slang::ast::InstanceBodySymbol>())
    for (auto &member : instBodySymbol->members()) {
      auto &outermostModule = member.getParentScope()->asSymbol();
      InstBodyVisitor visitor(*this, outermostModule, sameHierPaths);
      member.visit(visitor);
    }
}
