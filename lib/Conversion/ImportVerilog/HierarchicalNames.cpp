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

    if (isCompileTimeConstant(expr.symbol.kind))
      return;

    auto hierName = builder.getStringAttr(expr.symbol.name);
    const slang::ast::InstanceBodySymbol *parentInstBody = nullptr;

    // Record `name` on `body`, or add this symbol as an alias if another
    // instance already recorded the same path.
    auto addHierPath = [&](const slang::ast::InstanceBodySymbol *body,
                           mlir::StringAttr name,
                           slang::ast::ArgumentDirection direction) {
      for (auto &path : context.hierPaths[body])
        if (path.hierName == name) {
          if (!llvm::any_of(path.valueSyms, [&](auto &alias) {
                return alias.first == &expr.symbol;
              }))
            path.valueSyms.push_back({&expr.symbol, currentInstBody});
          return;
        }
      context.hierPaths[body].push_back(
          HierPathInfo{name, {}, direction, {{&expr.symbol, currentInstBody}}});
    };

    auto *tempInstBody = currentInstBody;
    while (tempInstBody) {
      tempInstBody = tempInstBody->parentInstance->getParentScope()
                         ->getContainingInstance();
      if (tempInstBody == outermostInstBody) {
        // The value flows upward: expose it as an output port on every module
        // boundary between the symbol and the outermost module.
        SmallVector<const slang::ast::InstanceSymbol *> instChain;
        for (auto *b = currentInstBody; b && b != outermostInstBody;
             b = b->parentInstance->getParentScope()->getContainingInstance())
          instChain.push_back(b->parentInstance);
        // Map each level of the chain to the body that module conversion
        // will actually visit, descending from the outermost module: slang's
        // getCanonicalBody only relates instances elaborated within the same
        // parent body, so a body forced into existence by this reference
        // does not defer to its equivalent in the canonical sibling subtree.
        // The by-name lookup only happens at levels that left the canonical
        // subtree, references on the canonical path skip it entirely.
        SmallVector<const slang::ast::InstanceBodySymbol *> canonChain(
            instChain.size());
        const slang::ast::InstanceBodySymbol *parentCanon = outermostInstBody;
        for (size_t i = instChain.size(); i-- > 0;) {
          const slang::ast::InstanceSymbol *inst = instChain[i];
          if (inst->getParentScope()->getContainingInstance() != parentCanon)
            if (auto *resolved = parentCanon->find(inst->name))
              if (auto *resolvedInst =
                      resolved->as_if<slang::ast::InstanceSymbol>())
                inst = resolvedInst;
          canonChain[i] = getCanonicalBody(*inst);
          parentCanon = canonChain[i];
        }
        for (size_t i = 0; i < instChain.size(); ++i) {
          addHierPath(canonChain[i], hierName,
                      slang::ast::ArgumentDirection::Out);
          hierName = builder.getStringAttr(
              instChain[i]->name + llvm::Twine(".") + hierName.getValue());
        }
        return;
      }
    }

    // The value flows downward: route it as an input port from the common
    // ancestor down to the referencing module.
    std::function<void(const slang::ast::InstanceBodySymbol *)>
        collectHierarchicalPaths = [&](auto sym) {
          addHierPath(getCanonicalBody(*sym->parentInstance), hierName,
                      slang::ast::ArgumentDirection::In);
          parentInstBody =
              sym->parentInstance->getParentScope()->getContainingInstance();
          if (parentInstBody && parentInstBody != currentInstBody)
            collectHierarchicalPaths(parentInstBody);
        };
    hierName = builder.getStringAttr(currentInstBody->parentInstance->name +
                                     llvm::Twine(".") + hierName.getValue());
    collectHierarchicalPaths(outermostInstBody);
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
