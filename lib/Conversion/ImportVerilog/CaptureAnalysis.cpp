//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CaptureAnalysis.h"
#include "slang/ast/ASTVisitor.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace slang::ast;
using namespace circt;
using namespace circt::ImportVerilog;

/// Check whether `var` is local to `func`. Walk up from the variable's parent
/// scope; if we reach `func` before hitting another function boundary, the
/// variable is local.
static bool isLocalToFunction(const ValueSymbol &var,
                              const SubroutineSymbol &func) {
  for (const Scope *scope = var.getParentScope(); scope;
       scope = scope->asSymbol().getParentScope()) {
    if (&scope->asSymbol() == &func)
      return true;
    if (scope->asSymbol().kind == SymbolKind::Subroutine)
      return false;
  }
  return false;
}

/// Workaround for a slang deficiency: when accessing a member of a virtual
/// interface (e.g., `vif.data`), slang resolves the entire dotted path during
/// name lookup and produces a `NamedValueExpression` that directly references
/// the signal symbol inside the interface's `InstanceBody`. Unlike struct
/// fields and class properties, which produce a `MemberAccessExpression`, there
/// is no syntactic indication on the expression that this was a member
/// projection.
///
/// This check matches variables that live inside an interface instance body.
/// A `NamedValueExpression` referencing such a symbol is the result of slang's
/// virtual interface member resolution, not a genuine variable capture. This is
/// expected to be fixed upstream in slang.
///
/// See https://github.com/MikePopoloski/slang/discussions/1770
static bool isVirtualInterfaceMemberAccess(const ValueSymbol &var) {
  // Walk up from the variable to find the nearest enclosing InstanceBody.
  for (const Scope *scope = var.getParentScope(); scope;
       scope = scope->asSymbol().getParentScope()) {
    auto *body = scope->asSymbol().as_if<InstanceBodySymbol>();
    if (!body)
      continue;
    return body->getDefinition().definitionKind == DefinitionKind::Interface;
  }
  return false;
}

/// Check whether `var` is a global variable. Walk up from the variable's parent
/// scope; if we hit a function or instance body, it's not global. Otherwise
/// (package, compilation unit, root) it is.
static bool isGlobalVariable(const ValueSymbol &var) {
  for (const Scope *scope = var.getParentScope(); scope;
       scope = scope->asSymbol().getParentScope()) {
    switch (scope->asSymbol().kind) {
    case SymbolKind::Subroutine:
    case SymbolKind::InstanceBody:
      return false;
    default:
      break;
    }
  }
  return true;
}

namespace {

/// Walk the entire AST to collect captured variables and the call graph for
/// each function. Uses slang's `ASTVisitor` with both statement and expression
/// visiting enabled so that we recurse into all function bodies.
struct CaptureWalker
    : public ASTVisitor<CaptureWalker, /*VisitStatements=*/true,
                        /*VisitExpressions=*/true> {

  /// The function whose body we are currently inside, or nullptr if we are at
  /// a scope outside any function.
  const SubroutineSymbol *currentFunc = nullptr;

  /// Captured variables per function.
  CaptureMap capturedVars;

  /// Inverse call graph: maps each callee to the set of callers that call it.
  /// Used to propagate captures from callees to their callers. Uses MapVector
  /// for deterministic iteration order during propagation.
  MapVector<const SubroutineSymbol *,
            SmallSetVector<const SubroutineSymbol *, 4>>
      callers;

  /// When we enter a function body, record it as the current function and
  /// recurse into its members and body statements.
  void handle(const SubroutineSymbol &func) {
    llvm::SaveAndRestore guard(currentFunc, &func);
    visitDefault(func);
  }

  /// When we see a named value reference inside a function, check if it needs
  /// to be captured.
  void handle(const NamedValueExpression &expr) {
    if (!currentFunc)
      return;

    auto &var = expr.symbol;

    // Class properties are accessed through `this`, not captured.
    if (var.kind == SymbolKind::ClassProperty)
      return;

    // Function arguments are local by definition.
    if (var.kind == SymbolKind::FormalArgument)
      return;

    // Compile-time constants are materialized inline and don't need capturing.
    // Slang monomorphizes modules and classes per parameterization, so within
    // any given elaborated scope these are fixed values.
    if (var.kind == SymbolKind::Parameter ||
        var.kind == SymbolKind::EnumValue || var.kind == SymbolKind::Genvar ||
        var.kind == SymbolKind::Specparam)
      return;

    // Only capture variables that are non-local and non-global.
    if (isLocalToFunction(var, *currentFunc) || isGlobalVariable(var))
      return;

    // Work around a slang deficiency where virtual interface member accesses
    // are resolved to NamedValueExpressions referencing symbols inside the
    // interface's instance body, indistinguishable from direct variable
    // references. See isVirtualInterfaceMemberAccess for details.
    if (isVirtualInterfaceMemberAccess(var))
      return;

    capturedVars[currentFunc].insert(&var);
  }

  /// Record call graph edges when we see a function call.
  void handle(const CallExpression &expr) {
    if (currentFunc)
      if (auto *const *callee =
              std::get_if<const SubroutineSymbol *>(&expr.subroutine))
        callers[*callee].insert(currentFunc);
    visitDefault(expr);
  }

  /// Propagate captures transitively through the call graph. For each callee
  /// that has captures, push each captured variable upward through all
  /// transitive callers using a worklist. A captured variable is only
  /// propagated to a caller if it is not local to that caller.
  void propagateCaptures() {
    using WorkItem = std::pair<const SubroutineSymbol *, const ValueSymbol *>;
    SmallSetVector<WorkItem, 16> worklist;

    for (auto &[func, _] : callers) {
      // Check if this function captures any variables. Nothing to do if it
      // doesn't.
      auto it = capturedVars.find(func);
      if (it == capturedVars.end())
        continue;

      // Prime the worklist with the captured variables.
      for (auto *var : it->second)
        worklist.insert({func, var});

      // Push each captured variables to the func's callers transitively.
      while (!worklist.empty()) {
        auto [func, cap] = worklist.pop_back_val();
        auto callersIt = callers.find(func);
        if (callersIt == callers.end())
          continue;
        for (auto *caller : callersIt->second)
          if (!isLocalToFunction(*cap, *caller))
            if (capturedVars[caller].insert(cap))
              worklist.insert({caller, cap});
      }
    }
  }
};

} // namespace

CaptureMap ImportVerilog::analyzeFunctionCaptures(const RootSymbol &root) {
  CaptureWalker walker;
  root.visit(walker);
  walker.propagateCaptures();
  return std::move(walker.capturedVars);
}
