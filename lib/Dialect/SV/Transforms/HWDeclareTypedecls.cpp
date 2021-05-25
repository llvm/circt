//===- HWDeclareTypedecls.cpp - HW DeclareTypedecls Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace circt;
using namespace hw;

namespace {
struct HWDeclareTypedeclsPass
    : public sv::HWDeclareTypedeclsBase<HWDeclareTypedeclsPass> {
  void runOnOperation() override;

private:
  void visitOperation(Operation *op);
  void visitType(Operation *op, Type type);
  ModuleOp parentModule;
  OpBuilder *builder;
};
} // namespace

void HWDeclareTypedeclsPass::visitOperation(Operation *op) {
  for (auto type : op->getOperandTypes())
    visitType(op, type);
  for (auto type : op->getResultTypes())
    visitType(op, type);
  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks())
      for (auto type : block.getArgumentTypes())
        visitType(op, type);
}

void HWDeclareTypedeclsPass::visitType(Operation *op, Type type) {
  // Look for any TypeAliasTypes.
  auto alias = type.dyn_cast<TypeAliasType>();
  if (!alias)
    return;

  // If no scope exists, create one in the parent module.
  StringRef scopeName = alias.getRef().getRootReference();
  TypeScopeOp scope = parentModule.lookupSymbol<TypeScopeOp>(scopeName);
  if (!scope) {
    builder->setInsertionPointToStart(parentModule.getBody());
    scope = builder->create<TypeScopeOp>(op->getLoc(), scopeName);
    scope.body().emplaceBlock();
  }

  // If no typedecl exists, create one in the scope.
  StringRef symbolName = alias.getRef().getLeafReference();
  TypedeclOp typeDecl = scope.lookupSymbol<TypedeclOp>(symbolName);
  if (!typeDecl) {
    // TODO: The insertion point should generate type aliases of type aliases in
    // an order that respects def-before-use, or fails on mutually recursive
    // type aliases. For now, insert at the end as we go.
    builder->setInsertionPointToEnd(scope.getBodyBlock());
    builder->create<TypedeclOp>(op->getLoc(), symbolName, alias.getInnerType(),
                                StringAttr());
    return;
  }

  // If a typedecl exists with a different type for the same name, emit an
  // error.
  if (typeDecl.type() != alias.getInnerType())
    op->emitOpError("redefining type definition for ") << typeDecl;
}

void HWDeclareTypedeclsPass::runOnOperation() {
  parentModule = getOperation();
  OpBuilder tmpBuilder(parentModule);
  builder = &tmpBuilder;
  parentModule.walk([&](Operation *op) { visitOperation(op); });
}

std::unique_ptr<mlir::Pass> circt::sv::createHWDeclareTypedeclsPass() {
  return std::make_unique<HWDeclareTypedeclsPass>();
}
