//===- RTLDeclareTypedefs.cpp - RTL DeclareTypedefs Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include <mlir/IR/Builders.h>

using namespace circt;
using namespace rtl;
using namespace sv;

namespace {
struct RTLDeclareTypedefsPass
    : public sv::RTLDeclareTypedefsBase<RTLDeclareTypedefsPass> {
  void runOnOperation() override;

private:
  void visitOperation(Operation *op);
  void visitType(Operation *op, Type type);
  ModuleOp parentModule;
  OpBuilder *builder;
};
} // namespace

void RTLDeclareTypedefsPass::visitOperation(Operation *op) {
  for (auto type : op->getOperandTypes())
    visitType(op, type);
  for (auto type : op->getResultTypes())
    visitType(op, type);
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto type : block.getArgumentTypes())
        visitType(op, type);
      block.walk([&](Operation *op) { visitOperation(op); });
    }
  }
}

void RTLDeclareTypedefsPass::visitType(Operation *op, Type type) {
  // Look for any TypeAliasTypes.
  auto alias = type.dyn_cast<TypeAliasType>();
  if (!alias)
    return;

  // If no typedef exists, create one in the parent module.
  auto typeDef = parentModule.lookupSymbol<TypeDefOp>(alias.getName());
  if (!typeDef) {
    builder->setInsertionPointToStart(parentModule.getBody());
    builder->create<TypeDefOp>(op->getLoc(), alias.getName(), alias.getInner());
    return;
  }

  // If a typedef exists with a different type for the same name, emit an error.
  if (typeDef.type() != alias.getInner())
    op->emitOpError("redefining type definition for ")
        << typeDef.getOperation();
}

void RTLDeclareTypedefsPass::runOnOperation() {
  parentModule = getOperation();
  OpBuilder tmpBuilder(parentModule);
  builder = &tmpBuilder;
  parentModule.walk([&](Operation *op) { visitOperation(op); });
}

std::unique_ptr<mlir::Pass> circt::sv::createRTLDeclareTypedefsPass() {
  return std::make_unique<RTLDeclareTypedefsPass>();
}
