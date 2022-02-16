//===- BackedgeBuilder.cpp - Support for building backedges ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provide support for building backedges.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;

Backedge::Backedge(mlir::Operation *op) : value(op->getResult(0)) {}

void Backedge::setValue(mlir::Value newValue) {
  assert(value.getType() == newValue.getType());
  assert(!set && "backedge already set to a value!");
  value.replaceAllUsesWith(newValue);
  set = true;
}

BackedgeBuilder::~BackedgeBuilder() {
  for (Operation *op : edges) {
    auto users = op->getUsers();
    assert(users.empty() && "Backedge still in use");
    if (rewriter)
      rewriter->eraseOp(op);
    else
      op->erase();
  }
}

Backedge::operator mlir::Value() { return value; }

BackedgeBuilder::BackedgeBuilder(OpBuilder &builder, Location loc)
    : builder(builder), rewriter(nullptr), loc(loc) {
  loc.getContext()->allowUnregisteredDialects();
}
BackedgeBuilder::BackedgeBuilder(PatternRewriter &rewriter, Location loc)
    : builder(rewriter), rewriter(&rewriter), loc(loc) {
  loc.getContext()->allowUnregisteredDialects();
}
Backedge BackedgeBuilder::get(Type t) {
  OperationState s(loc, "TemporaryBackedge");
  s.addTypes(t);
  auto op = builder.createOperation(s);
  edges.push_back(op);
  return Backedge(op);
}
