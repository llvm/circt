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
  for (auto &use : value.getUses())
    use.set(newValue);
  value = newValue;
}

BackedgeBuilder::~BackedgeBuilder() {
  for (Operation *op : edges) {
    auto users = op->getUsers();
    assert(users.empty() && "Backedge still in use");
    rewriter.eraseOp(op);
  }
}

Backedge::operator mlir::Value() { return value; }

BackedgeBuilder::BackedgeBuilder(PatternRewriter &rewriter, Location loc)
    : rewriter(rewriter), loc(loc) {}

Backedge BackedgeBuilder::get(Type t) {
  OperationState s(loc, "TemporaryBackedge");
  s.addTypes(t);
  auto op = rewriter.createOperation(s);
  edges.push_back(op);
  return Backedge(op);
}
