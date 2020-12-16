//===- Backedge.cpp - Support classes for building backedges ----*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Backedge.h"

#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace mlir;

Backedge::Backedge(mlir::Operation *op) : value(op->getResult(0)) {}

void Backedge::operator=(mlir::Value newValue) {
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

Backedge BackedgeBuilder::operator()(Type t) {
  OperationState s(loc, "TemporaryBackedge");
  s.addTypes(t);
  auto op = rewriter.createOperation(s);
  edges.push_back(op);
  return Backedge(op);
}
