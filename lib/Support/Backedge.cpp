//===- Backedge.cpp - Support classes for building backedges ----*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Backedge.h"

using namespace circt;
using namespace mlir;

// StringRef Backedge::getOperationName() { return "TemporaryBackedge"; }
// void Backedge::build(mlir::OpBuilder &b, mlir::OperationState &state,
//                      mlir::Type resultType) {
//   state.addTypes(resultType);
// }

Backedge::Backedge(BackedgeBuilder &parent, mlir::Operation *op)
    : parent(parent), op(op) {}

BackedgeBuilder::~BackedgeBuilder() {
  for (Operation *op : edges) {
    auto users = op->getUsers();
    if (users.empty())
      rewriter.eraseOp(op);
    else
      for (Operation *user : op->getUsers())
        user->emitOpError("Still using temporary backedge");
  }
}

Backedge::operator mlir::Value() { return op->getResult(0); }

BackedgeBuilder::BackedgeBuilder(PatternRewriter &rewriter, Location loc)
    : rewriter(rewriter), loc(loc) {}

Backedge BackedgeBuilder::operator()(Type t) {
  OperationState s(loc, "TemporaryBackedge");
  s.addTypes(t);
  auto op = rewriter.createOperation(s);
  edges.push_back(op);
  return Backedge(*this, op);
}
