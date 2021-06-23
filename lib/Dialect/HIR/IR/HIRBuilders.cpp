#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
using namespace circt;
using namespace hir;

Block *ForOp::addEntryBlock(MLIRContext *context, Type inductionVarTy) {
  Block *entry = new Block;
  entry->addArgument(inductionVarTy);              // induction var
  entry->addArgument(hir::TimeType::get(context)); // iter time
  getLoopBody().push_back(entry);
  return entry;
}
void ForOp::beginRegion(OpBuilder &builder) {
  builder.setInsertionPointToStart(&getLoopBody().front());
}
void ForOp::endRegion(OpBuilder &builder) {
  builder.create<hir::TerminatorOp>(builder.getUnknownLoc());
  builder.setInsertionPointAfter(*this);
}

Block *UnrollForOp::addEntryBlock(MLIRContext *context) {
  Block *entry = new Block;
  entry->addArgument(IndexType::get(context)); // induction var
  entry->addArgument(TimeType::get(context));  // iter time
  getLoopBody().push_back(entry);
  return entry;
}

void UnrollForOp::beginRegion(OpBuilder &builder) {
  builder.setInsertionPointToStart(&getLoopBody().front());
}
void UnrollForOp::endRegion(OpBuilder &builder) {
  builder.create<hir::TerminatorOp>(builder.getUnknownLoc());
  builder.setInsertionPointAfter(*this);
}
