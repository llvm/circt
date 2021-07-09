//=========- HIROpCanonicalizer.cpp - Canonicalize HIR Ops ----------------===//
//
// This file implements op canonicalizers for HIR dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR//helper.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace hir;
using namespace llvm;

template <typename OPTYPE>
static LogicalResult splitOffsetIntoSeparateOp(OPTYPE op,
                                               PatternRewriter &rewriter) {
  auto *context = rewriter.getContext();
  if (!op.offset())
    return success();
  if (op.offset().getValue() == 0)
    return success();

  Value tstart = rewriter.create<hir::TimeOp>(
      op.getLoc(), helper::getTimeType(context), op.tstart(), op.offsetAttr());

  op.tstartMutable().assign(tstart);
  op.offsetAttr(rewriter.getI64IntegerAttr(0));
  return success();
}

LogicalResult LoadOp::canonicalize(LoadOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult StoreOp::canonicalize(StoreOp op,
                                    ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult SendOp::canonicalize(SendOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult RecvOp::canonicalize(RecvOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
LogicalResult AddIOp::canonicalize(AddIOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
LogicalResult SubIOp::canonicalize(SubIOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
LogicalResult MulIOp::canonicalize(MulIOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
LogicalResult AddFOp::canonicalize(AddFOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
LogicalResult SubFOp::canonicalize(SubFOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
LogicalResult MulFOp::canonicalize(MulFOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
LogicalResult ForOp::canonicalize(ForOp op, PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult UnrollForOp::canonicalize(UnrollForOp op,
                                        PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult CallOp::canonicalize(CallOp op, PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult YieldOp::canonicalize(YieldOp op,
                                    ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}
