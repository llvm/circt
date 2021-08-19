//=========- HIROpCanonicalizer.cpp - Canonicalize HIR Ops ----------------===//
//
// This file implements op canonicalizers for HIR dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR//helper.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace hir;
using namespace llvm;

template <typename OPTYPE>
static LogicalResult splitOffsetIntoSeparateOp(OPTYPE op,
                                               PatternRewriter &rewriter) {
  auto *context = rewriter.getContext();
  if (!op.offset())
    return failure();
  if (op.offset().getValue() == 0)
    return failure();

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

LogicalResult CallInstanceOp::canonicalize(CallInstanceOp op,
                                           PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult YieldOp::canonicalize(YieldOp op,
                                    ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

// LogicalResult TimeOp::canonicalize(TimeOp op,
//                                   ::mlir::PatternRewriter &rewriter) {
// ImplicitLocOpBuilder builder(op.getLoc(), op);
//// If there is a chain of TimeOps then replace it with one TimeOp.
// auto timeVar = op.timevar();
// auto delay = op.delay();
// while (auto timeOp = dyn_cast_or_null<TimeOp>(timeVar.getDefiningOp())) {
//  timeVar = timeOp.timevar();
//  delay += timeOp.delay();
//}
// op.timevarMutable().assign(timeVar);
// op.delayAttr(builder.getI64IntegerAttr(delay));
// return success();
//}

LogicalResult MemrefExtractOp::canonicalize(MemrefExtractOp op,
                                            ::mlir::PatternRewriter &rewriter) {
  auto uses = op.res().getUses();
  bool hasUseInCallOp;
  for (auto use : uses) {
    if (use.)
  }
  return success();
}

OpFoldResult TimeOp::fold(ArrayRef<Attribute> operands) {
  auto timeVar = this->timevar();
  auto delay = this->delay();
  if (delay == 0)
    return timeVar;
  while (auto timeOp = dyn_cast_or_null<TimeOp>(timeVar.getDefiningOp())) {
    timeVar = timeOp.timevar();
    delay += timeOp.delay();
  }
  this->timevarMutable().assign(timeVar);
  this->delayAttr(helper::getI64IntegerAttr(this->getContext(), delay));

  return {};
}

OpFoldResult LatchOp::fold(ArrayRef<Attribute> operands) {
  auto input = this->input();
  if (auto latchOp = dyn_cast_or_null<hir::LatchOp>(input.getDefiningOp())) {
    if (this->tstart() == latchOp.tstart() &&
        this->offset() == latchOp.offset())
      return input;
  }
  return {};
}
