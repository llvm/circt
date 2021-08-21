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

LogicalResult CallOp::canonicalize(CallOp op, PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult CallInstanceOp::canonicalize(CallInstanceOp op,
                                           PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult MemrefExtractOp::canonicalize(MemrefExtractOp op,
                                            ::mlir::PatternRewriter &rewriter) {
  auto uses = op.res().getUses();
  // Check and return if there are any uses in CallOp.
  for (auto &use : uses) {
    if (dyn_cast<hir::CallOp>(use.getOwner())) {
      return failure();
    }
  }

  // For each use, update the operand and the port.
  for (auto &use : uses) {
    if (auto useOp = dyn_cast<hir::LoadOp>(use.getOwner())) {
      if (!useOp.port().hasValue())
        continue;
      uint64_t mappedPortNum = useOp.port().getValue();
      uint64_t origPortNum =
          op.portNums()[mappedPortNum].dyn_cast<IntegerAttr>().getInt();
      useOp.portAttr(rewriter.getI64IntegerAttr(origPortNum));
    } else if (auto useOp = dyn_cast<hir::StoreOp>(use.getOwner())) {
      if (!useOp.port().hasValue())
        continue;
      uint64_t mappedPortNum = useOp.port().getValue();
      uint64_t origPortNum =
          op.portNums()[mappedPortNum].dyn_cast<IntegerAttr>().getInt();
      useOp.portAttr(rewriter.getI64IntegerAttr(origPortNum));
    } else if (auto useOp = dyn_cast<hir::MemrefExtractOp>(use.getOwner())) {
      return op.emitWarning("Could not canonicalize MemrefExtractOp. "
                            "Unsupported operation in use list.");
    }
  }
  rewriter.replaceOp(op, op.mem());
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
