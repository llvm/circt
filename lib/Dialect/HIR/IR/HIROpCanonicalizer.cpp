//=========- HIROpCanonicalizer.cpp - Canonicalize HIR Ops ----------------===//
//
// This file implements op canonicalizers for HIR dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR//helper.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
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

LogicalResult BusSendOp::canonicalize(BusSendOp op,
                                      ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult BusRecvOp::canonicalize(BusRecvOp op,
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

LogicalResult IfOp::canonicalize(IfOp op, mlir::PatternRewriter &rewriter) {
  LogicalResult result = splitOffsetIntoSeparateOp(op, rewriter);

  auto constantOp =
      dyn_cast_or_null<mlir::arith::ConstantOp>(op.condition().getDefiningOp());
  if (!constantOp)
    return result;

  int condition = constantOp.value().dyn_cast<IntegerAttr>().getInt();
  BlockAndValueMapping operandMap;
  SmallVector<Value> yieldedValues;
  operandMap.map(op.getRegionTimeVar(), op.tstart());
  Region &selectedRegion = condition ? op.if_region() : op.else_region();
  for (Operation &operation : selectedRegion.front()) {
    if (auto yieldOp = dyn_cast<hir::YieldOp>(operation)) {
      assert(yieldOp.operands().size() == op.results().size());
      for (Value value : yieldOp.operands())
        yieldedValues.push_back(helper::lookupOrOriginal(operandMap, value));
      continue;
    }
    rewriter.clone(operation, operandMap);
  }
  if (yieldedValues.size() > 0)
    rewriter.replaceOp(op, yieldedValues);
  return success();
}

LogicalResult NextIterOp::canonicalize(NextIterOp op,
                                       mlir::PatternRewriter &rewriter) {

  return splitOffsetIntoSeparateOp(op, rewriter);
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
  // If its a chain of latch ops then remove this op.
  if (auto latchOp = dyn_cast_or_null<hir::LatchOp>(input.getDefiningOp())) {
    if (this->tstart() == latchOp.tstart() &&
        this->offset() == latchOp.offset())
      return input;
  }
  // If prev op is a constant then this latch op is unnecessary.
  if (auto constantOp =
          dyn_cast_or_null<mlir::arith::ConstantOp>(input.getDefiningOp()))
    return input;
  return {};
}
