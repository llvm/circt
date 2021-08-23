//=========- LoopSimplification.cpp - Lower memref type--------------------===//
//
// This file implements loop simplification pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include <iostream>
using namespace circt;
using namespace hir;

class SimplifyLoopPass : public hir::SimplifyLoopBase<SimplifyLoopPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(ForOp);
};

void SimplifyLoopPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::ForOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}
/// Convert ForOp into a WhileOp.
LogicalResult SimplifyLoopPass::visitOp(ForOp forOp) {
  // The condition var = cmpi "ult",lb,ub: i4.
  // The loop induction & condition vars are calculated using a CallOp.
  // Example:
  // %cond_next, %iv_next = hir.call @HIR_for_op_cond_and_iv(%iv,%lb,%ub,%step)
  //    at %t {WIDTH_IV = 4}

  if (forOp->hasAttr("unroll"))
    return success();

  OpBuilder builder(forOp);
  auto *context = builder.getContext();
  builder.setInsertionPoint(forOp);
  Type ivTy = forOp.getInductionVar().getType();
  Value initialCondition;
  if (forOp.lb().getType().isSignedInteger())
    initialCondition = builder.create<mlir::CmpIOp>(builder.getUnknownLoc(),
                                                    mlir::CmpIPredicate::slt,
                                                    forOp.lb(), forOp.ub());
  else
    initialCondition = builder.create<mlir::CmpIOp>(builder.getUnknownLoc(),
                                                    mlir::CmpIPredicate::ult,
                                                    forOp.lb(), forOp.ub());
  auto rdPort = helper::getDictionaryAttr(builder, "rd_latency",
                                          builder.getI64IntegerAttr(0));
  auto wrPort = helper::getDictionaryAttr(builder, "wr_latency",
                                          builder.getI64IntegerAttr(1));

  auto ivRegTy = hir::MemrefType::get(
      context, {1}, builder.getIntegerType(ivTy.getIntOrFloatBitWidth()),
      {BANK});

  auto ivReg = builder.create<hir::AllocaOp>(
      builder.getUnknownLoc(), ivRegTy, builder.getStringAttr("REG"),
      builder.getArrayAttr({rdPort, wrPort}));

  auto whileOp = builder.create<hir::WhileOp>(
      forOp.getLoc(), initialCondition, forOp.tstart(), forOp.offsetAttr());
  auto forNextIterOp = dyn_cast<ForNextIterOp>(&forOp.body().begin()->back());
  assert(forNextIterOp);

  whileOp.addEntryBlock();
  builder.setInsertionPointToStart(whileOp.getBody(0));

  {
    Value c0 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(),
                                                builder.getIndexType(),
                                                builder.getIndexAttr(0));
    Value iv = builder.create<hir::LoadOp>(
        builder.getUnknownLoc(), ivTy, ivReg, c0, builder.getI64IntegerAttr(0),
        builder.getI64IntegerAttr(0), whileOp.getIterTimeVar(),
        builder.getI64IntegerAttr(0));
    auto zeroDelayAttr = helper::getDictionaryAttr(
        builder, "hir.delay", builder.getI64IntegerAttr(0));

    auto funcTy = hir::FuncType::get(
        context, {ivTy, ivTy, ivTy, ivTy},
        {zeroDelayAttr, zeroDelayAttr, zeroDelayAttr, zeroDelayAttr},
        {builder.getI1Type(), ivTy}, {zeroDelayAttr, zeroDelayAttr});

    auto conditionAndNextIV =
        builder
            .create<hir::CallOp>(
                builder.getUnknownLoc(),
                SmallVector<Type>(
                    {builder.getI1Type(), forOp.getInductionVar().getType()}),
                FlatSymbolRefAttr::get(context, "HIR_for_op_entry"),
                TypeAttr::get(funcTy),
                SmallVector<Value>({iv, forOp.lb(), forOp.ub(), forOp.step()}),
                whileOp.getIterTimeVar(), builder.getI64IntegerAttr(0))
            .getResults();

    builder.create<hir::StoreOp>(
        builder.getUnknownLoc(), conditionAndNextIV[1], ivReg, c0,
        builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1),
        whileOp.getIterTimeVar(), builder.getI64IntegerAttr(0));

    BlockAndValueMapping operandMap;
    // Latch the captures and update them as the new captures for the next
    // iteration.

    operandMap.map(forOp.getInductionVar(), iv);
    operandMap.map(forOp.getIterTimeVar(), whileOp.getIterTimeVar());

    // Copy the loop body.
    for (auto &operation : forOp.getLoopBody().front()) {
      if (auto forNextIterOp = dyn_cast<hir::ForNextIterOp>(&operation)) {
        builder.create<hir::WhileNextIterOp>(
            builder.getUnknownLoc(), conditionAndNextIV[0],
            operandMap.lookup(forNextIterOp.tstart()),
            forNextIterOp.offsetAttr());
        continue;
      }
      builder.clone(operation, operandMap);
    }
  }
  return success();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createSimplifyLoopPass() {
  return std::make_unique<SimplifyLoopPass>();
}
} // namespace hir
} // namespace circt
