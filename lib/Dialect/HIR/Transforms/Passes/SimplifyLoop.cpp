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

static mlir::FlatSymbolRefAttr createUniqueFuncName(MLIRContext *context,
                                                    StringRef name,
                                                    ArrayRef<uint64_t> params) {
  std::string newName(name);
  newName += "_";
  for (uint64_t i = 0; i < params.size(); i++) {
    if (i > 0)
      newName += "x";
    newName += std::to_string(params[i]);
  }
  return FlatSymbolRefAttr::get(context, newName);
}

class SimplifyLoopPass : public hir::SimplifyLoopBase<SimplifyLoopPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(ForOp);
  LogicalResult visitOp(IfOp);
};

void SimplifyLoopPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::ForOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    if (auto op = dyn_cast<hir::IfOp>(operation)) {
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
  Type ivTy = forOp.getInductionVar().getType();
  if (ivTy.isIndex())
    return success();

  OpBuilder builder(forOp);
  auto *context = builder.getContext();
  builder.setInsertionPoint(forOp);
  Value initialCondition;
  if (forOp.lb().getType().isSignedInteger())
    initialCondition = builder.create<mlir::arith::CmpIOp>(
        builder.getUnknownLoc(), mlir::arith::CmpIPredicate::slt, forOp.lb(),
        forOp.ub());
  else
    initialCondition = builder.create<mlir::arith::CmpIOp>(
        builder.getUnknownLoc(), mlir::arith::CmpIPredicate::ult, forOp.lb(),
        forOp.ub());
  auto whileOp = builder.create<hir::WhileOp>(
      forOp.getLoc(), initialCondition, forOp.tstart(), forOp.offsetAttr());
  auto forNextIterOp = dyn_cast<NextIterOp>(&forOp.body().begin()->back());
  assert(forNextIterOp);

  whileOp.addEntryBlock();
  builder.setInsertionPointToStart(whileOp.getBody(0));

  {
    Value isFirstIter = builder.create<hir::IsFirstIterOp>(
        builder.getUnknownLoc(), builder.getI1Type());
    auto zeroDelayAttr = helper::getDictionaryAttr(
        builder, "hir.delay", builder.getI64IntegerAttr(0));

    auto funcTy = hir::FuncType::get(
        context, {builder.getI1Type(), ivTy, ivTy, ivTy},
        {zeroDelayAttr, zeroDelayAttr, zeroDelayAttr, zeroDelayAttr},
        {builder.getI1Type(), ivTy}, {zeroDelayAttr, zeroDelayAttr});

    auto forOpEntryName = createUniqueFuncName(context, "HIR_ForOp_entry",
                                               ivTy.getIntOrFloatBitWidth());
    auto forOpEntry = builder.create<hir::CallOp>(
        builder.getUnknownLoc(),
        SmallVector<Type>(
            {builder.getI1Type(), forOp.getInductionVar().getType()}),
        StringAttr(), forOpEntryName, TypeAttr::get(funcTy),
        SmallVector<Value>({isFirstIter, forOp.lb(), forOp.ub(), forOp.step()}),
        whileOp.getIterTimeVar(), builder.getI64IntegerAttr(0));
    auto width = builder.getI64IntegerAttr(ivTy.getIntOrFloatBitWidth());
    auto params =
        builder.getDictionaryAttr({builder.getNamedAttr("WIDTH", width)});
    helper::setNames(forOpEntry,
                     {forOp.getInductionVarName().str() + "_loop_condition",
                      forOp.getInductionVarName()});
    forOpEntry->setAttr("params", params);
    helper::declareExternalFuncForCall(
        forOpEntry, {"isFirstIter", "LowerBound", "UpperBound", "Step"},
        {forOp.getInductionVarName().str() + "_loop_condition",
         forOp.getInductionVarName()});

    auto condition = forOpEntry.getResult(0);
    auto iv = forOpEntry.getResult(1);
    BlockAndValueMapping operandMap;

    operandMap.map(forOp.getInductionVar(), iv);
    operandMap.map(forOp.getIterTimeVar(), whileOp.getIterTimeVar());

    // Copy the loop body.
    for (auto &operation : forOp.getLoopBody().front()) {
      if (auto nextIterOp = dyn_cast<hir::NextIterOp>(operation)) {
        builder.create<hir::NextIterOp>(builder.getUnknownLoc(), condition,
                                        operandMap.lookup(nextIterOp.tstart()),
                                        nextIterOp.offsetAttr());
      } else {
        builder.clone(operation, operandMap);
      }
    }
  }

  if (auto attr = forOp->getAttrOfType<ArrayAttr>("names"))
    whileOp->setAttr("names", attr);
  forOp.replaceAllUsesWith((Operation *)whileOp);
  forOp.erase();
  return success();
}

SmallVector<Value> inlineRegion(OpBuilder &builder,
                                BlockAndValueMapping &operandMap,
                                mlir::Region &r) {
  SmallVector<Value> regionOutput;
  for (auto &operation : r.front()) {
    if (auto yieldOp = dyn_cast<hir::YieldOp>(operation)) {
      for (auto operand : yieldOp.operands())
        regionOutput.push_back(operandMap.lookup(operand));
    } else {
      builder.clone(operation, operandMap);
    }
  }
  return regionOutput;
}

LogicalResult SimplifyLoopPass::visitOp(IfOp op) {
  assert(op.offset().getValue() == 0);
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  BlockAndValueMapping ifRegionOperandMap;
  BlockAndValueMapping elseRegionOperandMap;
  auto c1 = helper::materializeIntegerConstant(builder, 1, 1);
  Value tstartBus = builder.create<hir::CastOp>(
      builder.getUnknownLoc(),
      hir::BusType::get(builder.getContext(), builder.getI1Type()),
      op.tstart());
  auto conditionBus = builder.create<hir::BusOp>(
      builder.getUnknownLoc(),
      BusType::get(builder.getContext(), builder.getI1Type()));

  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c1, conditionBus,
                           op.tstart(), op.offsetAttr())
      ->setAttr("default", IntegerAttr::get(builder.getI1Type(), 0));

  // This acts as conditionBus && tstartBus.
  Value tstartRegionBus = builder.create<hir::BusSelectOp>(
      builder.getUnknownLoc(), tstartBus.getType(), conditionBus, tstartBus,
      conditionBus);
  Value tstartRegion = builder.create<hir::CastOp>(
      builder.getUnknownLoc(), hir::TimeType::get(builder.getContext()),
      tstartRegionBus);

  ifRegionOperandMap.map(op.getRegionTimeVar(), tstartRegion);
  elseRegionOperandMap.map(op.getRegionTimeVar(), tstartRegion);

  hir::YieldOp ifYield;
  hir::YieldOp elseYield;

  auto ifResults = inlineRegion(builder, ifRegionOperandMap, op.if_region());
  auto elseResults =
      inlineRegion(builder, elseRegionOperandMap, op.else_region());
  for (size_t i = 0; i < op.getNumResults(); i++) {
    op.results()[i].replaceAllUsesWith(builder.create<hir::SelectOp>(
        builder.getUnknownLoc(), ifResults[i].getType(), op.condition(),
        ifResults[i], elseResults[i]));
  }
  op.erase();
  return success();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createSimplifyLoopPass() {
  return std::make_unique<SimplifyLoopPass>();
}
} // namespace hir
} // namespace circt
