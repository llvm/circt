//=========- LoopSimplification.cpp - Lower memref type--------------------===//
//
// This file implements loop simplification pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "SimplifyCtrlUtils.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include <iostream>
using namespace circt;
using namespace hir;

class SimplifyCtrlPass : public hir::SimplifyCtrlBase<SimplifyCtrlPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(ForOp);
  LogicalResult visitOp(IfOp);
};

void SimplifyCtrlPass::runOnOperation() {
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
LogicalResult SimplifyCtrlPass::visitOp(ForOp forOp) {
  // The condition var = cmpi "ult",lb,ub: i4.
  assert(!forOp->hasAttr("unroll"));
  assert(!forOp.getInductionVar().getType().isa<mlir::IndexType>());
  // FIXME: We dont yet handle iter_args for non-unrolled loops.
  assert(forOp.iter_args().empty());

  OpBuilder builder(forOp);
  builder.setInsertionPoint(forOp);
  Value initialCondition = builder.create<comb::ICmpOp>(
      builder.getUnknownLoc(),
      forOp.lb().getType().isSignedInteger() ? comb::ICmpPredicate::slt
                                             : comb::ICmpPredicate::ult,
      forOp.lb(), forOp.ub());

  SmallVector<Value> iterArgs;
  for (auto arg : forOp.iter_args())
    iterArgs.push_back(arg);
  auto whileOp =
      builder.create<hir::WhileOp>(forOp.getLoc(), initialCondition, iterArgs,
                                   forOp.tstart(), forOp.offsetAttr());
  auto forNextIterOp = dyn_cast<NextIterOp>(&forOp.body().begin()->back());
  assert(forNextIterOp);

  whileOp.addEntryBlock();
  builder.setInsertionPointToStart(whileOp.getBody(0));

  Value isFirstIter = builder.create<hir::IsFirstIterOp>(
      builder.getUnknownLoc(), builder.getI1Type());

  auto conditionAndIV =
      insertForOpEntryLogic(builder, isFirstIter, forOp.lb(), forOp.ub(),
                            forOp.step(), whileOp.getIterTimeVar());
  auto condition = conditionAndIV.first;
  auto iv = conditionAndIV.second;

  // Create the operandMap.
  BlockAndValueMapping operandMap;
  operandMap.map(forOp.getInductionVar(), iv);
  for (size_t i = 0; i < forOp.iter_args().size(); i++)
    operandMap.map(forOp.body().front().getArgument(i),
                   whileOp.body().front().getArgument(i));

  operandMap.map(forOp.getIterTimeVar(), whileOp.getIterTimeVar());

  // Copy the loop body.
  for (auto &operation : forOp.getLoopBody().front()) {
    if (auto nextIterOp = dyn_cast<hir::NextIterOp>(operation)) {
      SmallVector<Value> mappedIterArgs;
      for (auto iterArg : nextIterOp.iter_args()) {
        auto mappedIterArg = operandMap.lookupOrNull(iterArg);
        mappedIterArg = mappedIterArg ? mappedIterArg : iterArg;
        mappedIterArgs.push_back(mappedIterArg);
      }

      builder.create<hir::NextIterOp>(
          builder.getUnknownLoc(), condition, mappedIterArgs,
          operandMap.lookup(nextIterOp.tstart()), nextIterOp.offsetAttr());
    } else {
      builder.clone(operation, operandMap);
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
      for (auto operand : yieldOp.operands()) {
        auto mappedOperand = operandMap.lookupOrNull(operand);
        mappedOperand = mappedOperand ? mappedOperand : operand;
        regionOutput.push_back(mappedOperand);
      }
    } else {
      builder.clone(operation, operandMap);
    }
  }
  return regionOutput;
}

LogicalResult SimplifyCtrlPass::visitOp(IfOp op) {
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
      .create<hir::BusSendOp>(builder.getUnknownLoc(), c1, conditionBus,
                              op.tstart(), op.offsetAttr())
      ->setAttr("default", IntegerAttr::get(builder.getI1Type(), 0));

  // This acts as conditionBus && tstartBus.
  Value tstartIfRegionBus =
      builder
          .create<hir::BusMapOp>(
              builder.getUnknownLoc(),
              SmallVector<Value>({conditionBus, tstartBus}),
              [](OpBuilder &builder, ArrayRef<Value> operands) {
                Value res = builder.create<comb::AndOp>(
                    builder.getUnknownLoc(), operands[0], operands[1]);
                return builder.create<hir::YieldOp>(builder.getUnknownLoc(),
                                                    res);
              })
          .getResult(0);

  Value tstartIfRegion = builder.create<hir::CastOp>(
      builder.getUnknownLoc(), hir::TimeType::get(builder.getContext()),
      tstartIfRegionBus);

  Value tstartElseRegionBus =
      builder
          .create<hir::BusMapOp>(
              builder.getUnknownLoc(),
              SmallVector<Value>({conditionBus, tstartBus}),
              [](OpBuilder &builder, ArrayRef<Value> operands) {
                Value c0 = builder.create<hw::ConstantOp>(
                    builder.getUnknownLoc(),
                    IntegerAttr::get(builder.getI1Type(), 0));
                Value res = builder.create<comb::MuxOp>(
                    builder.getUnknownLoc(), operands[0], c0, operands[1]);
                return builder.create<hir::YieldOp>(builder.getUnknownLoc(),
                                                    res);
              })
          .getResult(0);

  Value tstartElseRegion = builder.create<hir::CastOp>(
      builder.getUnknownLoc(), hir::TimeType::get(builder.getContext()),
      tstartElseRegionBus);

  ifRegionOperandMap.map(op.getRegionTimeVar(), tstartIfRegion);
  elseRegionOperandMap.map(op.getRegionTimeVar(), tstartElseRegion);

  hir::YieldOp ifYield;
  hir::YieldOp elseYield;

  auto ifResults = inlineRegion(builder, ifRegionOperandMap, op.if_region());
  auto elseResults =
      inlineRegion(builder, elseRegionOperandMap, op.else_region());
  for (size_t i = 0; i < op.getNumResults(); i++) {
    op.results()[i].replaceAllUsesWith(builder.create<comb::MuxOp>(
        builder.getUnknownLoc(), ifResults[i].getType(), op.condition(),
        ifResults[i], elseResults[i]));
  }
  op.erase();
  return success();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createSimplifyCtrlPass() {
  return std::make_unique<SimplifyCtrlPass>();
}
} // namespace hir
} // namespace circt
