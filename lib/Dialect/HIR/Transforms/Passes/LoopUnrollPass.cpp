//=========- LoopUnrollPass.cpp - Canonicalize varius instructions---===//
//
// This file implements the HIR canonicalization pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace circt;
namespace {

class LoopUnrollPass : public hir::LoopUnrollBase<LoopUnrollPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

Value lookupOrOriginal(BlockAndValueMapping &mapper, Value originalValue) {
  if (mapper.contains(originalValue))
    return mapper.lookup(originalValue);
  return originalValue;
}

LogicalResult unrollLoopFull(hir::ForOp forOp) {
  Block &loopBodyBlock = forOp.getLoopBody().front();
  // auto builder = OpBuilder::atBlockTerminator(&loopBodyBlock);
  auto builder = OpBuilder(forOp);
  builder.setInsertionPointAfter(forOp);

  if (failed(helper::isConstantIntValue(forOp.lb())))
    return forOp.emitError("Expected lower bound to be constant.");
  if (failed(helper::isConstantIntValue(forOp.ub())))
    return forOp.emitError("Expected upper bound to be constant.");
  if (failed(helper::isConstantIntValue(forOp.step())))
    return forOp.emitError("Expected step to be constant.");

  int64_t lb = helper::getConstantIntValue(forOp.lb()).getValue();
  int64_t ub = helper::getConstantIntValue(forOp.ub()).getValue();
  int64_t step = helper::getConstantIntValue(forOp.step()).getValue();

  Block::iterator srcBlockEnd = std::prev(loopBodyBlock.end(), 1);

  Value iterTimeVar = forOp.getIterTimeVar();
  assert(forOp.offset().getValue() == 0);
  Value iterTStart = forOp.tstart();
  auto *context = builder.getContext();

  // insert the unrolled body.

  for (int i = lb; i < ub; i += step) {
    Value loopIV = builder
                       .create<mlir::ConstantOp>(builder.getUnknownLoc(),
                                                 IndexType::get(context),
                                                 builder.getIndexAttr(0))
                       .getResult();

    BlockAndValueMapping operandMap;
    // Latch the captures and update them as the new captures for the next
    // iteration.
    for (auto capture : forOp.getCapturedValues()) {
      operandMap.map(capture,
                     builder.create<hir::LatchOp>(
                         builder.getUnknownLoc(), capture.getType(), capture,
                         iterTStart, builder.getI64IntegerAttr(0)));
    }
    operandMap.map(iterTimeVar, iterTStart);
    operandMap.map(forOp.getInductionVar(), loopIV);

    // Copy the loop body.
    for (auto it = loopBodyBlock.begin(); it != srcBlockEnd; it++) {
      if (auto forNextIterOp = dyn_cast<hir::NextIterOp>(it)) {
        assert(forNextIterOp.offset().getValue() == 0);
        iterTStart = {lookupOrOriginal(operandMap, forNextIterOp.tstart())};
        continue;
      }
      builder.clone(*it, operandMap);
    }
  }

  // replace the ForOp results.
  auto results = forOp->getResults();
  assert(results.size() <= 1);
  results[0].replaceAllUsesWith(iterTStart);

  forOp.erase();
  return success();
}

void LoopUnrollPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([](Operation *operation) -> WalkResult {
    if (auto forOp = dyn_cast<hir::ForOp>(operation)) {
      if (forOp->getAttr("unroll"))
        if (failed(unrollLoopFull(forOp)))
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}
} // namespace hir
} // namespace circt
