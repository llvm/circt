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
  auto builder = OpBuilder::atBlockTerminator(&loopBodyBlock);
  builder.setInsertionPointAfter(forOp);

  if (failed(helper::isConstantIntValue(forOp.lb())))
    return forOp.emitError("Expected lower bound to be constant.");
  if (failed(helper::isConstantIntValue(forOp.ub())))
    return forOp.emitError("Expected upper bound to be constant.");
  if (failed(helper::isConstantIntValue(forOp.step())))
    return forOp.emitError("Expected step to be constant.");

  int64_t lb = helper::getConstantIntValue(forOp.lb());
  int64_t ub = helper::getConstantIntValue(forOp.ub());
  int64_t step = helper::getConstantIntValue(forOp.step());

  Block::iterator srcBlockEnd = std::prev(loopBodyBlock.end(), 1);

  Value iterTimeVar = forOp.getIterTimeVar();
  assert(forOp.offset().getValue() == 0);
  Value iterTStart = forOp.tstart();
  auto *context = builder.getContext();

  // insert the unrolled body.
  // captures must be a new SmallVector since we update it later.
  SmallVector<Value> captures = forOp.captures();

  auto latcheInputs = forOp.getLatchedInputs();
  for (int i = lb; i < ub; i += step) {
    Value loopIV = builder
                       .create<mlir::ConstantOp>(builder.getUnknownLoc(),
                                                 IndexType::get(context),
                                                 builder.getIndexAttr(0))
                       .getResult();

    BlockAndValueMapping operandMap;
    // Latch the captures and update them as the new captures for the next
    // iteration.
    for (size_t i = 0; i < captures.size(); i++) {
      captures[i] = builder.create<hir::LatchOp>(
          builder.getUnknownLoc(), captures[i].getType(), captures[i],
          iterTStart, builder.getI64IntegerAttr(0));
    }
    operandMap.map(latcheInputs, captures);
    operandMap.map(iterTimeVar, iterTStart);
    operandMap.map(forOp.getInductionVar(), loopIV);

    // Copy the loop body.
    for (auto it = loopBodyBlock.begin(); it != srcBlockEnd; it++) {
      if (auto yieldOp = dyn_cast<hir::YieldOp>(it)) {
        assert(yieldOp.offset().getValue() == 0);
        iterTStart = {lookupOrOriginal(operandMap, yieldOp.tstart())};
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
      if (Attribute unrollAttr = forOp->getAttr("unroll")) {
        if (!unrollAttr.dyn_cast<mlir::UnitAttr>()) {
          forOp.emitError(
              "We only support full unrolling, i.e. unroll attr should be "
              "mlir::UnitAttr.");
          return WalkResult::interrupt();
        }
        if (failed(unrollLoopFull(forOp)))
          return WalkResult::interrupt();
      }
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
