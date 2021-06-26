//=========- LoopUnrollPass.cpp - Canonicalize varius instructions---===//
//
// This file implements the HIR canonicalization pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace circt;
namespace {

class LoopUnrollPass : public hir::LoopUnrollBase<LoopUnrollPass> {
public:
  void runOnOperation() override;
  void unrollLoopFull(hir::UnrollForOp);
};
} // end anonymous namespace

Value lookupOrOriginal(BlockAndValueMapping &mapper, Value originalValue) {
  if (mapper.contains(originalValue))
    return mapper.lookup(originalValue);
  return originalValue;
}

void LoopUnrollPass::unrollLoopFull(hir::UnrollForOp unrollForOp) {
  Block &loopBodyBlock = unrollForOp.getLoopBody().front();
  int lb = unrollForOp.lb();
  int ub = unrollForOp.ub();
  int step = unrollForOp.step();
  auto builder = OpBuilder::atBlockTerminator(&loopBodyBlock);
  builder.setInsertionPointAfter(unrollForOp);
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock.end(), 1);

  SmallVector<Value, 4> iterArgs = {unrollForOp.getIterTimeVar()};
  assert(!unrollForOp.offset());
  SmallVector<Value, 4> iterValues = {unrollForOp.tstart()};
  auto *context = builder.getContext();

  // insert the unrolled body.
  for (int i = lb; i < ub; i += step) {
    Value loopIV = builder
                       .create<hir::ConstantOp>(
                           builder.getUnknownLoc(), IndexType::get(context),
                           helper::getIntegerAttr(context, 0))
                       .getResult();

    BlockAndValueMapping operandMap;
    operandMap.map(iterArgs, iterValues);
    operandMap.map(unrollForOp.getInductionVar(), loopIV);

    // Copy the loop body.
    for (auto it = loopBodyBlock.begin(); it != srcBlockEnd; it++) {
      if (auto yieldOp = dyn_cast<hir::YieldOp>(it)) {
        assert(!yieldOp.offset());
        iterValues = {lookupOrOriginal(operandMap, yieldOp.tstart())};
        operandMap.map(iterArgs, iterValues);
        continue;
      }
      builder.clone(*it, operandMap);
    }
  }

  // replace the UnrollForOp results.
  auto results = unrollForOp->getResults();
  for (unsigned i = 0; i < results.size(); i++) {
    Value unrollForOpResult = results[i];
    Value newResult = iterValues[i];
    unrollForOpResult.replaceAllUsesWith(newResult);
  }

  unrollForOp.erase();
}

void LoopUnrollPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto unrollForOp = dyn_cast<hir::UnrollForOp>(operation))
      unrollLoopFull(unrollForOp);
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
