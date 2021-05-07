//=========- LoopUnrollPass.cpp - Canonicalize varius instructions---===//
//
// This file implements the HIR canonicalization pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;
namespace {

class LoopUnrollPass : public hir::LoopUnrollBase<LoopUnrollPass> {
public:
  void runOnOperation() override;
  void unrollLoopFull(hir::UnrollForOp);
};
} // end anonymous namespace

void LoopUnrollPass::unrollLoopFull(hir::UnrollForOp op) {
  Block &loopBodyBlock = op.getLoopBody().front();
  int lb = op.lb();
  int ub = op.ub();
  int step = op.step();
  auto builder = OpBuilder::atBlockTerminator(&loopBodyBlock);
  builder.setInsertionPointAfter(op);
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock.end(), 1);

  for (int i = lb; i < ub; i += step) {
    BlockAndValueMapping operandMap;

    // Copy the loop body.
    for (auto it = loopBodyBlock.begin(); it != srcBlockEnd; it++) {
      if (auto yieldOp = dyn_cast<hir::YieldOp>(it))
        continue;
      builder.clone(*it, operandMap);
    }
  }
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

namespace mlir {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}
} // namespace hir
} // namespace mlir
