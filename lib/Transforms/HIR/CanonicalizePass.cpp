//=========- CanonicalizationPass.cpp - Canonicalize varius instructions---===//
//
// This file implements the HIR canonicalization pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
using namespace mlir;
namespace {

class CanonicalizationPass
    : public hir::CanonicalizationBase<CanonicalizationPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void inspectOp(hir::DefOp op) {}
void inspectOp(hir::ConstantOp op) {}
void inspectOp(hir::ForOp op) {}
void inspectOp(hir::UnrollForOp op) {}
void inspectOp(hir::AddOp op) {}
void inspectOp(hir::SubtractOp op) {}
void inspectOp(hir::MemWriteOp op) {}
void inspectOp(hir::ReturnOp op) {}
void inspectOp(hir::YieldOp op) {}
void inspectOp(hir::WireWriteOp op) {}
void inspectOp(hir::WireReadOp op) {}
void inspectOp(hir::AllocOp op) {}
void inspectOp(hir::DelayOp op) {}
void inspectOp(hir::CallOp op) {}

void processMemReadOp(hir::MemReadOp op) {
  if (op.offset()) {
    mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
    builder.setInsertionPoint(op);
    hir::DelayOp newDelayOp = builder.create<hir::DelayOp>(
        op.getLoc(), op.tstart().getType(), op.tstart(), op.offset(),
        op.tstart(), mlir::Value());
    hir::MemReadOp newMemReadOp = builder.create<hir::MemReadOp>(
        op.getLoc(), op.res().getType(), op.mem(), op.addr(), newDelayOp,
        mlir::Value());
    op.replaceAllUsesWith(newMemReadOp.getOperation());
    op.getOperation()->dropAllReferences();
    op.getOperation()->dropAllUses();
    op.getOperation()->erase();
  }
}

void CanonicalizationPass::runOnOperation() {
  hir::DefOp funcOp = getOperation();
  WalkResult result = funcOp.walk([](Operation *operation) -> WalkResult {
    if (hir::MemReadOp op = dyn_cast<hir::MemReadOp>(operation))
      processMemReadOp(op);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace mlir {
namespace hir {
std::unique_ptr<OperationPass<hir::DefOp>> createCanonicalizationPass() {
  return std::make_unique<CanonicalizationPass>();
}
} // namespace hir
} // namespace mlir
