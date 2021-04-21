//=========- CanonicalizationPass.cpp - Canonicalize varius instructions---===//
//
// This file implements the HIR canonicalization pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
using namespace mlir;
namespace {

class CanonicalizationPass
    : public hir::CanonicalizationBase<CanonicalizationPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void inspectOp(hir::FuncOp op) {}
void inspectOp(hir::ConstantOp op) {}
void inspectOp(hir::ForOp op) {}
void inspectOp(hir::UnrollForOp op) {}
void inspectOp(hir::AddOp op) {}
void inspectOp(hir::SubtractOp op) {}
void inspectOp(hir::StoreOp op) {}
void inspectOp(hir::ReturnOp op) {}
void inspectOp(hir::YieldOp op) {}
void inspectOp(hir::SendOp op) {}
void inspectOp(hir::RecvOp op) {}
void inspectOp(hir::AllocaOp op) {}
void inspectOp(hir::DelayOp op) {}
void inspectOp(hir::CallOp op) {}

void processLoadOp(hir::LoadOp op) {
  /*if (op.offset()) {
    mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
    builder.setInsertionPoint(op);
    hir::DelayOp newDelayOp = builder.create<hir::DelayOp>(
        op.getLoc(), op.tstart().getType(), op.tstart(), op.offset(),
        op.tstart(), mlir::Value());
    hir::LoadOp newLoadOp =
        builder.create<hir::LoadOp>(op.getLoc(), op.res().getType(), op.mem(),
                                    op.addr(), newDelayOp, mlir::Value());
    op.replaceAllUsesWith(newLoadOp.getOperation());
    op.getOperation()->dropAllReferences();
    op.getOperation()->dropAllUses();
    op.getOperation()->erase();
  }
  */
}

void CanonicalizationPass::runOnOperation() {
  /*
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([](Operation *operation) -> WalkResult {
    if (hir::LoadOp op = dyn_cast<hir::LoadOp>(operation))
      processLoadOp(op);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
  */
}

namespace mlir {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createCanonicalizationPass() {
  return std::make_unique<CanonicalizationPass>();
}
} // namespace hir
} // namespace mlir
