//=========- LoopUnrollPass.cpp - Canonicalize varius instructions---===//
//
// This file implements the HIR canonicalization pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
namespace {

class LoopUnrollPass : public hir::LoopUnrollBase<LoopUnrollPass> {
public:
  void runOnOperation() override;

public:
  bool visitOp(Operation *);
  bool visitOp(hir::ConstantOp);
  bool visitOp(hir::ForOp) { return true; }
  bool visitOp(hir::UnrollForOp) { return true; }
  bool visitOp(hir::AddOp) { return true; }
  bool visitOp(hir::SubtractOp) { return true; }
  bool visitOp(hir::LoadOp) { return true; }
  bool visitOp(hir::StoreOp) { return true; }
  bool visitOp(hir::ReturnOp) { return true; }
  bool visitOp(hir::YieldOp) { return true; }
  bool visitOp(hir::SendOp) { return true; }
  bool visitOp(hir::RecvOp) { return true; }
  bool visitOp(hir::AllocaOp) { return true; }
  bool visitOp(hir::DelayOp) { return true; }
  bool visitOp(hir::CallOp) { return true; }
};
} // end anonymous namespace

bool LoopUnrollPass::visitOp(hir::ConstantOp op) {
  op.clone();
  return true;
}
bool LoopUnrollPass::visitOp(Operation *operation) {
  if (auto op = dyn_cast<hir::ConstantOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ConstantOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ForOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::UnrollForOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::AddOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::SubtractOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::LoadOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::StoreOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ReturnOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::YieldOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::SendOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::RecvOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::AllocaOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::DelayOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CallOp>(operation))
    return visitOp(op);
  return emitError(operation->getLoc(),
                   "[hir::LoopUnrollPass] Unsupported operation!"),
         false;
}

void LoopUnrollPass::runOnOperation() {
  hir::UnrollForOp unrollForOp = getOperation();
  WalkResult result =
      unrollForOp.walk([this](Operation *operation) -> WalkResult {
        visitOp(operation);
        return WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace mlir {
namespace hir {
std::unique_ptr<OperationPass<hir::UnrollForOp>> createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}
} // namespace hir
} // namespace mlir
