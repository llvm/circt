//=======-- SeqSchedulerPass.cpp - Populate a sequential schedule.---------===//
//
// This file implements a simple sequential scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;
namespace {
class SeqSchedulerPass : public hir::SeqSchedulerBase<SeqSchedulerPass> {
public:
  void runOnOperation() override;
  LogicalResult updateRegion(Region &);
  LogicalResult updateOp(hir::ForOp);
  LogicalResult updateOp(hir::LoadOp);
  LogicalResult updateOp(hir::StoreOp);
  LogicalResult updateOp(hir::AddIOp);
  LogicalResult updateOp(hir::SubIOp);
  LogicalResult updateOp(hir::MulIOp);
  LogicalResult updateOp(hir::AddFOp);
  LogicalResult updateOp(hir::SubFOp);
  LogicalResult updateOp(hir::MulFOp);
  LogicalResult updateOp(hir::CallOp);

private:
  Value currentTimeVar;
  int64_t nextFreeOffset;
  llvm::DenseMap<Value, int64_t> mapValue2Offset;
};

} // end anonymous namespace

LogicalResult SeqSchedulerPass::updateOp(ForOp op) { return success(); }

LogicalResult SeqSchedulerPass::updateOp(LoadOp op) { return success(); }

LogicalResult SeqSchedulerPass::updateOp(hir::StoreOp) { return success(); }
LogicalResult SeqSchedulerPass::updateOp(hir::AddIOp) { return success(); }
LogicalResult SeqSchedulerPass::updateOp(hir::SubIOp) { return success(); }
LogicalResult SeqSchedulerPass::updateOp(hir::MulIOp) { return success(); }
LogicalResult SeqSchedulerPass::updateOp(hir::AddFOp) { return success(); }
LogicalResult SeqSchedulerPass::updateOp(hir::SubFOp) { return success(); }
LogicalResult SeqSchedulerPass::updateOp(hir::MulFOp) { return success(); }

LogicalResult SeqSchedulerPass::updateRegion(Region &region) {
  for (auto &operation : region.front()) {
    if (auto op = dyn_cast<hir::ForOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::LoadOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::StoreOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::AddIOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::SubIOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::MulIOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::AddFOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::SubFOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::MulFOp>(operation)) {
      if (failed(updateOp(op)))
        return failure();
    } else
      operation.emitError("Unsupported op for SeqSchedulerPass.");
  }
  return failure();
}

void SeqSchedulerPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();

  if (failed(updateRegion(funcOp.getFuncBody())))
    signalPassFailure();
  return;
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createSeqSchedulerPass() {
  return std::make_unique<SeqSchedulerPass>();
}
} // namespace hir
} // namespace circt
