//=======-- SeqSchedulerPass.cpp - Populate a sequential schedule.---------===//
//
// This file implements a simple sequential scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace circt;
using namespace hir;
namespace {
class SeqSchedulerPass : public hir::SeqSchedulerBase<SeqSchedulerPass> {
public:
  void runOnOperation() override {
    hir::FuncOp funcOp = getOperation();

    this->currentTimeVar = funcOp.getRegionTimeVar();
    this->nextFreeOffset = 0;
    if (failed(updateRegion(funcOp.getFuncBody())))
      signalPassFailure();
    return;
  }

private:
  LogicalResult updateRegion(Region &region) {
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
      } else if (auto op = dyn_cast<mlir::arith::ConstantOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<hir::ReturnOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<hir::NextIterOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else
        operation.emitError("Unsupported op for SeqSchedulerPass.");
    }
    return success();
  }

private:
  template <typename T>
  LogicalResult populateSchedule(T op) {
    OpBuilder builder(op);
    if (!this->currentTimeVar)
      return op.emitError(
          "Could not find currentTimeVar while scheduling this op.");
    op.tstartMutable().assign(this->currentTimeVar);
    op.offsetAttr(builder.getI64IntegerAttr(this->nextFreeOffset));
    return success();
  }

private:
  LogicalResult updateOp(hir::ForOp);
  LogicalResult updateOp(hir::LoadOp);
  LogicalResult updateOp(hir::StoreOp);
  LogicalResult updateOp(hir::NextIterOp);
  LogicalResult updateOp(hir::CallOp);

private:
  Value currentTimeVar;
  int64_t nextFreeOffset;
  llvm::DenseMap<Value, int64_t> mapValue2Offset;
};

} // end anonymous namespace

//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Functions to update ops with new schedule.
//-----------------------------------------------------------------------------
LogicalResult SeqSchedulerPass::updateOp(ForOp op) {
  // Schedule the ForOp in the next available slot.
  if (failed(populateSchedule(op)))
    return failure();

  // Update the time var and offset for use inside the loop body.
  this->currentTimeVar = op.getIterTimeVar();
  this->nextFreeOffset = 0;

  // Schedule  the loop body;
  if (failed(updateRegion(op.getLoopBody())))
    return failure();

  // Update the time var and offset for instructions after the ForOp.
  this->currentTimeVar = op.t_end();
  this->nextFreeOffset = 0;

  return success();
}

LogicalResult SeqSchedulerPass::updateOp(LoadOp op) {
  if (failed(populateSchedule(op)))
    return failure();
  // Even if load happens in same cycle (like in case of LUTRAM), next load
  // operation can not be scheduled safely on the same cycle if it is on the
  // same memref and with different address.
  this->nextFreeOffset += op.delay().getValueOr(1);
  return success();
}

LogicalResult SeqSchedulerPass::updateOp(hir::StoreOp op) {
  if (failed(populateSchedule(op)))
    return failure();
  this->nextFreeOffset += op.delay().getValueOr(0);
  return success();
}

LogicalResult SeqSchedulerPass::updateOp(hir::NextIterOp op) {
  if (failed(populateSchedule(op)))
    return failure();
  return success();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createSeqSchedulerPass() {
  return std::make_unique<SeqSchedulerPass>();
}
} // namespace hir
} // namespace circt
