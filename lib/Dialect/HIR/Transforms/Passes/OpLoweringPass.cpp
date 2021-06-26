//=========- OpLoweringPass.cpp - Lower all ops---===//
//
// This file implements lowering pass to lower ops for codegen.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/Analysis/FanoutAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;
namespace {

class OpLoweringPass : public hir::OpLoweringBase<OpLoweringPass> {
public:
  void runOnOperation() override;
  void dispatchOp(Operation *);
  void lowerOp(hir::FuncOp);
  void lowerOp(hir::BusInstantiateOp);

private:
  llvm::DenseMap<Value, Value> mapMemrefRdAddrSend;
  llvm::DenseMap<Value, Value> mapMemrefRdDataRecv;
  llvm::DenseMap<Value, Value> mapMemrefWrSend;
};
} // end anonymous namespace

void OpLoweringPass::lowerOp(hir::FuncOp) {}
void OpLoweringPass::lowerOp(hir::BusInstantiateOp op) {}

void OpLoweringPass::dispatchOp(Operation *operation) {
  if (auto op = dyn_cast<hir::FuncOp>(operation))
    lowerOp(op);
  if (auto op = dyn_cast<hir::BusInstantiateOp>(operation))
    lowerOp(op);
}

void OpLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    dispatchOp(operation);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createOpLoweringPass() {
  return std::make_unique<OpLoweringPass>();
}
} // namespace hir
} // namespace circt
