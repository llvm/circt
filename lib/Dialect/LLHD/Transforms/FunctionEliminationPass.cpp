#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

namespace {

struct FunctionEliminationPass
    : public llhd::FunctionEliminationBase<FunctionEliminationPass> {
  void runOnOperation() override;
};

void FunctionEliminationPass::runOnOperation() {
  ModuleOp module = getOperation();

  module.walk([this](CallOp op) {
    if (dyn_cast<llhd::ProcOp>(op.getParentOp()) ||
        dyn_cast<llhd::EntityOp>(op.getParentOp())) {
      emitError(op.getLoc(),
                "Not all functions are inlined, there is at least "
                "one function call left within a llhd.proc or llhd.entity.");
      signalPassFailure();
    }
  });

  module.walk([](FuncOp op) { op.erase(); });
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createFunctionEliminationPass() {
  return std::make_unique<FunctionEliminationPass>();
}
