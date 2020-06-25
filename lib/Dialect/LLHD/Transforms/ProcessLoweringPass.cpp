#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

namespace {

struct ProcessLoweringPass
    : public llhd::ProcessLoweringBase<ProcessLoweringPass> {
  void runOnOperation() override;
};

void ProcessLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  module.walk([this](llhd::ProcOp op) {
    // Check invariants
    size_t numBlocks = op.body().getBlocks().size();
    if (numBlocks == 1) {
      if (!dyn_cast<llhd::HaltOp>(op.body().back().getTerminator())) {
        op.emitOpError("Process-lowering: Entry block is required to be "
                       "terminated by a HaltOp from the LLHD dialect.");
        signalPassFailure();
      }
    } else if (numBlocks == 2) {
      Block &first = op.body().front();
      Block &last = op.body().back();
      if (last.getArguments().size() != 0) {
        op.emitOpError("Process-lowering: The second block (containing the "
                       "llhd.wait) is not allowed to have arguments.");
        signalPassFailure();
      }
      if (!dyn_cast<BranchOp>(first.getTerminator())) {
        op.emitOpError("Process-lowering: The first block has to be terminated "
                       "by a BranchOp from the standard dialect.");
        signalPassFailure();
      }
      if (auto wait = dyn_cast<llhd::WaitOp>(last.getTerminator())) {
        // No optional time argument is allowed
        if (wait.time()) {
          wait.emitOpError(
              "Process-lowering: llhd.wait terminators with optional time "
              "argument cannot be lowered to structural LLHD.");
          signalPassFailure();
        }
        // Every probed signal has to occur in the observed signals list in
        // the wait instruction
        bool isValid = true;
        op.walk([&isValid, &wait](llhd::PrbOp prbOp) {
          if (!llvm::is_contained(wait.obs(), prbOp.signal())) {
            wait.emitOpError(
                "Process-lowering: The wait terminator is required to have "
                "all probed signals as arguments!");
            isValid = false;
          }
        });
        if (!isValid)
          signalPassFailure();
      } else {
        op.emitOpError(
            "Process-lowering: The second block must be terminated by "
            "a WaitOp from the LLHD dialect.");
        signalPassFailure();
      }
    } else {
      op.emitOpError(
          "Process-lowering only supports processes with either one basic "
          "block terminated by a llhd.halt operation or two basic blocks where "
          "the first one contains a std.br terminator and the second one "
          "is terminated by a llhd.wait operation.");
      signalPassFailure();
    }

    OpBuilder builder(op);

    // Replace proc with entity
    llhd::EntityOp entity =
        builder.create<llhd::EntityOp>(op.getLoc(), op.ins());
    // Set the symbol name of the entity to the same as the process (as the
    // process gets deleted anyways).
    entity.setName(op.getName());
    // Move all blocks from the process to the entity, the process does not have
    // a region afterwards.
    entity.body().takeBody(op.body());
    entity.setAttr("type", op.getAttr("type"));
    // In the case that wait is used to suspend the process, we need to merge
    // the two blocks as we needed the second block to have a target for wait
    // (the entry block cannot be targeted).
    if (entity.body().getBlocks().size() == 2) {
      Block &first = entity.body().front();
      Block &second = entity.body().back();
      // Delete the BranchOp operation in the entry block
      first.getTerminator()->dropAllReferences();
      first.getTerminator()->erase();
      // Move operations of second block in entry block.
      first.getOperations().splice(first.end(), second.getOperations());
      // Drop all references to the second block and delete it.
      second.dropAllReferences();
      second.dropAllDefinedValueUses();
      second.erase();
    }

    // Delete the process as it is now replaced by an entity.
    op.getOperation()->dropAllReferences();
    op.getOperation()->dropAllDefinedValueUses();
    op.getOperation()->erase();

    // Replace the llhd.halt or llhd.wait with the implicit entity terminator
    builder.setInsertionPointToEnd(&entity.body().front());
    Operation *terminator = entity.body().front().getTerminator();
    builder.create<llhd::TerminatorOp>(terminator->getLoc());
    terminator->dropAllReferences();
    terminator->dropAllUses();
    terminator->erase();
  });
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createProcessLoweringPass() {
  return std::make_unique<ProcessLoweringPass>();
}
