//===- ProcessLoweringPass.cpp - Implement Process Lowering Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to transform combinational processes to entities.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Visitors.h"

using namespace circt;

namespace {

struct ProcessLoweringPass
    : public llhd::ProcessLoweringBase<ProcessLoweringPass> {
  void runOnOperation() override;
};

void ProcessLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  WalkResult result = module.walk([](llhd::ProcOp op) -> WalkResult {
    // Check invariants
    size_t numBlocks = op.body().getBlocks().size();
    if (numBlocks == 1) {
      if (!isa<llhd::HaltOp>(op.body().back().getTerminator())) {
        return op.emitOpError("Process-lowering: Entry block is required to be "
                              "terminated by a HaltOp from the LLHD dialect.");
      }
    } else if (numBlocks == 2) {
      Block &first = op.body().front();
      Block &last = op.body().back();
      if (last.getArguments().size() != 0) {
        return op.emitOpError(
            "Process-lowering: The second block (containing the "
            "llhd.wait) is not allowed to have arguments.");
      }
      if (!isa<mlir::BranchOp>(first.getTerminator())) {
        return op.emitOpError(
            "Process-lowering: The first block has to be terminated "
            "by a BranchOp from the standard dialect.");
      }
      if (auto wait = dyn_cast<llhd::WaitOp>(last.getTerminator())) {
        // No optional time argument is allowed
        if (wait.time()) {
          return wait.emitOpError(
              "Process-lowering: llhd.wait terminators with optional time "
              "argument cannot be lowered to structural LLHD.");
        }
        // Every probed signal has to occur in the observed signals list in
        // the wait instruction
        WalkResult result = op.walk([&wait](llhd::PrbOp prbOp) -> WalkResult {
          if (!llvm::is_contained(wait.obs(), prbOp.signal())) {
            return wait.emitOpError(
                "Process-lowering: The wait terminator is required to have "
                "all probed signals as arguments!");
          }
          return WalkResult::advance();
        });
        if (result.wasInterrupted()) {
          return result;
        }
      } else {
        return op.emitOpError(
            "Process-lowering: The second block must be terminated by "
            "a WaitOp from the LLHD dialect.");
      }
    } else {
      return op.emitOpError(
          "Process-lowering only supports processes with either one basic "
          "block terminated by a llhd.halt operation or two basic blocks where "
          "the first one contains a std.br terminator and the second one "
          "is terminated by a llhd.wait operation.");
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
    entity->setAttr("type", op->getAttr("type"));
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

    // Remove the remaining llhd.halt or llhd.wait terminator
    Operation *terminator = entity.body().front().getTerminator();
    terminator->dropAllReferences();
    terminator->dropAllUses();
    terminator->erase();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    signalPassFailure();
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
circt::llhd::createProcessLoweringPass() {
  return std::make_unique<ProcessLoweringPass>();
}
