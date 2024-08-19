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

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_PROCESSLOWERING
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

struct ProcessLoweringPass
    : public circt::llhd::impl::ProcessLoweringBase<ProcessLoweringPass> {
  void runOnOperation() override;
};

static LogicalResult isProcValidToLower(llhd::ProcessOp op) {
  size_t numBlocks = op.getBody().getBlocks().size();

  if (numBlocks == 1) {
    if (!isa<llhd::HaltOp>(op.getBody().back().getTerminator()))
      return op.emitOpError("during process-lowering: entry block is required "
                            "to be terminated by llhd.halt");
    return success();
  }

  if (numBlocks == 2) {
    Block &first = op.getBody().front();
    Block &last = op.getBody().back();

    if (!last.getArguments().empty())
      return op.emitOpError(
          "during process-lowering: the second block (containing the "
          "llhd.wait) is not allowed to have arguments");

    if (!isa<cf::BranchOp>(first.getTerminator()))
      return op.emitOpError("during process-lowering: the first block has to "
                            "be terminated by a cf.br operation");

    if (auto wait = dyn_cast<llhd::WaitOp>(last.getTerminator())) {
      // No optional time argument is allowed
      if (wait.getTime())
        return wait.emitOpError(
            "during process-lowering: llhd.wait terminators with optional time "
            "argument cannot be lowered to structural LLHD");

      SmallVector<Value> observedSignals;
      for (Value obs : wait.getObserved())
        if (auto prb = obs.getDefiningOp<llhd::PrbOp>())
          if (!op.getBody().isAncestor(prb->getParentRegion()))
            observedSignals.push_back(prb.getSignal());

      // Every probed signal has to occur in the observed signals list in
      // the wait instruction
      WalkResult result = op.walk([&](Operation *operation) -> WalkResult {
        // TODO: value does not need to be observed if all values this value is
        // a combinatorial result of are observed.
        for (Value operand : operation->getOperands())
          if (!op.getBody().isAncestor(operand.getParentRegion()) &&
              !llvm::is_contained(wait.getObserved(), operand) &&
              !llvm::is_contained(observedSignals, operand))
            return wait.emitOpError(
                "during process-lowering: the wait terminator is required to "
                "have values used in the process as arguments");

        return WalkResult::advance();
      });
      return failure(result.wasInterrupted());
    }

    return op.emitOpError("during process-lowering: the second block must be "
                          "terminated by llhd.wait");
  }

  return op.emitOpError(
      "process-lowering only supports processes with either one basic block "
      "terminated by a llhd.halt operation or two basic blocks where the first "
      "one contains a cf.br terminator and the second one is terminated by a "
      "llhd.wait operation");
}

void ProcessLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  WalkResult result = module.walk([](llhd::ProcessOp op) -> WalkResult {
    // Check invariants
    if (failed(isProcValidToLower(op)))
      return WalkResult::interrupt();

    // In the case that wait is used to suspend the process, we need to merge
    // the two blocks as we needed the second block to have a target for wait
    // (the entry block cannot be targeted).
    if (op.getBody().getBlocks().size() == 2) {
      Block &first = op.getBody().front();
      Block &second = op.getBody().back();
      // Delete the BranchOp operation in the entry block
      first.getTerminator()->erase();
      // Move operations of second block in entry block.
      first.getOperations().splice(first.end(), second.getOperations());
      // Drop all references to the second block and delete it.
      second.dropAllReferences();
      second.dropAllDefinedValueUses();
      second.erase();
    }

    // Remove the remaining llhd.halt or llhd.wait terminator
    op.getBody().front().getTerminator()->erase();

    IRRewriter rewriter(op);
    rewriter.inlineBlockBefore(&op.getBody().front(), op);
    op.erase();

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
