//===- Wire2Node.cpp - topo sort and convert wires to nodes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace firrtl;


namespace {
    struct Wire2Node {
        FModuleOp mod;

        Wire2Node(FModuleOp mod) : mod(mod) {}
        LogicalResult run();
        void orderBlock(Block* block);
    };
}



bool sortTopologicallyEx(
    Block *block, 
    function_ref<bool(Value, Operation *)> isOperandReady) {
        if (block->empty())
        return true;
        auto ops =   block->back().hasTrait<OpTrait::IsTerminator>() ?
     block->without_terminator() : * block;
     
  // The set of operations that have not yet been scheduled.
  DenseSet<Operation *> unscheduledOps;
  // Mark all operations as unscheduled.
  for (Operation &op : ops)
    unscheduledOps.insert(&op);

  Block::iterator nextScheduledOp = ops.begin();
  Block::iterator end = ops.end();

  bool allOpsScheduled = true;
  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (move it before the `nextScheduledOp`).
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      if (!isOpReady(&op, unscheduledOps, isOperandReady))
        continue;

      // Schedule the operation by moving it to the start.
      unscheduledOps.erase(&op);
      op.moveBefore(block, nextScheduledOp);
      scheduledAtLeastOnce = true;
      // Move the iterator forward if we schedule the operation at the front.
      if (&op == &*nextScheduledOp)
        ++nextScheduledOp;
    }
    // If no operations were scheduled, give up and advance the iterator.
    if (!scheduledAtLeastOnce) {
      allOpsScheduled = false;
      unscheduledOps.erase(&*nextScheduledOp);
      ++nextScheduledOp;
    }
  }

  return allOpsScheduled;
}


void Wire2Node::orderBlock(Block* block) {
    if (block->empty())
        return;
    auto* last = &block->back();
    for (auto ii = block->rbegin(), ee = block->rend(); ii != ee; ) {
        Operation& op = *ii++;
        if (isa<ConnectOp, StrictConnectOp>(op)) {
                op.moveAfter(last);
                last = &op;
        }
}
}

LogicalResult Wire2Node::run() {
    // First, sink all connects to the end of the module
    auto* block = mod.getBodyBlock();
    orderBlock(block);

//    mlir::sortTopologically(mod.getBodyBlock(), *mod.getBodyBlock(), wireMover );
    return success();
}


//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct Wire2NodePass : public Wire2NodeBase<Wire2NodePass> {
  Wire2NodePass() = default;
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the wire2node pass.
void Wire2NodePass::runOnOperation() {
    llvm::errs() << "Running on: " << getOperation().getName() << "\n";
    Wire2Node transform(getOperation());
    if (failed(transform.run()))
      signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createWire2NodePass() {
  return std::make_unique<Wire2NodePass>();
}
