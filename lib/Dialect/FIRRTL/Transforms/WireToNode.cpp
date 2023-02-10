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
#include "circt/Dialect/FIRRTL/FIRRTLFieldSource.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace firrtl;


namespace {
    struct Wire2Node {
        FModuleOp mod;
        FieldSource& fields;

        Wire2Node(FModuleOp mod, FieldSource& fields) : mod(mod), fields(fields) {}
        LogicalResult run();
    };
}

static bool isIndexing(Operation* op) {
    return isa<SubfieldOp, SubindexOp, SubaccessOp>(op);
}

/// Return `true` if the given operation is ready to be scheduled.
static bool isOpReadyEx(Operation *op, DenseSet<Operation *> &unscheduledOps, 
FieldSource& sources, DenseSet<Value>& readsSeen) {
  // An operation is ready to be scheduled if all its operands are ready. An
  // operation is ready if:
  const auto isReady = [&](Value value) {
    Operation *parent = value.getDefiningOp();
    // - it is a block argument,
    if (!parent)
      return true;
    // - or it is not defined by an unscheduled op (and also not nested within
    //   an unscheduled op).
    do {
      // Stop traversal when op under examination is reached.
      if (parent == op)
        return true;
      if (unscheduledOps.contains(parent)) {
        return false;
      }
    } while ((parent = parent->getParentOp()));
    // No unscheduled op found.
    // If this is a write, it's fine as long as the rhs has been seen.
    if (isa<ConnectOp, StrictConnectOp>(op) && value == op->getOperand(0) &&
        readsSeen.count(sources.nodeForValue(op->getOperand(1))->src))
        return true;
    // If this reads, then it is ready if its source is in the read set.  That
    // is, we've had to schedule a read already.  This prioritizes writes.
    const auto* node = sources.nodeForValue(value);
    if (node) 
       return readsSeen.contains(node->src);
    // It is not a read, so it is ready
    return true;
  };

  // An operation is recursively ready to be scheduled of it and its nested
  // operations are ready.
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    return llvm::all_of(nestedOp->getOperands(),
                        [&](Value operand) { return isReady(operand); })
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return !readyToSchedule.wasInterrupted();
}


static SmallVector<Operation*> sortTopologicallyEx(
    Block *block, FieldSource& sources) {
        if (block->empty())
        return {};
        auto ops =   block->back().hasTrait<OpTrait::IsTerminator>() ?
     block->without_terminator() : * block;
     
  // The set of operations that have not yet been scheduled.
  DenseSet<Operation *> unscheduledOps;
  DenseSet<Value> readsSeen;
  // Mark all operations as unscheduled.
  for (Operation &op : ops)
    unscheduledOps.insert(&op);

  // All block arguments are already read.  This way we reduce the checks in isReady.
  for (auto& barg : block->getArguments())
    readsSeen.insert(barg);

  Block::iterator nextScheduledOp = ops.begin();
  Block::iterator end = ops.end();

  SmallVector<Operation*> nodePoints;
  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (move it before the `nextScheduledOp`).
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      if (!isOpReadyEx(&op, unscheduledOps, sources, readsSeen))
        continue;

      // Schedule the operation by moving it to the start.
      unscheduledOps.erase(&op);
      op.moveBefore(block, nextScheduledOp);
      scheduledAtLeastOnce = true;
      if (isa<ConnectOp, StrictConnectOp>(&op) && dyn_cast_or_null<WireOp>(op.getOperand(0).getDefiningOp()))
        nodePoints.push_back(&op);
        // It doesn't matter that we mark the write value as a read
      for(auto operand : nextScheduledOp->getOperands())
        readsSeen.insert(operand); 
      // Move the iterator forward if we schedule the operation at the front.
      if (&op == &*nextScheduledOp)
        ++nextScheduledOp;
    }
    // If no operations were scheduled, give up and advance the iterator.
    if (!scheduledAtLeastOnce) {
          for(auto operand : nextScheduledOp->getOperands())
            if (const auto* node = sources.nodeForValue(operand))
                readsSeen.insert(node->src);
      unscheduledOps.erase(&*nextScheduledOp);
      ++nextScheduledOp;
    }
  }

  return nodePoints;
}

LogicalResult Wire2Node::run() {
    // First, sink all connects to the end of the module
//    auto* block = mod.getBodyBlock();
//    orderBlock(block);
    
    auto replacePoints = sortTopologicallyEx(mod.getBodyBlock(), fields );
    OpBuilder builder(mod.getContext());
    for (auto* op : replacePoints) {
      auto wire = cast<WireOp>(op->getOperand(0).getDefiningOp());
        builder.setInsertionPointAfter(op);
        auto node = builder.create<NodeOp>(wire.getLoc(), wire.getType(),op->getOperand(1), wire.getName(), wire.getNameKind(), wire.getAnnotations(), wire.getInnerSym() ? *wire.getInnerSym() : nullptr);
        wire.replaceAllUsesWith(node.getResult());
        wire.erase();
        op->erase();

    }
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
    auto& fields = getAnalysis<FieldSource>();
    Wire2Node transform(getOperation(), fields);
    if (failed(transform.run()))
      signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createWire2NodePass() {
  return std::make_unique<Wire2NodePass>();
}
