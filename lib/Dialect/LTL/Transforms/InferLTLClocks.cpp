//===- InferLTLClocks.cpp - Infer clocks for ltl.delay ops ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass propagates clock information from ltl.clock operations down to
// ltl.delay operations that have placeholder clocks.
//
// When a single delay is used by multiple clock ops with different clocks,
// this pass creates separate copies of the delay (and its upstream chain)
// for each clock, ensuring each clock op gets a correctly clocked version.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLPasses.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace ltl {
#define GEN_PASS_DEF_INFERLTLCLOCKS
#include "circt/Dialect/LTL/LTLPasses.h.inc"
} // namespace ltl
} // namespace circt

using namespace circt;
using namespace ltl;
using namespace hw;

namespace {

/// Check if a value is a placeholder clock (hw.constant true).
///
/// We use `hw.constant true` as a placeholder because it is semantically
/// invalid as a real clock: a constant-high signal has no edges (no 0→1 or
/// 1→0 transitions), so there are no sampling points and "clock cycles"
/// cannot be defined. No legitimate hardware design would use a constant
/// as a clock signal for sequential timing.
static bool isPlaceholderClock(Value clock) {
  if (auto constOp = clock.getDefiningOp<hw::ConstantOp>()) {
    // Check if it's a 1-bit constant with value 1 (true)
    auto value = constOp.getValue();
    return value.getBitWidth() == 1 && value.isOne();
  }
  return false;
}

/// Check if a subtree (rooted at value) contains any placeholder delays.
static bool hasPlaceholderDelays(Value value, DenseSet<Operation *> &visited) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || visited.contains(defOp))
    return false;
  visited.insert(defOp);

  if (auto delayOp = dyn_cast<DelayOp>(defOp)) {
    if (isPlaceholderClock(delayOp.getClock()))
      return true;
    return hasPlaceholderDelays(delayOp.getInput(), visited);
  }

  // For clock ops, don't traverse into the clock signal itself
  if (auto clockOp = dyn_cast<ClockOp>(defOp)) {
    return hasPlaceholderDelays(clockOp.getInput(), visited);
  }

  for (Value operand : defOp->getOperands()) {
    if (hasPlaceholderDelays(operand, visited))
      return true;
  }
  return false;
}

/// Recursively clone/update a value's subtree, replacing placeholder delays
/// with properly clocked versions. Returns the new value (may be the same
/// if no changes needed).
static Value cloneWithClock(Value value, Value clock, ClockEdge edge,
                            OpBuilder &builder, IRMapping &mapping,
                            DenseSet<Operation *> &visited) {
  // If already mapped, return the mapped value
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  // If it's a block argument or has no defining op, return as-is
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    mapping.map(value, value);
    return value;
  }

  // If already visited (cycle detection), return mapped or original
  if (visited.contains(defOp)) {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    return value;
  }
  visited.insert(defOp);

  // Handle DelayOp specially
  if (auto delayOp = dyn_cast<DelayOp>(defOp)) {
    // First, recursively process the input
    Value newInput = cloneWithClock(delayOp.getInput(), clock, edge, builder,
                                    mapping, visited);

    if (isPlaceholderClock(delayOp.getClock())) {
      // Create a new delay with the correct clock
      builder.setInsertionPointAfter(defOp);
      auto newDelay = DelayOp::create(
          builder, delayOp.getLoc(), clock,
          ClockEdgeAttr::get(builder.getContext(), edge), newInput,
          delayOp.getDelayAttr(), delayOp.getLengthAttr());
      mapping.map(value, newDelay.getResult());
      return newDelay.getResult();
    } else if (newInput != delayOp.getInput()) {
      // Input changed but clock is not placeholder - clone with new input
      builder.setInsertionPointAfter(defOp);
      auto newDelay = DelayOp::create(
          builder, delayOp.getLoc(), delayOp.getClock(), delayOp.getEdgeAttr(),
          newInput, delayOp.getDelayAttr(), delayOp.getLengthAttr());
      mapping.map(value, newDelay.getResult());
      return newDelay.getResult();
    } else {
      // No changes needed
      mapping.map(value, value);
      return value;
    }
  }

  // Handle ClockOp - don't traverse into clock signal, only input
  if (auto clockOp = dyn_cast<ClockOp>(defOp)) {
    Value newInput = cloneWithClock(clockOp.getInput(), clock, edge, builder,
                                    mapping, visited);
    if (newInput != clockOp.getInput()) {
      builder.setInsertionPointAfter(defOp);
      auto newClockOp = ClockOp::create(builder, clockOp.getLoc(), newInput,
                                        clockOp.getEdge(), clockOp.getClock());
      mapping.map(value, newClockOp.getResult());
      return newClockOp.getResult();
    }
    mapping.map(value, value);
    return value;
  }

  // For other ops, check if any operand needs updating
  SmallVector<Value> newOperands;
  bool anyChanged = false;
  for (Value operand : defOp->getOperands()) {
    Value newOperand =
        cloneWithClock(operand, clock, edge, builder, mapping, visited);
    newOperands.push_back(newOperand);
    if (newOperand != operand)
      anyChanged = true;
  }

  if (!anyChanged) {
    mapping.map(value, value);
    return value;
  }

  // Clone the operation with new operands
  builder.setInsertionPointAfter(defOp);
  Operation *newOp = builder.clone(*defOp, mapping);
  // Update operands (clone uses mapping, but let's be explicit)
  for (auto [idx, newOperand] : llvm::enumerate(newOperands)) {
    newOp->setOperand(idx, newOperand);
  }

  Value newResult = newOp->getResult(0);
  mapping.map(value, newResult);
  return newResult;
}

struct InferLTLClocksPass
    : public circt::ltl::impl::InferLTLClocksBase<InferLTLClocksPass> {
  void runOnOperation() override;
};

} // namespace

void InferLTLClocksPass::runOnOperation() {
  auto module = getOperation();
  OpBuilder builder(module.getContext());
  bool changed = false;

  // Collect all clock ops first to avoid iterator invalidation
  SmallVector<ClockOp> clockOps;
  module.walk([&](ClockOp clockOp) { clockOps.push_back(clockOp); });

  for (ClockOp clockOp : clockOps) {
    // Check if this clock op's input subtree has any placeholder delays
    DenseSet<Operation *> checkVisited;
    if (!hasPlaceholderDelays(clockOp.getInput(), checkVisited))
      continue;

    Value clock = clockOp.getClock();
    ClockEdge edge = clockOp.getEdge();

    // Clone/update the input subtree with the correct clock
    IRMapping mapping;
    DenseSet<Operation *> visited;
    Value newInput = cloneWithClock(clockOp.getInput(), clock, edge, builder,
                                    mapping, visited);

    if (newInput != clockOp.getInput()) {
      clockOp.getInputMutable().assign(newInput);
      changed = true;
    }
  }

  // Clean up dead operations (original delays that are no longer used)
  // We need to iterate until fixpoint because deleting an op may make its
  // operands dead.
  if (changed) {
    bool erased = true;
    while (erased) {
      erased = false;
      SmallVector<Operation *> toErase;
      module.walk([&](Operation *op) {
        // Only clean up LTL ops that have no uses
        if (isa<DelayOp, ConcatOp, AndOp, OrOp, IntersectOp, RepeatOp,
                GoToRepeatOp, NonConsecutiveRepeatOp, NotOp, ImplicationOp,
                UntilOp, EventuallyOp>(op)) {
          if (op->use_empty())
            toErase.push_back(op);
        }
      });
      for (Operation *op : llvm::reverse(toErase)) {
        op->erase();
        erased = true;
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}
