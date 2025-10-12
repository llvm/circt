//===- LowerVariadic.cpp - Lowering Variadic to Binary Ops ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers variadic operations to binary operations using a
// delay-aware algorithm for commutative operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PriorityQueue.h"

#define DEBUG_TYPE "synth-lower-variadic"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_LOWERVARIADIC
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

//===----------------------------------------------------------------------===//
// Lower Variadic pass
//===----------------------------------------------------------------------===//

namespace {

/// Helper class for delay-aware variadic operation lowering.
/// Stores a value along with its arrival time for priority queue ordering.
class ValueWithArrivalTime {
  /// The value and an optional inversion flag packed together.
  /// The inversion flag is used for AndInverterOp lowering.
  llvm::PointerIntPair<Value, 1, bool> value;

  /// The arrival time (delay) of this value in the circuit.
  int64_t arrivalTime;

  /// Value numbering for deterministic ordering when arrival times are equal.
  /// This ensures consistent results across runs when multiple values have
  /// the same delay.
  size_t valueNumbering = 0;

public:
  ValueWithArrivalTime(Value value, int64_t arrivalTime, bool invert,
                       size_t valueNumbering)
      : value(value, invert), arrivalTime(arrivalTime),
        valueNumbering(valueNumbering) {}

  Value getValue() const { return value.getPointer(); }
  bool isInverted() const { return value.getInt(); }

  /// Comparison operator for priority queue. Values with earlier arrival times
  /// have higher priority. When arrival times are equal, use value numbering
  /// for determinism.
  bool operator>(const ValueWithArrivalTime &other) const {
    return arrivalTime > other.arrivalTime ||
           (arrivalTime == other.arrivalTime &&
            valueNumbering > other.valueNumbering);
  }
};

struct LowerVariadicPass : public impl::LowerVariadicBase<LowerVariadicPass> {
  using LowerVariadicBase::LowerVariadicBase;
  void runOnOperation() override;
};

} // namespace

/// Construct a balanced binary tree from a variadic operation using a
/// delay-aware algorithm. This function builds the tree by repeatedly combining
/// the two values with the earliest arrival times, which minimizes the critical
/// path delay.
static LogicalResult replaceWithBalancedTree(
    IncrementalLongestPathAnalysis *analysis, mlir::IRRewriter &rewriter,
    Operation *op, llvm::function_ref<bool(OpOperand &)> isInverted,
    llvm::function_ref<Value(ValueWithArrivalTime, ValueWithArrivalTime)>
        createBinaryOp) {
  // Min-heap priority queue ordered by arrival time.
  // Values with earlier arrival times are processed first.
  llvm::PriorityQueue<ValueWithArrivalTime, std::vector<ValueWithArrivalTime>,
                      std::greater<ValueWithArrivalTime>>
      queue;

  // Counter for deterministic ordering when arrival times are equal.
  size_t valueNumber = 0;

  auto push = [&](Value value, bool invert) {
    int64_t delay = 0;
    // If analysis is available, use it to compute the delay.
    // If not available, use zero delay and `valueNumber` will be used instead.
    if (analysis) {
      auto result = analysis->getMaxDelay(value);
      if (failed(result))
        return failure();
      delay = *result;
    }
    ValueWithArrivalTime entry(value, delay, invert, valueNumber++);
    queue.push(entry);
    return success();
  };

  // Enqueue all operands with their arrival times and inversion flags.
  for (size_t i = 0, e = op->getNumOperands(); i < e; ++i)
    if (failed(push(op->getOperand(i), isInverted(op->getOpOperand(i)))))
      return failure();

  // Build balanced tree by repeatedly combining the two earliest values.
  // This greedy approach minimizes the maximum depth of late-arriving signals.
  while (queue.size() >= 2) {
    auto lhs = queue.top();
    queue.pop();
    auto rhs = queue.top();
    queue.pop();
    // Create and enqueue the combined value.
    if (failed(push(createBinaryOp(lhs, rhs), /*inverted=*/false)))
      return failure();
  }

  // Get the final result and replace the original operation.
  auto result = queue.top().getValue();
  rewriter.replaceOp(op, result);
  return success();
}

void LowerVariadicPass::runOnOperation() {
  // Topologically sort operations in graph regions to ensure operands are
  // defined before uses.
  if (failed(synth::topologicallySortGraphRegionBlocks(
          getOperation(), [](Value, Operation *op) -> bool {
            return !isa_and_nonnull<comb::CombDialect, synth::SynthDialect>(
                op->getDialect());
          })))
    return signalPassFailure();

  // Get longest path analysis if timing-aware lowering is enabled.
  synth::IncrementalLongestPathAnalysis *analysis = nullptr;
  if (timingAware.getValue())
    analysis = &getAnalysis<synth::IncrementalLongestPathAnalysis>();

  auto moduleOp = getOperation();

  // Build set of operation names to lower if specified.
  SmallVector<OperationName> names;
  for (const auto &name : opNames)
    names.push_back(OperationName(name, &getContext()));

  // Return true if the operation should be lowered.
  auto shouldLower = [&](Operation *op) {
    // If no names specified, lower all variadic ops.
    if (names.empty())
      return true;
    return llvm::find(names, op->getName()) != names.end();
  };

  mlir::IRRewriter rewriter(&getContext());
  rewriter.setListener(analysis);

  auto result = moduleOp->walk([&](Operation *op) {
    // Skip operations that don't need lowering or are already binary.
    if (!shouldLower(op) || op->getNumOperands() <= 2)
      return WalkResult::advance();

    rewriter.setInsertionPoint(op);

    // Handle AndInverterOp specially to preserve inversion flags.
    if (auto andInverterOp = dyn_cast<aig::AndInverterOp>(op)) {
      auto result = replaceWithBalancedTree(
          analysis, rewriter, op,
          // Check if each operand is inverted.
          [&](OpOperand &operand) {
            return andInverterOp.isInverted(operand.getOperandNumber());
          },
          // Create binary AndInverterOp with inversion flags.
          [&](ValueWithArrivalTime lhs, ValueWithArrivalTime rhs) {
            return rewriter.create<aig::AndInverterOp>(
                op->getLoc(), lhs.getValue(), rhs.getValue(), lhs.isInverted(),
                rhs.isInverted());
          });
      return result.succeeded() ? WalkResult::advance()
                                : WalkResult::interrupt();
    }

    // Handle commutative operations (and, or, xor, mul, add, etc.) using
    // delay-aware lowering to minimize critical path.
    if (isa_and_nonnull<comb::CombDialect>(op->getDialect()) &&
        op->hasTrait<OpTrait::IsCommutative>()) {
      auto result = replaceWithBalancedTree(
          analysis, rewriter, op,
          // No inversion flags for standard commutative operations.
          [](OpOperand &) { return false; },
          // Create binary operation with the same operation type.
          [&](ValueWithArrivalTime lhs, ValueWithArrivalTime rhs) {
            OperationState state(op->getLoc(), op->getName());
            state.addOperands(ValueRange{lhs.getValue(), rhs.getValue()});
            state.addTypes(op->getResult(0).getType());
            auto *newOp = Operation::create(state);
            rewriter.insert(newOp);
            return newOp->getResult(0);
          });
      return result.succeeded() ? WalkResult::advance()
                                : WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();
}
