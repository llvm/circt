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
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/OpDefinition.h"

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
  // Collect all operands with their arrival times and inversion flags
  SmallVector<ValueWithArrivalTime> operands;
  size_t valueNumber = 0;

  for (size_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    int64_t delay = 0;
    // If analysis is available, use it to compute the delay.
    // If not available, use zero delay and `valueNumber` will be used instead.
    if (analysis) {
      auto result = analysis->getMaxDelay(op->getOperand(i));
      if (failed(result))
        return failure();
      delay = *result;
    }
    operands.push_back(ValueWithArrivalTime(op->getOperand(i), delay,
                                            isInverted(op->getOpOperand(i)),
                                            valueNumber++));
  }

  // Use shared tree building utility
  auto result = buildBalancedTreeWithArrivalTimes<ValueWithArrivalTime>(
      operands,
      // Combine: create binary operation and compute new arrival time
      [&](const ValueWithArrivalTime &lhs, const ValueWithArrivalTime &rhs) {
        Value combined = createBinaryOp(lhs, rhs);
        int64_t newDelay = 0;
        if (analysis) {
          auto delayResult = analysis->getMaxDelay(combined);
          if (succeeded(delayResult))
            newDelay = *delayResult;
        }
        return ValueWithArrivalTime(combined, newDelay, false, valueNumber++);
      });

  rewriter.replaceOp(op, result.getValue());
  return success();
}

void LowerVariadicPass::runOnOperation() {
  // Topologically sort operations in graph regions to ensure operands are
  // defined before uses.
  if (!mlir::sortTopologically(
          getOperation().getBodyBlock(), [](Value val, Operation *op) -> bool {
            if (isa_and_nonnull<hw::HWDialect>(op->getDialect()))
              return isa<hw::InstanceOp>(op);
            return !isa_and_nonnull<comb::CombDialect, synth::SynthDialect>(
                op->getDialect());
          })) {
    mlir::emitError(getOperation().getLoc())
        << "Failed to topologically sort graph region blocks";
    return signalPassFailure();
  }

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

  // FIXME: Currently only top-level operations are lowered due to the lack of
  //        topological sorting in across nested regions.
  for (auto &opRef :
       llvm::make_early_inc_range(moduleOp.getBodyBlock()->getOperations())) {
    auto *op = &opRef;
    // Skip operations that don't need lowering or are already binary.
    if (!shouldLower(op) || op->getNumOperands() <= 2)
      continue;

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
            return aig::AndInverterOp::create(
                rewriter, op->getLoc(), lhs.getValue(), rhs.getValue(),
                lhs.isInverted(), rhs.isInverted());
          });
      if (failed(result))
        return signalPassFailure();
      continue;
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
      if (failed(result))
        return signalPassFailure();
    }
  }
}
