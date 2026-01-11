//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPS_H
#define CIRCT_DIALECT_SYNTH_SYNTHOPS_H

#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.h.inc"

#include "llvm/ADT/PriorityQueue.h"

namespace circt {
namespace synth {
struct AndInverterVariadicOpConversion
    : mlir::OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(aig::AndInverterOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// This function performs a topological sort on the operations within each
/// block of graph regions in the given operation. It uses MLIR's topological
/// sort utility as a wrapper, ensuring that operations are ordered such that
/// all operands are defined before their uses. The `isOperandReady` callback
/// allows customization of when an operand is considered ready for sorting.
LogicalResult topologicallySortGraphRegionBlocks(
    mlir::Operation *op,
    llvm::function_ref<bool(mlir::Value, mlir::Operation *)> isOperandReady);

//===----------------------------------------------------------------------===//
// Delay-Aware Tree Building Utilities
//===----------------------------------------------------------------------===//

/// Helper class for delay-aware tree building.
/// Stores a value along with its arrival time and inversion flag.
class ValueWithArrivalTime {
  /// The value and an optional inversion flag packed together.
  /// The inversion flag is used for AndInverterOp lowering.
  llvm::PointerIntPair<mlir::Value, 1, bool> value;

  /// The arrival time (delay) of this value in the circuit.
  int64_t arrivalTime;

  /// Value numbering for deterministic ordering when arrival times are equal.
  /// This ensures consistent results across runs when multiple values have
  /// the same delay.
  size_t valueNumbering = 0;

public:
  ValueWithArrivalTime(mlir::Value value, int64_t arrivalTime, bool invert,
                       size_t valueNumbering = 0)
      : value(value, invert), arrivalTime(arrivalTime),
        valueNumbering(valueNumbering) {}

  mlir::Value getValue() const { return value.getPointer(); }
  bool isInverted() const { return value.getInt(); }
  int64_t getArrivalTime() const { return arrivalTime; }

  /// Comparison operator for priority queue. Values with earlier arrival times
  /// have higher priority. When arrival times are equal, use value numbering
  /// for determinism.
  bool operator>(const ValueWithArrivalTime &other) const {
    return std::tie(arrivalTime, valueNumbering) >
           std::tie(other.arrivalTime, other.valueNumbering);
  }
};

/// Build a balanced binary tree using a priority queue to greedily pair
/// elements with earliest arrival times. This minimizes the critical path
/// delay.
///
/// Template parameters:
///   T - The element type (must have operator> defined)
///
/// The algorithm uses a min-heap to repeatedly combine the two elements with
/// the earliest arrival times, which is optimal for minimizing maximum delay.
template <typename T>
T buildBalancedTreeWithArrivalTimes(llvm::ArrayRef<T> elements,
                                    llvm::function_ref<T(T, T)> combine) {
  assert(!elements.empty() && "Cannot build tree from empty elements");

  if (elements.size() == 1)
    return elements[0];
  if (elements.size() == 2)
    return combine(elements[0], elements[1]);

  // Min-heap priority queue ordered by operator>
  llvm::PriorityQueue<T, std::vector<T>, std::greater<T>> pq(elements.begin(),
                                                             elements.end());

  // Greedily pair the two earliest-arriving elements
  while (pq.size() > 1) {
    T e1 = pq.top();
    pq.pop();
    T e2 = pq.top();
    pq.pop();

    // Combine the two elements
    T combined = combine(e1, e2);
    pq.push(combined);
  }

  return pq.top();
}

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_H
