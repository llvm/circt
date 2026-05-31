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
#include "circt/Dialect/Synth/SynthOpInterfaces.h"
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
#include <vector>

namespace circt {
namespace synth {
void populateVariadicAndInverterLoweringPatterns(
    mlir::RewritePatternSet &patterns);
void populateVariadicXorInverterLoweringPatterns(
    mlir::RewritePatternSet &patterns);
bool isLogicNetworkOp(mlir::Operation *op);

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

  ValueWithArrivalTime &flipInversion() {
    value.setInt(!value.getInt());
    return *this;
  }

  /// Comparison operator for priority queue. Values with earlier arrival times
  /// have higher priority. When arrival times are equal, use value numbering
  /// for determinism.
  bool operator>(const ValueWithArrivalTime &other) const {
    return std::tie(arrivalTime, valueNumbering) >
           std::tie(other.arrivalTime, other.valueNumbering);
  }
};

/// Build a balanced binary tree using a priority queue to greedily pair the
/// cheapest elements first. `T::operator>` defines the heap ordering, which is
/// typically based on arrival time and a deterministic tie-breaker. `combine`
/// creates the parent node for each selected pair and computes its new cost.
///
/// Template parameters:
///   T - The element type (must have operator> defined)
///   InlineCapacity - Inline storage capacity for the internal node arena
///
/// The priority queue stores indices into a node arena rather than `T` values
/// so large plan nodes can be kept stable and only moved when appended.
template <typename T, unsigned InlineCapacity = 8>
T buildBalancedTreeWithArrivalTimes(
    llvm::ArrayRef<T> elements,
    llvm::function_ref<T(const T &, const T &)> combine) {
  assert(!elements.empty() && "Cannot build tree from empty elements");

  if (elements.size() == 1)
    return elements[0];
  if (elements.size() == 2)
    return combine(elements[0], elements[1]);

  llvm::SmallVector<T, InlineCapacity> nodes(elements.begin(), elements.end());
  nodes.reserve(elements.size() * 2 - 1);

  auto compare = [&](unsigned lhs, unsigned rhs) {
    return nodes[lhs] > nodes[rhs];
  };
  // Min-heap priority queue ordered by the referenced node. Store indices
  // rather than values to avoid repeatedly moving large node payloads.
  llvm::PriorityQueue<unsigned, std::vector<unsigned>, decltype(compare)> pq(
      compare);
  for (unsigned i = 0, e = nodes.size(); i != e; ++i)
    pq.push(i);

  // Greedily pair the two best-ranked elements.
  while (pq.size() > 1) {
    unsigned e1 = pq.top();
    pq.pop();
    unsigned e2 = pq.top();
    pq.pop();

    nodes.push_back(combine(nodes[e1], nodes[e2]));
    pq.push(nodes.size() - 1);
  }

  return nodes[pq.top()];
}

/// Evaluate the Boolean function `x ^ (z | (x & y))`.
template <typename T>
T evaluateDotLogic(const T &x, const T &y, const T &z) {
  return x ^ (z | (x & y));
}

template <typename T>
T evaluateMajorityLogic(const T &a, const T &b, const T &c) {
  return (a & b) | (a & c) | (b & c);
}

inline llvm::APInt invertBooleanLogic(llvm::APInt value) {
  value.flipAllBits();
  return value;
}

inline llvm::KnownBits invertBooleanLogic(llvm::KnownBits value) {
  std::swap(value.Zero, value.One);
  return value;
}

inline llvm::KnownBits applyInputInversion(llvm::KnownBits value,
                                           bool inverted) {
  if (inverted)
    std::swap(value.Zero, value.One);
  return value;
}

template <typename T>
T evaluateOneHotLogic(const T &a, const T &b, const T &c) {
  auto allSet = a & b & c;
  return (a ^ b ^ c) & invertBooleanLogic(allSet);
}

template <typename T>
T evaluateMuxLogic(const T &a, const T &b, const T &c) {
  return (a & b) | (invertBooleanLogic(a) & c);
}

template <typename T>
T evaluateGambleLogic(const T &a, const T &b, const T &c) {
  auto orSet = a | b | c;
  return (a & b & c) | invertBooleanLogic(orSet);
}

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_H
