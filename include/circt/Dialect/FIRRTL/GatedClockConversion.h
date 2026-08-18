//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
#define CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// GatedClockConversion
//===----------------------------------------------------------------------===//

enum class EdgeKind { Alias, Gate, InstanceIn, InstanceOut };

struct ClockEdge {
  Value dst;
  Operation *op; // null for Alias
  EdgeKind kind;

  // Checked views of `op`, so using the wrong one for a kind is a hard error.
  ClockGateIntrinsicOp gate() const { return cast<ClockGateIntrinsicOp>(op); }
  InstanceOp instance() const { return cast<InstanceOp>(op); }
};

/// Sink gated-clock enables into ops across module boundaries.
///
/// Backward traversal from each root's clock builds the clock flow graph, a
/// forward pass *plans* the (base, AND-of-enables) pair of every clock value,
/// and `applyPlan()` then materializes the whole plan in one shot.
///
/// Invariant: analysis and planning never mutate the IR, and no plan record is
/// read after the mutation it describes happened. That is what makes stale
/// value remapping structurally impossible; see `MatRef`.
///
/// Preconditions: run after `firrtl-expand-whens`, so every clock net and
/// register has a single driver. Clock loops are not diagnosed here; see
/// `firrtl-check-comb-loops`.
///
/// NOT thread-safe: port insertion mutates module signatures globally.
class GatedClockConversion {
public:
  explicit GatedClockConversion(InstanceGraph &ig) : ig(ig) {}

  LogicalResult addRoot(Operation *op);

  void dump() const;

private:
  // -- Worklist analysis (no IR mutation) -------------------------------

  // Fails on an undriven clock net, which this utility cannot plan around.
  LogicalResult analyzeFrom(ArrayRef<Value> seeds);

  // Mark every clock value downstream of a clock gate. This answers "does this
  // module clock port need a (base, enable) pair?" before planning starts, so
  // that planning never has to block on a sibling instance. Monotone, hence a
  // single sweep suffices and a cycle saturates instead of diverging.
  void computeGatedClocks();

  InstanceGraph &ig;

  SmallVector<std::pair<Operation *, Value>> roots;

  // -- Analysis state and output ----------------------------------------

  DenseSet<Value> visited;

  DenseMap<Value, SmallVector<ClockEdge>> srcToDstClocks;

  SmallVector<Value> baseClks;

  // Every clock value downstream of a clock gate; see `computeGatedClocks()`.
  DenseSet<Value> gatedClocks;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
