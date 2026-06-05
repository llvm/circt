//===- GatedClockConversion.h - Gated clock conversion ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GatedClockConversion utility class.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
#define CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// GatedClockConversion
//===----------------------------------------------------------------------===//

/// Describes how a source clock drives a destination clock in the flow graph.
enum class EdgeKind { Alias, Gate, InstanceIn, InstanceOut };

/// One edge in the clock flow graph built during backward BFS.
struct ClockEdge {
  Value dst;
  Operation *op; ///< gate or instance op; null for Alias
  EdgeKind kind;
};

/// Sink gated-clock enables into "interesting" ops across module boundaries.
///
/// Usage:
///   1. Construct with the design's `InstanceGraph`.
///   2. Call `addRoot(op)` for every op whose clock should be rewritten
///      (RefForce/RefRelease/Reg/RegReset; *_initial variants are no-ops).
///   3. Call `run()`.
///
/// Algorithm: from each root's clock operand we BFS backward through the
/// IR, looking through node/wire/cast aliases.  At each step:
///     - input-port BlockArgument → fan out to every caller's operand at
///                                  every instance of the module;
///     - `firrtl.int.clock_gate`  → push the gate's base input;
///     - InstanceOp result        → descend into the referenced module and
///                                  push the driver of the matching output
///                                  port;
///     - anything else            → reached the base clock.
///
/// We build a source-to-destination clock mapping (`srcToDstClocks`) during
/// the backward BFS, tracking how clock values flow through the design.
/// Then a forward walk from each base clock materializes (baseClock,
/// AND-of-enables) pairs locally in each module — lazily inserting
/// `(base, enable)` input or output port pairs on intermediate modules as
/// needed.  Each registered root op is then rewritten in place: its clock
/// is replaced with the base, and the AND-reduced enable is folded into
/// the op's predicate / next-state mux per op kind.  Finally, temporary
/// wires created during materialization are eliminated by forwarding their
/// values directly when safe to do so.
///
/// `run()` is sequential and NOT thread-safe — port insertion mutates
/// module signatures and replaces instance ops globally.
class GatedClockConversion {
public:
  explicit GatedClockConversion(InstanceGraph &ig) : ig(ig) {}

  /// Register an op whose clock operand should be sunk.  The clock operand
  /// is identified per op kind:
  ///   RefForceOp / RefReleaseOp        → getClock()
  ///   RegOp / RegResetOp               → getClockVal()
  ///   RefForceInitialOp /
  ///   RefReleaseInitialOp              → no clock; recorded but ignored.
  void addRoot(Operation *op) { roots.push_back(op); }

  /// Perform the worklist + materialisation + IR rewrite on every
  /// registered root.  Sequential; NOT thread-safe.
  LogicalResult run();

  /// Dump the analysis results (srcToDstClocks and baseClks) to llvm::dbgs().
  void dump() const;

private:
  /// Indices of a (base, enable) sibling pair on a module's port list.
  struct PortPair {
    unsigned baseIdx;
    unsigned enableIdx;
  };

  // ── Worklist analysis (no IR mutation) ───────────────────────────────

  /// Reset per-call analysis state (`visited`).
  void clearAnalysis();

  /// BFS from each value in `seeds`, populating `srcToDstClocks`, `baseClks`,
  /// and `visited`.  Pure read of IR.
  void analyzeFrom(ArrayRef<Value> seeds);

  // ── Materialisation (IR-mutating) ────────────────────────────────────

  /// Forward pass that propagates (base, enable) pairs from base clocks through
  /// the clock flow graph, inserting ports and wires as needed.
  void materialize();

  /// Apply the per-op-kind IR rewrite given the materialized
  /// `(base, enable)` pair for `op`'s clock.  No-op when `enable` is null.
  LogicalResult rewriteRoot(Operation *op, Value base, Value enable);

  // ── Helpers ──────────────────────────────────────────────────────────

  /// Collect clock operands from pending roots and prepare them for analysis.
  /// Takes ownership of the pending roots, clearing the `roots` queue.
  /// Returns a pair of (seeds, rootClocks) where seeds are clock values for
  /// analysis and rootClocks map each operation to its original clock value.
  /// Skips operations without clock operands (*_initial variants).
  std::pair<SmallVector<Value>, SmallVector<std::pair<Operation *, Value>>>
  collectSeeds();

  /// Insert a (base, enable) port pair into `mod`'s signature and
  /// propagate the new slots to every existing instance.
  /// If `existingClockIdx` is provided, reuses that port for the clock;
  /// otherwise, creates a new clock port.
  /// If `existingEnableIdx` is provided, reuses that port for the enable;
  /// otherwise, creates a new enable port.
  /// Returns the indices of the clock and enable ports (reused or newly
  /// created).
  PortPair insertOrReusePort(FModuleOp mod, StringRef tag, Direction dir,
                             std::optional<unsigned> existingClockIdx = {},
                             std::optional<unsigned> existingEnableIdx = {});

  /// Materialize `enable | test_enable` for `gate` (just `enable` when
  /// the gate has no test_enable).  Cached.
  Value gateEnableOf(ClockGateIntrinsicOp gate);

  /// Connect materialized clock and enable values to instance ports.
  /// This helper creates MatchingConnectOps to drive the instance's
  /// clock and enable input ports with the materialized values.
  /// If `checkUseEmpty` is true, connections are only created if the
  /// port values are not yet used.
  void connectMaterializedToInstancePorts(InstanceOp inst,
                                          unsigned clkPortIndex,
                                          unsigned enPortIndex,
                                          Value materializedClk,
                                          Value materializedEn, Location loc,
                                          bool checkUseEmpty = false);

  /// Eliminate temporary wires created during materialization.
  /// For each wire in `wireOps`, check if there are exactly two uses:
  /// one connect writing to the wire and one connect reading from it.
  /// If so, and if the write dominates the read, replace the read with
  /// the write source and erase the wire and both connects.
  void eliminateTemporaryWires();

  // ── materialize() dispatched handlers ────────────────────────────────

  /// Handle a wire/node/cast alias edge: create temporary wires in `srcMod`
  /// and record `materialized[dstClk]`.
  void materializeAlias(Value dstClk, FModuleOp srcMod, Value materializedClk,
                        Value materializedEn);

  /// Handle a clock-gate edge: AND the gate enable with any upstream enable
  /// and record `materialized[dstClk]`.
  void materializeGate(ClockGateIntrinsicOp gate, Value dstClk,
                       Value materializedClk, Value materializedEn);

  /// Unified handler for instance input (dir=In) and output (dir=Out) edges.
  /// `liveAnchor` is the live instance result whose result-number gives the
  /// gated-clock port index: `liveValue(dstClk)` for Out, `liveValue(srcClk)`
  /// for In.
  void materializeInstancePort(Direction dir, InstanceOp inst, Value dstClk,
                               Value liveAnchor, Value materializedClk,
                               Value materializedEn);

  /// Re-initialize the gated (base, enable) ports of `inst` when an
  /// InstanceIn destination has already been materialized by another caller
  /// (multiply-instantiated module case).
  void reinitMultiplyInstantiatedInput(Value liveSrc, Value materializedClk,
                                       Value materializedEn);

  /// Look up or create the (base-clock, enable) port pair for `(childMod,
  /// gatedClkIndex)` in direction `dir`, wire connects on first creation, and
  /// return the resulting port indices.  `inst` is updated in-place if a new
  /// instance is cloned to accommodate inserted ports.
  PortPair findOrInsertGatedPorts(InstanceOp &inst, FModuleOp childMod,
                                  unsigned gatedClkIndex, Direction dir,
                                  Value materializedClk, Value materializedEn);

  /// Resolve a value through `valueReplaceMap` to its currently-live SSA value.
  /// `insertOrReusePort` erases the old instance and records old-result →
  /// new-result mappings here; the lookup is purely pointer-based so it is safe
  /// to chase even though the original value's IR storage has been freed.
  Value liveValue(Value v) const {
    while (true) {
      auto it = valueReplaceMap.find(v);
      if (it == valueReplaceMap.end())
        return v;
      v = it->second;
    }
  }

  InstanceGraph &ig;

  /// Roots registered via `addRoot`.
  SmallVector<Operation *> roots;

  // ── Per-call analysis state (cleared by `clearAnalysis`) ─────────────

  /// Dedup set; prevents revisiting a value via aliasing paths.
  DenseSet<Value> visited;

  // ── Long-lived port-plumbing caches ──────────────────────────────────

  /// Track instance replacements as `insertOrReusePort` erases old instances.
  mutable DenseMap<InstanceOp, InstanceOp> instReplaceMap;

  /// Same replacement chain as `instReplaceMap`, but keyed by the raw
  /// `Operation *` so a stale (erased) instance pointer can be resolved to its
  /// live replacement without ever dereferencing the dangling pointer.
  DenseMap<Operation *, Operation *> opReplaceMap;

  /// Chase `opReplaceMap` to the live operation for a possibly-erased op.
  Operation *liveOp(Operation *op) const {
    while (true) {
      auto it = opReplaceMap.find(op);
      if (it == opReplaceMap.end())
        return op;
      op = it->second;
    }
  }

  /// Track per-result SSA value replacements as `insertOrReusePort` erases old
  /// instances.  Maps old (now-dangling) result values to the corresponding
  /// result of the replacement instance, accounting for inserted port indices.
  DenseMap<Value, Value> valueReplaceMap;

  /// Cache of gate → materialized `enable | test_enable`.
  DenseMap<ClockGateIntrinsicOp, Value> gateEnableCache;

  /// Maps source clocks to their driven destination clocks. Built during
  /// backward BFS analysis; each edge carries an explicit EdgeKind.
  DenseMap<Value, SmallVector<ClockEdge>> srcToDstClocks;

  /// List of base clocks (clocks with no further source) found during analysis.
  SmallVector<Value> baseClks;

  /// Maps each clock value to its materialized (base, enable) pair.
  /// Built during forward materialization pass.
  DenseMap<Value, std::pair<Value, Value>> materialized;

  /// Maps (module, port-index) to the inserted (base-port-index,
  /// enable-port-index). Caches port insertions to avoid duplicating ports
  /// across multiple uses.
  DenseMap<std::pair<FModuleOp, unsigned>, std::pair<unsigned, unsigned>>
      modArgToMaterialized;

  /// MLIR context, cached from the first seed value during run().
  MLIRContext *context;

  /// Clock and U1 type, cached to avoid repeated creation.
  Type clockType, u1Type;

  /// Operations to be erased, collected during transformation and erased at
  /// end.
  SmallVector<InstanceOp> opsToErase;

  /// Temporary wires created during transformation. Try to eliminate them.
  SmallVector<WireOp> wireOps;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
