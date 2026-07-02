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

  // Typed views of `op` for the kinds that carry one. The cast is checked, so
  // calling the wrong accessor for an edge kind is a hard error.
  ClockGateIntrinsicOp gate() const { return cast<ClockGateIntrinsicOp>(op); }
  InstanceOp instance() const { return cast<InstanceOp>(op); }
};

/// Sink gated-clock enables into ops across module boundaries.
///
/// Algorithm: backward DFS from each root's clock operand through aliases.
/// Build `srcToDstClocks` mapping during BFS. Forward pass materializes
/// (base, AND-of-enables) pairs per module, lazily inserting port pairs.
/// Rewrite each root: replace clock with base, fold enable into predicate/mux.
/// Eliminate temporary wires when safe.
///
/// NOT thread-safe: port insertion mutates module signatures globally.
class GatedClockConversion {
public:
  explicit GatedClockConversion(InstanceGraph &ig) : ig(ig) {}

  LogicalResult addRoot(Operation *op);

  LogicalResult run();

  void dump() const;

private:
  // ── Worklist analysis (no IR mutation) ───────────────────────────────

  void analyzeFrom(ArrayRef<Value> seeds);

  // Returns a cycle node if found, else null.
  Value findClockFlowCycle() const;

  // ── Materialisation (IR-mutating) ────────────────────────────────────

  void materialize();

  // Materialize a single outgoing edge of `srcClk` during the forward pass and
  // return the clock value(s) to enqueue next: the freshly-reachable
  // destination, plus `srcClk` again when a multiply-instantiated input edge
  // must be retried.
  SmallVector<Value, 2> processEdge(const ClockEdge &edge, Value srcClk,
                                    FModuleOp srcMod, Value materializedClk,
                                    Value materializedEn);

  LogicalResult rewriteRoot(Operation *op, Value base, Value enable);

  // ── Helpers ──────────────────────────────────────────────────────────

  // Skips operations without clock operands (*_initial variants).
  std::pair<SmallVector<Value>, SmallVector<std::pair<Operation *, Value>>>
  collectSeeds();

  // Updates all instances of `mod` globally.
  std::pair<unsigned, unsigned> insertPorts(FModuleOp mod, StringRef tag,
                                            Direction dir);

  // Cached: returns `enable | test_enable` or just `enable`.
  Value gateEnableOf(ClockGateIntrinsicOp gate);

  // Cached: returns a constant 1 value for the given module, creating it at the
  // beginning of the module body block if not already cached.
  Value getOrCreateConstU1One(FModuleOp mod);

  void connectMaterializedToInstancePorts(InstanceOp inst,
                                          unsigned clkPortIndex,
                                          unsigned enPortIndex,
                                          Value materializedClk,
                                          Value materializedEn, Location loc,
                                          bool checkUseEmpty = false);

  // Forward wire to source when exactly one writer dominates one reader.
  void eliminateTemporaryWires();

  // ── materialize() dispatched handlers ────────────────────────────────

  void materializeAlias(Value dstClk, FModuleOp srcMod, Value materializedClk,
                        Value materializedEn);

  void materializeGate(ClockGateIntrinsicOp gate, Value dstClk,
                       Value materializedClk, Value materializedEn);

  bool materializeInstancePort(Direction dir, InstanceOp inst, Value dstClk,
                               Value srcClk, Value materializedClk,
                               Value materializedEn);

  // Handle multiply-instantiated modules.
  void reinitMultiplyInstantiatedInput(Value liveSrc, Value materializedClk,
                                       Value materializedEn);

  // Updates `inst` reference if ports are inserted.
  std::pair<unsigned, unsigned>
  findOrInsertGatedPorts(InstanceOp &inst, FModuleOp childMod,
                         unsigned gatedClkIndex, Direction dir,
                         Value materializedClk, Value materializedEn);

  // Pointer-based lookup safe for erased IR.
  Value liveValue(Value v) const {
    while (true) {
      auto it = valueReplaceMap.find(v);
      if (it == valueReplaceMap.end())
        return v;
      v = it->second;
    }
  }

  InstanceGraph &ig;

  // Single-use guard: `run()` may only be called once per instance.
  bool hasRun = false;

  SmallVector<Operation *> roots;

  // ── Per-call analysis state (cleared by `clearAnalysis`) ─────────────

  DenseSet<Value> visited;

  // ── Long-lived port-plumbing caches ──────────────────────────────────

  DenseMap<InstanceOp, InstanceOp> opReplaceMap;

  InstanceOp liveOp(InstanceOp op) const {
    while (true) {
      auto it = opReplaceMap.find(op);
      if (it == opReplaceMap.end())
        return op;
      op = it->second;
    }
  }

  DenseMap<Value, Value> valueReplaceMap;

  DenseMap<ClockGateIntrinsicOp, Value> gateEnableCache;

  DenseMap<Value, SmallVector<ClockEdge>> srcToDstClocks;

  SmallVector<Value> baseClks;

  DenseMap<Value, std::pair<Value, Value>> clockEnablePairs;

  DenseMap<std::pair<FModuleOp, unsigned>, std::pair<unsigned, unsigned>>
      modArgToMaterialized;

  // Cache of constant 1 values per module
  DenseMap<FModuleOp, Value> constU1Cache;

  MLIRContext *context;

  Type clockType, u1Type;

  SmallVector<InstanceOp> opsToErase;

  SmallVector<WireOp> wireOps;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
