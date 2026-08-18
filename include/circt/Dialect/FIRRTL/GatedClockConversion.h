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
#include "llvm/ADT/MapVector.h"

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
  // -- The plan data model --------------------------------------------

  /// Sentinel `EnableNode` index meaning "no enable at all".
  static constexpr unsigned kNoEnable = ~0u;

  /// A reference to a clock/enable value that can also name values which do not
  /// exist yet (a planned port, a planned wire).
  ///
  /// Inserting a port re-creates every instance of a module, so instance
  /// results are the only values `applyPlan()` invalidates. Hence the rule this
  /// class enforces: name them symbolically as `(instance, resultIndex)`, never
  /// raw.
  class MatRef {
  public:
    enum class Kind {
      None,        ///< Null reference, e.g. "no enable".
      Direct,      ///< A `Value` that `applyPlan()` never invalidates.
      InstResult,  ///< Result #index of an instance (existing or planned port).
      ModuleArg,   ///< Block argument #index of a module (existing or planned).
      PlannedWire, ///< Entry #index of `plannedWireValues`.
      GateEnable,  ///< `gate.enable | gate.test_enable`, lowered on demand.
    };

    MatRef() = default;

    static MatRef direct(Value v) {
      assert(v && "use MatRef() for a null reference");
      assert(!v.getDefiningOp<FInstanceLike>() &&
             "instance results must be symbolic; use MatRef::instResult()");
      MatRef r;
      r.kind = Kind::Direct;
      r.value = v;
      return r;
    }
    static MatRef instResult(FInstanceLike inst, unsigned index) {
      return opRef(Kind::InstResult, inst, index);
    }
    static MatRef moduleArg(FModuleOp mod, unsigned index) {
      return opRef(Kind::ModuleArg, mod, index);
    }
    static MatRef plannedWire(unsigned index) {
      return opRef(Kind::PlannedWire, nullptr, index);
    }
    static MatRef gateEnable(ClockGateIntrinsicOp gate) {
      return opRef(Kind::GateEnable, gate, 0);
    }

    /// Instance results become symbolic refs, every other value is stable.
    static MatRef of(Value v) {
      if (!v)
        return MatRef();
      if (auto inst = v.getDefiningOp<FInstanceLike>())
        return instResult(inst, cast<OpResult>(v).getResultNumber());
      return direct(v);
    }

    Kind getKind() const { return kind; }
    Value getValue() const { return value; }
    unsigned getIndex() const { return index; }
    Operation *getOp() const { return op; }
    ClockGateIntrinsicOp gate() const { return cast<ClockGateIntrinsicOp>(op); }

    void print(llvm::raw_ostream &os) const;

  private:
    static MatRef opRef(Kind kind, Operation *op, unsigned index) {
      MatRef r;
      r.kind = kind;
      r.op = op;
      r.index = index;
      return r;
    }

    Kind kind = Kind::None;
    Value value;             ///< `Direct` only.
    Operation *op = nullptr; ///< Instance / module / gate, by kind.
    unsigned index = 0;      ///< Result, argument or wire index.
  };

  /// Enable accumulation DAG node:
  ///   value(id) = parent == kNoEnable ? term : (value(parent) & term)
  /// Nodes are shared by every consumer of a pair, so a cascade of gates emits
  /// one `and` per gate no matter how many roots it feeds.
  struct EnableNode {
    unsigned parent;
    MatRef term;
    /// Insert the `and` after this value.
    MatRef anchor;
    Location loc;
  };

  /// (baseClk, enable) pair planned for a clock value.
  struct ClockPairPlan {
    MatRef baseClk;
    unsigned enableId = kNoEnable;
  };

  /// Root rewrite applied by `applyPlan()`: clock the op by `baseClk` and sink
  /// `enableId` into it.
  struct RootRewrite {
    Operation *op;
    MatRef baseClk;
    unsigned enableId = kNoEnable;
  };

  /// (baseClock, enable) port pair to append to a module.
  struct PortPairPlan {
    FModuleOp mod;
    /// Clock port this pair shadows (naming only).
    unsigned gatedClkIndex;
    Direction dir;
    /// Final port indices, pre-assigned at planning time.
    unsigned baseIdx, enIdx;
    /// `dir == Out` only: values inside `mod` driving the new output ports.
    MatRef outBaseClk = MatRef();
    unsigned outEnableId = kNoEnable;
  };

  /// The `(module, clock port index)` key of a `PortPairPlan`.
  using PortPlanKey = std::pair<FModuleOp, unsigned>;

  /// Caller-side connects driving a planned *input* port pair.
  struct InstanceDrive {
    InstanceOp inst;
    unsigned baseIdx, enIdx;
    MatRef baseClk;
    /// `kNoEnable` drives a constant 1.
    unsigned enableId = kNoEnable;
  };

  /// Temporary (base clock, enable) carrier wires at the top of `mod`, standing
  /// in for a wire/node alias of a gated clock.
  /// Wires `2*i` / `2*i+1` of `plannedWireValues` belong to `wirePlans[i]`.
  struct WirePairPlan {
    FModuleOp mod;
    MatRef baseClk;
    unsigned enableId;
    Location loc;
  };

  // -- Worklist analysis (no IR mutation) -------------------------------

  // Fails on an undriven clock net, which this utility cannot plan around.
  LogicalResult analyzeFrom(ArrayRef<Value> seeds);

  // Mark every clock value downstream of a clock gate. This answers "does this
  // module clock port need a (base, enable) pair?" before planning starts, so
  // that planning never has to block on a sibling instance. Monotone, hence a
  // single sweep suffices and a cycle saturates instead of diverging.
  void computeGatedClocks();

  // -- Planning (no IR mutation) ----------------------------------------

  void plan();

  // Plan one outgoing edge of `srcClk` and return the value to enqueue next:
  // the destination, or null if it was already planned.
  Value processEdge(const ClockEdge &edge, Value srcClk, FModuleOp srcMod,
                    MatRef baseClk, unsigned enableId);

  // Append an accumulation node and return its id. A node with no parent is a
  // leaf: its value is `term` and the anchor is unused.
  unsigned newEnableNode(unsigned parent, MatRef term, Location loc,
                         MatRef anchor);
  unsigned newEnableLeaf(MatRef term, Location loc) {
    return newEnableNode(kNoEnable, term, loc, MatRef());
  }

  // -- Plan application (the only IR-mutating phase) --------------------

  LogicalResult applyPlan();

  // Sweep A: create the planned carrier wires.
  void createPlannedWires();

  // Sweep B: append the planned ports and re-create every affected instance.
  void insertPlannedPorts();

  // Sweep C: emit the planned expressions, connects and root rewrites.
  LogicalResult emitPlannedIR();

  // Resolve a reference to the live value it names. Only valid once the sweep
  // that creates the referenced value has run.
  Value resolve(MatRef ref);

  // Materialize the accumulated enable of a node, memoized per node so each
  // node emits at most one `and`. Null for `kNoEnable`.
  Value lower(unsigned enableId);

  LogicalResult rewriteRoot(Operation *op, Value baseClk, Value enable);

  // -- Helpers ----------------------------------------------------------

  // Cached: returns `enable | test_enable` or just `enable`.
  Value gateEnableOf(ClockGateIntrinsicOp gate);

  // Cached: a constant 1 at the top of `mod`, so it dominates every use.
  Value getOrCreateConstU1One(FModuleOp mod);

  void connectMaterializedToInstancePorts(InstanceOp inst,
                                          unsigned clkPortIndex,
                                          unsigned enPortIndex,
                                          Value materializedClk,
                                          Value materializedEn);

  // Forward wire to source when exactly one writer dominates one reader.
  void eliminateTemporaryWires();

  // The live instance for `inst`, which `insertPlannedPorts()` may have
  // re-created. A single lookup suffices: all of a module's port pairs are
  // inserted in one call, so an instance is re-created at most once.
  Operation *liveInstance(Operation *inst) const {
    auto *clone = instClones.lookup(inst);
    return clone ? clone : inst;
  }

  void dumpPlan() const;

  // -- plan() dispatched handlers ---------------------------------------

  void planAlias(Value dstClk, FModuleOp srcMod, MatRef baseClk,
                 unsigned enableId);

  void planGate(ClockGateIntrinsicOp gate, Value dstClk, MatRef baseClk,
                unsigned enableId);

  void planInstancePort(Direction dir, InstanceOp inst, Value dstClk,
                        Value srcClk, MatRef baseClk, unsigned enableId);

  // Handle the 2nd-and-later callers of a multiply-instantiated module.
  void planMultiplyInstantiatedInput(Value srcClk, MatRef baseClk,
                                     unsigned enableId);

  // Plan (or look up) the gated port pair of a child module, and record the
  // drive of a planned input pair at `inst`.
  std::pair<unsigned, unsigned>
  planGatedPorts(InstanceOp inst, FModuleOp childMod, unsigned gatedClkIndex,
                 Direction dir, MatRef baseClk, unsigned enableId);

  void recordInstanceDrive(InstanceOp inst, const PortPairPlan &plan,
                           MatRef baseClk, unsigned enableId);

  InstanceGraph &ig;

  SmallVector<std::pair<Operation *, Value>> roots;

  // -- Analysis state and output ----------------------------------------

  DenseSet<Value> visited;

  DenseMap<Value, SmallVector<ClockEdge>> srcToDstClocks;

  SmallVector<Value> baseClks;

  // Every clock value downstream of a clock gate; see `computeGatedClocks()`.
  DenseSet<Value> gatedClocks;

  DenseMap<Value, ClockPairPlan> clockEnablePairs;

  // -- The plan ---------------------------------------------------------

  SmallVector<EnableNode> enableNodes;

  // Values of the wires created by `applyPlan()`, indexed by
  // `MatRef::plannedWire`.
  SmallVector<Value> plannedWireValues;

  // Memoized result of `lower()`, indexed like `enableNodes`.
  SmallVector<Value> loweredEnables;

  SmallVector<RootRewrite> rootRewrites;

  SmallVector<WirePairPlan> wirePlans;

  // The gated port pairs to append, and the order in which each module's pairs
  // are appended. `MapVector` keeps emission deterministic.
  llvm::MapVector<PortPlanKey, PortPairPlan> portPlans;
  llvm::MapVector<FModuleOp, SmallVector<PortPlanKey>> plansPerModule;

  // Per-module port index allocator, seeded with the module's port count.
  DenseMap<FModuleOp, unsigned> nextPortIdx;

  // Keyed by `(caller instance, base port index)`, which de-duplicates repeated
  // drives of the same port pair.
  llvm::MapVector<std::pair<InstanceOp, unsigned>, InstanceDrive>
      instanceDrives;

  // -- applyPlan() state ------------------------------------------------

  DenseMap<ClockGateIntrinsicOp, Value> gateEnableCache;

  // Cache of constant 1 values per module
  DenseMap<FModuleOp, Value> constU1Cache;

  // Old instance -> the instance re-created with the planned ports.
  DenseMap<Operation *, Operation *> instClones;

  // Replaced instances, erased once nothing reads the plan any more.
  SmallVector<InstanceOp> deadInstances;

  // Temporary carrier wires awaiting elimination.
  SmallVector<WireOp> wireOps;

  MLIRContext *context;

  Type clockType, u1Type;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
