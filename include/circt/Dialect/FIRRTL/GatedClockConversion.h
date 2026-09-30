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

enum class EdgeKind { Gate, InstanceIn, InstanceOut };

/// A clock flow graph edge. Both ends are driver-normal: wires, nodes and
/// clock casts are already looked through.
struct ClockEdge {
  Value dst;
  Operation *op;
  EdgeKind kind;

  ClockGateIntrinsicOp gate() const { return cast<ClockGateIntrinsicOp>(op); }
  InstanceOp instance() const { return cast<InstanceOp>(op); }
};

/// Sink gated-clock enables into ops across module boundaries.
///
/// Every clock gets a (base clock, enable) pair; a module port carrying a
/// gated clock gets a (base, enable) port pair. Each root is then clocked by
/// its base clock, with the enable sunk into it.
///
/// Only analysis can fail, and it runs before any mutation. Ports are inserted
/// before any pair is computed, so no stored value is invalidated later.
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

  LogicalResult run();

  void dump() const;

  /// Must-alias analysis over every clock of the circuit, valid right after a
  /// successful `run()`.
  ///
  /// Two values alias iff they are provably driven by the same base clock. A
  /// gate aliases its input, and an appended base port aliases the port it
  /// shadows. Muxes, `clock_div` and `clock_inv` start new base clocks. Values
  /// of a module that alias do so in every instance of it. `false` means "not
  /// proven equal", never "proven different".
  class ClockAliasAnalysis {
  public:
    bool alias(Value a, Value b) const {
      if (a == b)
        return true;
      unsigned na = lookup(a), nb = lookup(b);
      return na != kNoNode && nb != kNoNode && find(na) == find(nb);
    }
    bool isTracked(Value v) const { return nodeOf.contains(v); }
    /// Null if untracked.
    Value getRepresentative(Value v) const;
    /// Null if the class has no base clock, e.g. a port that different
    /// instances drive with different clocks.
    Value getBaseClock(Value v) const;
    /// Empty if untracked.
    SmallVector<Value> aliasSet(Value v) const;
    void print(llvm::raw_ostream &os) const;

  private:
    friend class GatedClockConversion;

    static constexpr unsigned kNoNode = ~0u;

    struct Node {
      Value value;
      mutable unsigned parent;
    };

    /// `shadowPairs` relate the appended base ports to the ports they shadow,
    /// which the IR alone does not when callers disagree.
    void build(InstanceGraph &ig,
               ArrayRef<std::pair<Value, Value>> shadowPairs);

    unsigned lookup(Value v) const { return nodeOf.lookup_or(v, kNoNode); }
    unsigned find(unsigned node) const;
    unsigned track(Value v);
    bool unionClocks(Value a, Value b);
    bool unionNodes(unsigned a, unsigned b);
    void closeOverInstances(InstanceGraph &ig);
    void setBaseClocks(ArrayRef<Value> baseClks);
    void clear();

    SmallVector<Node> nodes;
    DenseMap<Value, unsigned> nodeOf;
    /// Keyed by class root.
    DenseMap<unsigned, SmallVector<unsigned>> members;
    /// Keyed by class root.
    DenseMap<unsigned, Value> classBaseClock;
  };

  bool hasClockAliases() const { return aliasesBuilt; }
  const ClockAliasAnalysis &getClockAliases() const {
    assert(aliasesBuilt && "clock aliases are built by a successful run()");
    return aliases;
  }

private:
  struct Root {
    Operation *op;
    Value clock;
    /// `clockDriver(clock)`, set by `analyze()`.
    Value key = {};
    /// The single write of a register, which the enable is sunk into.
    FConnectLike dataWrite = {};
    unsigned numWrites = 0;

    bool isSinkable() const { return !isa<RegOp, RegResetOp>(op) || dataWrite; }
  };

  /// (base clock, enable); a null enable means ungated.
  using ClockPair = std::pair<Value, Value>;

  struct PortPair {
    Direction dir;
    unsigned baseIdx = 0, enIdx = 0;
    /// Output pairs are driven once for all instances.
    bool driven = false;
  };

  /// A module and the index of the clock port a `PortPair` shadows.
  using PortKey = std::pair<FModuleOp, unsigned>;

  LogicalResult analyze();
  void markGated();
  LogicalResult checkRoots();
  /// Returns the clocks to materialize, in BFS order.
  SmallVector<Value> selectPortPairs();
  void insertPorts();
  void materialize(ArrayRef<Value> order);
  void rewriteRoots();
  void buildClockAliases();

  void rewriteRoot(const Root &root, Value baseClk, Value enable);
  Value gateEnableOf(ClockGateIntrinsicOp gate);
  Value andEnables(Value upstream, Value gateEn, Location loc);
  Value getOrCreateConstU1One(FModuleOp mod);
  /// A null `en` drives a constant 1.
  void drivePair(Location loc, Value dstClk, Value dstEn, Value clk, Value en);
  /// `v`, or a wire carrying it if `v` is defined after `user`.
  Value getDominatingValue(Operation *user, Value v);
  /// Maps a value from before `insertPorts()` to the value it stands for now.
  Value live(Value v) const;
  Operation *liveInstance(Operation *inst) const {
    auto *clone = instClones.lookup(inst);
    return clone ? clone : inst;
  }

  InstanceGraph &ig;

  SmallVector<Root> roots;

  /// Every graph node, mapped to whether a sinkable root depends on it. Nodes
  /// only unsinkable roots reach are never materialized.
  DenseMap<Value, bool> analyzed;

  DenseMap<Value, SmallVector<ClockEdge>> srcToDstClocks;

  SmallVector<Value> baseClks;

  /// Every clock reachable from a base clock, mapped to whether it is gated.
  DenseMap<Value, bool> gated;

  llvm::MapVector<PortKey, PortPair> portPairs;

  /// Keyed by pre-`insertPorts()` values; the pairs themselves are live.
  DenseMap<Value, ClockPair> pairs;

  DenseMap<FModuleOp, Value> constU1Cache;

  DenseMap<Operation *, Operation *> instClones;

  /// Erased last: they are the keys of `pairs`.
  SmallVector<InstanceOp> deadInstances;

  DenseMap<Value, Value> dominatingValues;

  MLIRContext *context;

  Type clockType, u1Type;

  ClockAliasAnalysis aliases;
  bool aliasesBuilt = false;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
