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
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/MapVector.h"

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// GatedClockConversion
//===----------------------------------------------------------------------===//

enum class EdgeKind { Gate, InstanceIn, InstanceOut };

/// A clock flow graph edge. Both ends are driver-normal: wires, nodes and
/// casts are already looked through. `op` is the gate, or the instance.
struct ClockEdge {
  Value dst;
  Operation *op;
  EdgeKind kind;

  ClockGateIntrinsicOp gate() const { return cast<ClockGateIntrinsicOp>(op); }
};

/// Sink gated-clock enables into ops across module boundaries.
///
/// Every clock gets a (base clock, enable) pair; a module port carrying a
/// gated clock gets a (base, enable) port pair. Each root is then clocked by
/// its base clock, with the enable sunk into it. A port pair for port `p` is
/// named `_gatedClock_baseClock_p` and `_gatedClock_enable_p`.
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

  /// Fails if `op` is not a supported root.
  LogicalResult addRoot(Operation *op);

  /// Converts the added roots.
  LogicalResult run();

  void dump() const;

private:
  struct Root {
    Operation *op;
    Value clock;
    /// `clockDriver(clock)`, set by `analyze()`.
    Value key = {};
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
  /// Fills `gated` and `portPairs` in one walk. Returns the clocks to
  /// materialize, in BFS order.
  SmallVector<Value> markGatedAndSelectPortPairs();
  void insertPorts();
  void materialize(ArrayRef<Value> order);
  void rewriteRoots();

  void rewriteRoot(const Root &root, Value baseClk, Value enable);
  Value gateEnableOf(ClockGateIntrinsicOp gate);
  /// `upstream & gateEn`, built right before `gate`. `gateEn` must dominate
  /// `gate`.
  Value andEnables(Value upstream, Value gateEn, ClockGateIntrinsicOp gate);
  Value getOrCreateConstU1One(FModuleOp mod);
  /// A null `en` drives a constant 1.
  void drivePair(Location loc, Value dstClk, Value dstEn, Value clk, Value en);
  /// `v`, or a bounce wire carrying it if `v` does not dominate `user`.
  Value bounceValue(Operation *user, Value v);
  /// Maps a value from before `insertPorts()` to the value it stands for now.
  Value live(Value v) const;
  Operation *liveInstance(Operation *inst) const {
    auto *clone = instClones.lookup(inst);
    return clone ? clone : inst;
  }

  InstanceGraph &ig;

  SmallVector<Root> roots;

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

  /// A value mapped to the bounce wire that carries it to earlier users.
  DenseMap<Value, Value> bounceWires;

  mlir::DominanceInfo domInfo;

  MLIRContext *context;

  Type clockType, u1Type;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_GATEDCLOCKCONVERSION_H
