//===- Mem2Reg.cpp - Promote signal/memory slots to values ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#define DEBUG_TYPE "llhd-mem2reg"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_MEM2REGPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::PointerIntPair;
using llvm::SmallDenseSet;
using llvm::SpecificBumpPtrAllocator;

/// Check whether a value is defined by `llhd.constant_time <0ns, 0d, 1e>`.
static bool isEpsilonDelay(Value value) {
  if (auto timeOp = value.getDefiningOp<ConstantTimeOp>()) {
    auto t = timeOp.getValue();
    return t.getTime() == 0 && t.getDelta() == 0 && t.getEpsilon() == 1;
  }
  return false;
}

/// Check whether a value is defined by `llhd.constant_time <0ns, 1d, 0e>`.
static bool isDeltaDelay(Value value) {
  if (auto timeOp = value.getDefiningOp<ConstantTimeOp>()) {
    auto t = timeOp.getValue();
    return t.getTime() == 0 && t.getDelta() == 1 && t.getEpsilon() == 0;
  }
  return false;
}

/// Check whether an operation is a `llhd.drive` with an epsilon delay. This
/// corresponds to a blocking assignment in Verilog.
static bool isBlockingDrive(Operation *op) {
  if (auto driveOp = dyn_cast<DrvOp>(op))
    return isEpsilonDelay(driveOp.getTime());
  return false;
}

/// Check whether an operation is a `llhd.drive` with a delta delay. This
/// corresponds to a non-blocking assignment in Verilog.
static bool isDeltaDrive(Operation *op) {
  if (auto driveOp = dyn_cast<DrvOp>(op))
    return isDeltaDelay(driveOp.getTime());
  return false;
}

//===----------------------------------------------------------------------===//
// Reaching Definitions and Placeholders
//===----------------------------------------------------------------------===//

namespace {
/// Information about whether a definition is driven back onto its signal. For
/// example, probes provide a definition for their signal that does not have to
/// be driven back onto the signal. Drives on the other hand provide a
/// definition that eventually must be driven onto the signal.
struct DriveCondition {
  static DriveCondition never() { return ConditionAndMode(Value{}, Never); }
  static DriveCondition always() { return ConditionAndMode(Value{}, Always); }
  static DriveCondition conditional(Value condition = {}) {
    return ConditionAndMode(condition, Conditional);
  }

  bool isNever() const { return conditionAndMode.getInt() == Never; }
  bool isAlways() const { return conditionAndMode.getInt() == Always; }
  bool isConditional() const {
    return conditionAndMode.getInt() == Conditional;
  }

  Value getCondition() const { return conditionAndMode.getPointer(); }
  void setCondition(Value condition) { conditionAndMode.setPointer(condition); }

  bool operator==(const DriveCondition &other) const {
    return conditionAndMode == other.conditionAndMode;
  }
  bool operator!=(const DriveCondition &other) const {
    return conditionAndMode != other.conditionAndMode;
  }

private:
  enum {
    Never,
    Always,
    Conditional,
  };
  typedef PointerIntPair<Value, 2> ConditionAndMode;
  ConditionAndMode conditionAndMode;

  DriveCondition(ConditionAndMode conditionAndMode)
      : conditionAndMode(conditionAndMode) {}
  friend DenseMapInfo<DriveCondition>;
};

/// A definition for a memory slot that may not yet have a concrete SSA value.
/// These are created for blocks which need to merge distinct definitions for
/// the same slot coming from its predecssors, as a standin before block
/// arguments are created. They are also created for drives, where a concrete
/// value is already available in the form of the driven value.
struct Def {
  Block *block;
  Type type;
  Value value;
  DriveCondition condition;
  bool valueIsPlaceholder = false;
  bool conditionIsPlaceholder = false;

  Def(Value value, DriveCondition condition)
      : block(value.getParentBlock()), type(value.getType()), value(value),
        condition(condition) {}
  Def(Block *block, Type type, DriveCondition condition)
      : block(block), type(type), condition(condition) {}

  Value getValueOrPlaceholder();
  Value getConditionOrPlaceholder();
};
} // namespace

/// Return the SSA value for this definition if it already has one, or create
/// a placeholder value if no value exists yet.
Value Def::getValueOrPlaceholder() {
  if (!value) {
    auto builder = OpBuilder::atBlockBegin(block);
    value = builder
                .create<UnrealizedConversionCastOp>(builder.getUnknownLoc(),
                                                    type, ValueRange{})
                .getResult(0);
    valueIsPlaceholder = true;
  }
  return value;
}

/// Return the drive condition for this definition. Creates a constant false or
/// true SSA value if the drive mode is "never" or "always", respectively. If
/// the mode is "conditional", return the its condition value if it already has
/// one, or create a placeholder value if no value exists yet.
Value Def::getConditionOrPlaceholder() {
  if (!condition.getCondition()) {
    auto builder = OpBuilder::atBlockBegin(block);
    Value value;
    if (condition.isNever()) {
      value = builder.create<hw::ConstantOp>(builder.getUnknownLoc(),
                                             builder.getI1Type(), 0);
    } else if (condition.isAlways()) {
      value = builder.create<hw::ConstantOp>(builder.getUnknownLoc(),
                                             builder.getI1Type(), 1);
    } else {
      value = builder
                  .create<UnrealizedConversionCastOp>(builder.getUnknownLoc(),
                                                      builder.getI1Type(),
                                                      ValueRange{})
                  .getResult(0);
      conditionIsPlaceholder = true;
    }
    condition.setCondition(value);
  }
  return condition.getCondition();
}

// Allow `DriveCondition` to be used as hash map key.
template <>
struct llvm::DenseMapInfo<DriveCondition> {
  static DriveCondition getEmptyKey() {
    return DenseMapInfo<DriveCondition::ConditionAndMode>::getEmptyKey();
  }
  static DriveCondition getTombstoneKey() {
    return DenseMapInfo<DriveCondition::ConditionAndMode>::getTombstoneKey();
  }
  static unsigned getHashValue(DriveCondition d) {
    return DenseMapInfo<DriveCondition::ConditionAndMode>::getHashValue(
        d.conditionAndMode);
  }
  static bool isEqual(DriveCondition lhs, DriveCondition rhs) {
    return lhs == rhs;
  }
};

//===----------------------------------------------------------------------===//
// Lattice to Propagate Needed and Reaching Definitions
//===----------------------------------------------------------------------===//

/// The slot a reaching definition specifies a value for, alongside a bit
/// indicating whether the definition is from a delayed drive or a blocking
/// drive.
using DefSlot = PointerIntPair<Value, 1>;
static DefSlot blockingSlot(Value slot) { return {slot, 0}; }
static DefSlot delayedSlot(Value slot) { return {slot, 1}; }
static Value getSlot(DefSlot slot) { return slot.getPointer(); }
static bool isDelayed(DefSlot slot) { return slot.getInt(); }
static Type getStoredType(DefSlot slot) {
  return cast<hw::InOutType>(slot.getPointer().getType()).getElementType();
}
static Location getLoc(DefSlot slot) { return slot.getPointer().getLoc(); }

namespace {

struct LatticeNode;
struct BlockExit;
struct ProbeNode;
struct DriveNode;

struct LatticeValue {
  LatticeNode *nodeBefore = nullptr;
  LatticeNode *nodeAfter = nullptr;
  SmallDenseSet<Value, 1> neededDefs;
  SmallDenseMap<DefSlot, Def *, 1> reachingDefs;
};

struct LatticeNode {
  enum class Kind { BlockEntry, BlockExit, Probe, Drive };
  const Kind kind;
  LatticeNode(Kind kind) : kind(kind) {}
};

struct BlockEntry : public LatticeNode {
  Block *block;
  LatticeValue *valueAfter;
  SmallVector<BlockExit *, 2> predecessors;
  SmallVector<std::pair<Value, Def *>, 0> insertedProbes;
  SmallDenseMap<DefSlot, Def *, 1> mergedDefs;

  BlockEntry(Block *block, LatticeValue *valueAfter)
      : LatticeNode(Kind::BlockEntry), block(block), valueAfter(valueAfter) {
    assert(!valueAfter->nodeBefore);
    valueAfter->nodeBefore = this;
  }

  static bool classof(const LatticeNode *n) {
    return n->kind == Kind::BlockEntry;
  }
};

struct BlockExit : public LatticeNode {
  Block *block;
  LatticeValue *valueBefore;
  SmallVector<BlockEntry *, 2> successors;
  Operation *terminator;
  bool suspends;

  BlockExit(Block *block, LatticeValue *valueBefore)
      : LatticeNode(Kind::BlockExit), block(block), valueBefore(valueBefore),
        terminator(block->getTerminator()),
        suspends(isa<HaltOp, WaitOp>(terminator)) {
    assert(!valueBefore->nodeAfter);
    valueBefore->nodeAfter = this;
  }

  static bool classof(const LatticeNode *n) {
    return n->kind == Kind::BlockExit;
  }
};

struct OpNode : public LatticeNode {
  Operation *op;
  LatticeValue *valueBefore;
  LatticeValue *valueAfter;

  OpNode(Kind kind, Operation *op, LatticeValue *valueBefore,
         LatticeValue *valueAfter)
      : LatticeNode(kind), op(op), valueBefore(valueBefore),
        valueAfter(valueAfter) {
    assert(!valueBefore->nodeAfter);
    assert(!valueAfter->nodeBefore);
    valueBefore->nodeAfter = this;
    valueAfter->nodeBefore = this;
  }

  static bool classof(const LatticeNode *n) {
    return isa<ProbeNode, DriveNode>(n);
  }
};

struct ProbeNode : public OpNode {
  Value slot;

  ProbeNode(PrbOp op, LatticeValue *valueBefore, LatticeValue *valueAfter)
      : OpNode(Kind::Probe, op, valueBefore, valueAfter), slot(op.getSignal()) {
  }

  static bool classof(const LatticeNode *n) { return n->kind == Kind::Probe; }
};

struct DriveNode : public OpNode {
  DefSlot slot;
  Def *def;

  DriveNode(DrvOp op, Def *def, LatticeValue *valueBefore,
            LatticeValue *valueAfter)
      : OpNode(Kind::Drive, op, valueBefore, valueAfter),
        slot(isDeltaDrive(op) ? delayedSlot(op.getSignal())
                              : blockingSlot(op.getSignal())),
        def(def) {
    assert(isBlockingDrive(op) || isDeltaDrive(op));
  }

  static bool classof(const LatticeNode *n) { return n->kind == Kind::Drive; }
};

/// A lattice of block entry and exit nodes, nodes for relevant operations such
/// as probes and drives, and values flowing between the nodes.
struct Lattice {
  /// Create a new value on the lattice.
  LatticeValue *createValue() {
    auto *value = new (valueAllocator.Allocate()) LatticeValue();
    values.push_back(value);
    return value;
  }

  /// Create a new node on the lattice.
  template <class T, typename... Args>
  T *createNode(Args... args) {
    auto *node =
        new (getAllocator<T>().Allocate()) T(std::forward<Args>(args)...);
    nodes.push_back(node);
    return node;
  }

  /// Create a new reaching definition.
  template <typename... Args>
  Def *createDef(Args... args) {
    auto *def = new (defAllocator.Allocate()) Def(std::forward<Args>(args)...);
    defs.push_back(def);
    return def;
  }

  /// Create a new reaching definition for an existing value in the IR.
  Def *createDef(Value value, DriveCondition mode) {
    auto &slot = defsForValues[{value, mode}];
    if (!slot) {
      slot = new (defAllocator.Allocate()) Def(value, mode);
      defs.push_back(slot);
    }
    return slot;
  }

#ifndef NDEBUG
  void dump(llvm::raw_ostream &os = llvm::dbgs());
#endif

  /// All nodes in the lattice.
  std::vector<LatticeNode *> nodes;
  /// All values in the lattice.
  std::vector<LatticeValue *> values;
  /// All reaching defs in the lattice.
  std::vector<Def *> defs;
  /// The reaching defs for concrete values in the IR. This map is used to
  /// create a single def for the same SSA value to allow for pointer equality
  /// comparisons.
  DenseMap<std::pair<Value, DriveCondition>, Def *> defsForValues;

private:
  SpecificBumpPtrAllocator<LatticeValue> valueAllocator;
  SpecificBumpPtrAllocator<Def> defAllocator;
  SpecificBumpPtrAllocator<BlockEntry> blockEntryAllocator;
  SpecificBumpPtrAllocator<BlockExit> blockExitAllocator;
  SpecificBumpPtrAllocator<ProbeNode> probeAllocator;
  SpecificBumpPtrAllocator<DriveNode> driveAllocator;

  // Helper function to get the correct allocator given a lattice node class.
  template <class T>
  SpecificBumpPtrAllocator<T> &getAllocator();
};

// Specializations for the `getAllocator` template that map node types to the
// correct allocator.
template <>
SpecificBumpPtrAllocator<BlockEntry> &Lattice::getAllocator() {
  return blockEntryAllocator;
}
template <>
SpecificBumpPtrAllocator<BlockExit> &Lattice::getAllocator() {
  return blockExitAllocator;
}
template <>
SpecificBumpPtrAllocator<ProbeNode> &Lattice::getAllocator() {
  return probeAllocator;
}
template <>
SpecificBumpPtrAllocator<DriveNode> &Lattice::getAllocator() {
  return driveAllocator;
}

} // namespace

#ifndef NDEBUG
/// Print the lattice in human-readable form. Useful for debugging.
void Lattice::dump(llvm::raw_ostream &os) {
  // Helper functions to quickly come up with unique names for things.
  llvm::MapVector<Block *, unsigned> blockNames;
  llvm::MapVector<Value, unsigned> memNames;
  llvm::MapVector<Def *, unsigned> defNames;

  auto blockName = [&](Block *block) {
    unsigned id = blockNames.insert({block, blockNames.size()}).first->second;
    return std::string("bb") + llvm::utostr(id);
  };

  auto memName = [&](DefSlot value) {
    unsigned id =
        memNames.insert({getSlot(value), memNames.size()}).first->second;
    return std::string("mem") + llvm::utostr(id) +
           (isDelayed(value) ? "#" : "");
  };

  auto defName = [&](Def *def) {
    unsigned id = defNames.insert({def, defNames.size()}).first->second;
    return std::string("def") + llvm::utostr(id);
  };

  // Ensure the blocks are named in the order they were created.
  for (auto *node : nodes)
    if (auto *entry = dyn_cast<BlockEntry>(node))
      blockName(entry->block);

  // Iterate over all block entry nodes.
  os << "lattice {\n";
  for (auto *node : nodes) {
    auto *entry = dyn_cast<BlockEntry>(node);
    if (!entry)
      continue;

    // Print the opening braces and predecessors for the block.
    os << "  " << blockName(entry->block) << ":";
    if (entry->predecessors.empty()) {
      os << "  // no predecessors";
    } else {
      os << "  // from";
      for (auto *node : entry->predecessors)
        os << " " << blockName(node->block);
    }
    os << "\n";

    // Print all nodes following the block entry, up until the block exit.
    auto *value = entry->valueAfter;
    while (true) {
      // Print the needed defs at this lattice point.
      if (!value->neededDefs.empty()) {
        os << "    -> need";
        for (auto mem : value->neededDefs)
          os << " " << memName(blockingSlot(mem));
        os << "\n";
      }
      if (!value->reachingDefs.empty()) {
        os << "    -> def";
        for (auto [mem, def] : value->reachingDefs) {
          os << " " << memName(mem) << "=" << defName(def);
          if (def->condition.isNever())
            os << "[N]";
          else if (def->condition.isAlways())
            os << "[A]";
          else
            os << "[C]";
        }
        os << "\n";
      }
      if (isa<BlockExit>(value->nodeAfter))
        break;

      // Print the node.
      if (auto *node = dyn_cast<ProbeNode>(value->nodeAfter))
        os << "    probe " << memName(blockingSlot(node->slot)) << "\n";
      else if (auto *node = dyn_cast<DriveNode>(value->nodeAfter))
        os << "    drive " << memName(node->slot) << "\n";
      else
        os << "    unknown\n";

      // Advance to the next node.
      value = cast<OpNode>(value->nodeAfter)->valueAfter;
    }

    // Print the closing braces and successors for the block.
    auto *exit = cast<BlockExit>(value->nodeAfter);
    if (isa<WaitOp>(exit->terminator))
      os << "    wait";
    else if (exit->successors.empty())
      os << "    halt";
    else
      os << "    goto";
    for (auto *node : exit->successors)
      os << " " << blockName(node->block);
    if (exit->suspends)
      os << "  // suspends";
    os << "\n";
  }

  // Dump the memories.
  for (auto [mem, id] : memNames)
    os << "  mem" << id << ": " << mem << "\n";

  os << "}\n";
}
#endif

//===----------------------------------------------------------------------===//
// Drive/Probe to SSA Value Promotion
//===----------------------------------------------------------------------===//

namespace {
/// The main promoter forwarding drives to probes within a region.
struct Promoter {
  Promoter(Region &region) : region(region) {}
  LogicalResult promote();

  void findPromotableSlots();

  void captureAcrossWait();
  void captureAcrossWait(PrbOp probeOp, ArrayRef<WaitOp> waitOps,
                         Liveness &liveness, DominanceInfo &dominance);

  void constructLattice();
  void propagateBackward();
  void propagateBackward(LatticeNode *node);
  void propagateForward(bool optimisticMerges);
  void propagateForward(LatticeNode *node, bool optimisticMerges);
  void markDirty(LatticeNode *node);

  void insertProbeBlocks();
  void insertProbes();
  void insertProbes(BlockEntry *node);

  void insertDriveBlocks();
  void insertDrives();
  void insertDrives(BlockExit *node);
  void insertDrives(DriveNode *node);

  void resolveDefinitions();
  void resolveDefinitions(ProbeNode *node);

  void insertBlockArgs();
  bool insertBlockArgs(BlockEntry *node);
  void replaceValueWith(Value oldValue, Value newValue);

  /// The region we are promoting in.
  Region &region;

  /// The slots we are promoting. Mostly `llhd.sig` ops in practice. This
  /// establishes a deterministic order for slot allocations, such that
  /// everything else in the pass can operate using unordered maps and sets.
  SmallVector<Value> slots;
  /// The inverse of `slots`.
  SmallDenseMap<Value, unsigned> slotOrder;

  /// The lattice used to propagate needed definitions backwards and reaching
  /// definitions forwards.
  Lattice lattice;
  /// A worklist of lattice nodes used within calls to `propagate*`.
  SmallPtrSet<LatticeNode *, 4> dirtyNodes;
};
} // namespace

LogicalResult Promoter::promote() {
  if (region.empty())
    return success();

  findPromotableSlots();
  if (slots.empty())
    return success();

  captureAcrossWait();

  constructLattice();
  LLVM_DEBUG({
    llvm::dbgs() << "Initial lattice:\n";
    lattice.dump();
  });

  // Propagate the needed definitions backward across the lattice.
  propagateBackward();
  LLVM_DEBUG({
    llvm::dbgs() << "After backward propagation:\n";
    lattice.dump();
  });

  // Insert probes wherever a def is needed for the first time.
  insertProbeBlocks();
  insertProbes();
  LLVM_DEBUG({
    llvm::dbgs() << "After probe insertion:\n";
    lattice.dump();
  });

  // Propagate the reaching definitions forward across the lattice.
  propagateForward(true);
  propagateForward(false);
  LLVM_DEBUG({
    llvm::dbgs() << "After forward propagation:\n";
    lattice.dump();
  });

  // Resolve definitions.
  resolveDefinitions();

  // Insert drives wherever a reaching def can no longer propagate.
  insertDriveBlocks();
  insertDrives();
  LLVM_DEBUG({
    llvm::dbgs() << "After def resolution and drive insertion:\n";
    lattice.dump();
  });

  // Insert the necessary block arguments.
  insertBlockArgs();

  return success();
}

/// Identify any promotable slots probed or driven under the current region.
void Promoter::findPromotableSlots() {
  SmallPtrSet<Value, 8> seenSlots;
  region.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (!seenSlots.insert(operand).second)
        continue;

      // Ensure the slot is not used in any way we cannot reason about.
      if (!operand.getDefiningOp<llhd::SignalOp>())
        continue;
      if (!llvm::all_of(operand.getUsers(), [&](auto *user) {
            // We don't support nested probes and drives.
            if (region.isProperAncestor(user->getParentRegion()))
              return false;
            // Ignore uses outside of the region.
            if (user->getParentRegion() != &region)
              return true;
            return isa<PrbOp>(user) || isBlockingDrive(user) ||
                   isDeltaDrive(user);
          }))
        continue;

      slotOrder.insert({operand, slots.size()});
      slots.push_back(operand);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << slots.size() << " promotable slots\n");
}

/// Explicitly capture any probes that are live across an `llhd.wait` as block
/// arguments and destination operand of that wait. This ensures that replacing
/// the probe with a reaching definition later on will capture the value of the
/// reaching definition before the wait.
void Promoter::captureAcrossWait() {
  if (region.hasOneBlock())
    return;

  SmallVector<WaitOp> waitOps;
  for (auto &block : region)
    if (auto waitOp = dyn_cast<WaitOp>(block.getTerminator()))
      waitOps.push_back(waitOp);

  DominanceInfo dominance(region.getParentOp());
  Liveness liveness(region.getParentOp());

  SmallVector<WaitOp> crossingWaitOps;
  for (auto &block : region) {
    for (auto probeOp : block.getOps<PrbOp>()) {
      for (auto waitOp : waitOps)
        if (liveness.getLiveness(waitOp->getBlock())->isLiveOut(probeOp))
          crossingWaitOps.push_back(waitOp);
      if (!crossingWaitOps.empty()) {
        captureAcrossWait(probeOp, crossingWaitOps, liveness, dominance);
        crossingWaitOps.clear();
      }
    }
  }
}

/// Add a probe as block argument to a list of wait ops and update uses of the
/// probe to use the added block arguments as appropriate. This may insert
/// additional block arguments in case the probe and added block arguments both
/// reach the same block.
void Promoter::captureAcrossWait(PrbOp probeOp, ArrayRef<WaitOp> waitOps,
                                 Liveness &liveness, DominanceInfo &dominance) {
  LLVM_DEBUG({
    llvm::dbgs() << "Capture " << probeOp << "\n";
    for (auto waitOp : waitOps)
      llvm::dbgs() << "- Across " << waitOp << "\n";
  });

  // Calculate the merge points for this probe once it gets promoted to block
  // arguments across the wait ops.
  auto &domTree = dominance.getDomTree(&region);
  llvm::IDFCalculatorBase<Block, false> idfCalculator(domTree);

  // Calculate the set of blocks which will define this probe as a distinct
  // value.
  SmallPtrSet<Block *, 4> definingBlocks;
  definingBlocks.insert(probeOp->getBlock());
  for (auto waitOp : waitOps)
    definingBlocks.insert(waitOp.getDest());
  idfCalculator.setDefiningBlocks(definingBlocks);

  // Calculate where the probe is live.
  SmallPtrSet<Block *, 16> liveInBlocks;
  for (auto &block : region)
    if (liveness.getLiveness(&block)->isLiveIn(probeOp))
      liveInBlocks.insert(&block);
  idfCalculator.setLiveInBlocks(liveInBlocks);

  // Calculate the merge points where we will have to insert block arguments for
  // this probe.
  SmallVector<Block *> mergePointsVec;
  idfCalculator.calculate(mergePointsVec);
  SmallPtrSet<Block *, 16> mergePoints(mergePointsVec.begin(),
                                       mergePointsVec.end());
  for (auto waitOp : waitOps)
    mergePoints.insert(waitOp.getDest());
  LLVM_DEBUG(llvm::dbgs() << "- " << mergePoints.size() << " merge points\n");

  // Perform a depth-first search starting at the block containing the probe,
  // which dominates all its uses. When we encounter a block that is a merge
  // point, insert a block argument.
  struct WorklistItem {
    DominanceInfoNode *domNode;
    Value reachingDef;
  };
  SmallVector<WorklistItem> worklist;
  worklist.push_back({domTree.getNode(probeOp->getBlock()), probeOp});

  while (!worklist.empty()) {
    auto item = worklist.pop_back_val();
    auto *block = item.domNode->getBlock();

    // If this block is a merge point, insert a block argument for the probe.
    if (mergePoints.contains(block))
      item.reachingDef =
          block->addArgument(probeOp.getType(), probeOp.getLoc());

    // Replace any uses of the probe in this block with the current reaching
    // definition.
    for (auto &op : *block)
      op.replaceUsesOfWith(probeOp, item.reachingDef);

    // If the terminator of this block branches to a merge point, add the
    // current reaching definition as a destination operand.
    if (auto branchOp = dyn_cast<BranchOpInterface>(block->getTerminator())) {
      for (auto &blockOperand : branchOp->getBlockOperands())
        if (mergePoints.contains(blockOperand.get()))
          branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
              .append(item.reachingDef);
    } else if (auto waitOp = dyn_cast<WaitOp>(block->getTerminator())) {
      if (mergePoints.contains(waitOp.getDest()))
        waitOp.getDestOpsMutable().append(item.reachingDef);
    }

    for (auto *child : item.domNode->children())
      worklist.push_back({child, item.reachingDef});
  }
}

//===----------------------------------------------------------------------===//
// Lattice Construction and Propagation
//===----------------------------------------------------------------------===//

/// Populate the lattice with nodes and values corresponding to the blocks and
/// relevant operations in the region we're promoting.
void Promoter::constructLattice() {
  // Create entry nodes for each block.
  SmallDenseMap<Block *, BlockEntry *, 8> blockEntries;
  for (auto &block : region) {
    auto *entry = lattice.createNode<BlockEntry>(&block, lattice.createValue());
    blockEntries.insert({&block, entry});
  }

  // Create nodes for each operation that is relevant for the pass.
  for (auto &block : region) {
    auto *valueBefore = blockEntries.lookup(&block)->valueAfter;

    // Handle operations.
    for (auto &op : block.without_terminator()) {
      // Handle probes.
      if (auto probeOp = dyn_cast<PrbOp>(op)) {
        if (!slotOrder.contains(probeOp.getSignal()))
          continue;
        auto *node = lattice.createNode<ProbeNode>(probeOp, valueBefore,
                                                   lattice.createValue());
        valueBefore = node->valueAfter;
        continue;
      }

      // Handle drives.
      if (auto driveOp = dyn_cast<DrvOp>(op)) {
        if (!isBlockingDrive(&op) && !isDeltaDrive(&op))
          continue;
        if (!slotOrder.contains(driveOp.getSignal()))
          continue;
        auto condition = DriveCondition::always();
        if (auto enable = driveOp.getEnable())
          condition = DriveCondition::conditional(enable);
        auto *def = lattice.createDef(driveOp.getValue(), condition);
        auto *node = lattice.createNode<DriveNode>(driveOp, def, valueBefore,
                                                   lattice.createValue());
        valueBefore = node->valueAfter;
        continue;
      }
    }

    // Create the exit node for the block.
    auto *exit = lattice.createNode<BlockExit>(&block, valueBefore);
    for (auto *otherBlock : exit->terminator->getSuccessors()) {
      auto *otherEntry = blockEntries.lookup(otherBlock);
      exit->successors.push_back(otherEntry);
      otherEntry->predecessors.push_back(exit);
    }
  }
}

/// Propagate the lattice values backwards against control flow until a fixed
/// point is reached.
void Promoter::propagateBackward() {
  for (auto *node : lattice.nodes)
    propagateBackward(node);
  while (!dirtyNodes.empty()) {
    auto *node = *dirtyNodes.begin();
    dirtyNodes.erase(node);
    propagateBackward(node);
  }
}

/// Propagate the lattice value after a node backward to the value before a
/// node.
void Promoter::propagateBackward(LatticeNode *node) {
  auto update = [&](LatticeValue *value, auto &neededDefs) {
    if (value->neededDefs != neededDefs) {
      value->neededDefs = neededDefs;
      markDirty(value->nodeBefore);
    }
  };

  // Probes need a definition for the probed slot to be available.
  if (auto *probe = dyn_cast<ProbeNode>(node)) {
    auto needed = probe->valueAfter->neededDefs;
    needed.insert(probe->slot);
    update(probe->valueBefore, needed);
    return;
  }

  // Blocking drives kill the need for a definition to be available, since they
  // provide a definition themselves.
  if (auto *drive = dyn_cast<DriveNode>(node)) {
    auto needed = drive->valueAfter->neededDefs;
    if (!isDelayed(drive->slot))
      needed.erase(getSlot(drive->slot));
    update(drive->valueBefore, needed);
    return;
  }

  // Block entries simply trigger updates to all their predecessors.
  if (auto *entry = dyn_cast<BlockEntry>(node)) {
    for (auto *predecessor : entry->predecessors)
      markDirty(predecessor);
    return;
  }

  // Block exits merge any needed definitions from their successors.
  if (auto *exit = dyn_cast<BlockExit>(node)) {
    if (exit->suspends)
      return;
    SmallDenseSet<Value, 1> needed;
    for (auto *successors : exit->successors)
      needed.insert(successors->valueAfter->neededDefs.begin(),
                    successors->valueAfter->neededDefs.end());
    update(exit->valueBefore, needed);
    return;
  }

  assert(false && "unhandled node in backward propagation");
}

/// Propagate the lattice values forwards along with control flow until a fixed
/// point is reached. If `optimisticMerges` is true, block entry points
/// propagate definitions from their predecessors into the block without
/// creating a merge definition even if the definition is not available in all
/// predecessors. This is overly optimistic, but initially helps definitions
/// propagate through loop structures. If `optimisticMerges` is false, block
/// entry points create merge definitions for definitions that are not available
/// in all predecessors.
void Promoter::propagateForward(bool optimisticMerges) {
  for (auto *node : lattice.nodes)
    propagateForward(node, optimisticMerges);
  while (!dirtyNodes.empty()) {
    auto *node = *dirtyNodes.begin();
    dirtyNodes.erase(node);
    propagateForward(node, optimisticMerges);
  }
}

/// Propagate the lattice value before a node forward to the value after a node.
void Promoter::propagateForward(LatticeNode *node, bool optimisticMerges) {
  auto update = [&](LatticeValue *value, auto &reachingDefs) {
    if (value->reachingDefs != reachingDefs) {
      value->reachingDefs = reachingDefs;
      markDirty(value->nodeAfter);
    }
  };

  // Probes simply propagate any reaching defs.
  if (auto *probe = dyn_cast<ProbeNode>(node)) {
    update(probe->valueAfter, probe->valueBefore->reachingDefs);
    return;
  }

  // Drives propagate the driven value as a reaching def. Blocking drives kill
  // earlier non-blocking drives. This reflects Verilog and VHDL behaviour,
  // where a drive sequence like
  //
  //   a <= #10ns 42; // A
  //   a <= 43;       // B
  //   a = 44;        // C
  //
  // would see (B) override (A), because it happens earlier, and (C) override
  // (B), because it in turn happens earlier.
  if (auto *drive = dyn_cast<DriveNode>(node)) {
    auto reaching = drive->valueBefore->reachingDefs;
    reaching[drive->slot] = drive->def;
    if (!isDelayed(drive->slot))
      reaching.erase(delayedSlot(getSlot(drive->slot)));
    update(drive->valueAfter, reaching);
    return;
  }

  // Block entry points propagate any reaching definitions available in all
  // predecessors, plus any probes inserted locally.
  if (auto *entry = dyn_cast<BlockEntry>(node)) {
    // Propagate reaching definitions for each inserted probe.
    SmallDenseMap<DefSlot, Def *, 1> reaching;
    for (auto [slot, insertedProbe] : entry->insertedProbes)
      reaching[blockingSlot(slot)] = insertedProbe;

    // Propagate reaching definitions from predecessors, creating new
    // definitions in case of a merge.
    SmallDenseMap<DefSlot, Def *, 1> reachingDefs;
    for (auto *predecessor : entry->predecessors)
      if (!predecessor->suspends)
        reachingDefs.insert(predecessor->valueBefore->reachingDefs.begin(),
                            predecessor->valueBefore->reachingDefs.end());

    for (auto pair : reachingDefs) {
      DefSlot slot = pair.first;
      Def *reachingDef = pair.second;
      DriveCondition reachingDefCondition = reachingDef->condition;

      // Do not override inserted probes.
      if (reaching.contains(slot))
        continue;

      // Check if all predecessors provide a definition for this slot. If any
      // multiple definitions for the same slot reach us, simply set the
      // `reachingDef` to null such that we can insert a new merge definition.
      // Separately track whether the drive mode of all definitions is
      // identical. This is often the case, for example when the definitions of
      // two unconditional drives converge, and we would like to preserve that
      // both drives were unconditional, even if the driven value differs.
      if (llvm::any_of(entry->predecessors, [&](auto *predecessor) {
            return predecessor->suspends;
          }))
        continue;
      for (auto *predecessor : entry->predecessors) {
        auto otherDef = predecessor->valueBefore->reachingDefs.lookup(slot);
        if (!otherDef && optimisticMerges)
          continue;
        // If the definitions are not identical, indicate that we will have
        // to create a new merge def.
        if (reachingDef != otherDef)
          reachingDef = nullptr;
        // If the definitions have different modes, indicate that we will
        // have to create a conditional drive later.
        auto condition =
            otherDef ? otherDef->condition : DriveCondition::never();
        if (reachingDefCondition != condition)
          reachingDefCondition = DriveCondition::conditional();
      }

      // Create a merge definition if different definitions reach us from our
      // predecessors.
      if (!reachingDef)
        reachingDef = entry->mergedDefs.lookup(slot);
      if (!reachingDef) {
        reachingDef = lattice.createDef(entry->block, getStoredType(slot),
                                        reachingDefCondition);
        entry->mergedDefs.insert({slot, reachingDef});
      } else {
        reachingDef->condition = reachingDefCondition;
      }
      reaching.insert({slot, reachingDef});
    }

    update(entry->valueAfter, reaching);
    return;
  }

  // Block exits simply trigger updates to all their successors.
  if (auto *exit = dyn_cast<BlockExit>(node)) {
    for (auto *successor : exit->successors)
      markDirty(successor);
    return;
  }

  assert(false && "unhandled node in forward propagation");
}

/// Mark a lattice node to be updated during propagation.
void Promoter::markDirty(LatticeNode *node) {
  assert(node);
  dirtyNodes.insert(node);
}

//===----------------------------------------------------------------------===//
// Drive/Probe Insertion
//===----------------------------------------------------------------------===//

/// Insert additional probe blocks where needed. This can happen if a definition
/// is needed in a block which has a suspending and non-suspending predecessor.
/// In that case we would like to insert probes in the predecessor blocks, but
/// cannot do so because of the suspending predecessor.
void Promoter::insertProbeBlocks() {
  // Find all blocks that have any needed definition that can't propagate beyond
  // one of its predecessors. If that's the case, we need an additional probe
  // block after that predecessor.
  SmallDenseSet<std::pair<BlockExit *, BlockEntry *>, 1> worklist;
  for (auto *node : lattice.nodes) {
    if (auto *entry = dyn_cast<BlockEntry>(node)) {
      SmallVector<Value> partialSlots;
      for (auto slot : entry->valueAfter->neededDefs) {
        unsigned numIncoming = 0;
        for (auto *predecessor : entry->predecessors)
          if (predecessor->valueBefore->neededDefs.contains(slot))
            ++numIncoming;
        if (numIncoming != 0 && numIncoming != entry->predecessors.size())
          partialSlots.push_back(slot);
      }
      for (auto *predecessor : entry->predecessors)
        if (llvm::any_of(partialSlots, [&](auto slot) {
              return !predecessor->valueBefore->neededDefs.contains(slot);
            }))
          worklist.insert({predecessor, entry});
    }
  }

  // Insert probe blocks after all blocks we have identified.
  for (auto [predecessor, successor] : worklist) {
    LLVM_DEBUG(llvm::dbgs() << "- Inserting probe block towards " << successor
                            << " after " << *predecessor->terminator << "\n");
    OpBuilder builder(predecessor->terminator);
    auto *newBlock = builder.createBlock(successor->block);
    for (auto oldArg : successor->block->getArguments())
      newBlock->addArgument(oldArg.getType(), oldArg.getLoc());
    builder.create<cf::BranchOp>(predecessor->terminator->getLoc(),
                                 successor->block, newBlock->getArguments());
    for (auto &blockOp : predecessor->terminator->getBlockOperands())
      if (blockOp.get() == successor->block)
        blockOp.set(newBlock);

    // Create new nodes in the lattice for the added block.
    auto *value = lattice.createValue();
    value->neededDefs = successor->valueAfter->neededDefs;
    auto *newEntry = lattice.createNode<BlockEntry>(newBlock, value);
    auto *newExit = lattice.createNode<BlockExit>(newBlock, value);
    newEntry->predecessors.push_back(predecessor);
    newExit->successors.push_back(successor);
    llvm::replace(successor->predecessors, predecessor, newExit);
    llvm::replace(predecessor->successors, successor, newEntry);
  }
}

/// Insert probes wherever a definition is needed for the first time. This is
/// the case in the entry block, after any suspensions, and after operations
/// that have unknown effects on memory slots.
void Promoter::insertProbes() {
  for (auto *node : lattice.nodes) {
    if (auto *entry = dyn_cast<BlockEntry>(node))
      insertProbes(entry);
  }
}

/// Insert probes at the beginning of a block for definitions that are needed in
/// this block but not in its predecessors.
void Promoter::insertProbes(BlockEntry *node) {
  auto builder = OpBuilder::atBlockBegin(node->block);
  for (auto neededDef : slots) {
    if (!node->valueAfter->neededDefs.contains(neededDef))
      continue;
    if (!node->predecessors.empty() &&
        llvm::all_of(node->predecessors, [&](auto *predecessor) {
          return predecessor->valueBefore->neededDefs.contains(neededDef);
        }))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "- Inserting probe for " << neededDef
                            << " in block " << node->block << "\n");
    auto value = builder.create<PrbOp>(neededDef.getLoc(), neededDef);
    auto *def = lattice.createDef(value, DriveCondition::never());
    node->insertedProbes.push_back({neededDef, def});
  }
}

/// Insert additional drive blocks where needed. This can happen if a definition
/// continues into some of a block's successors, but not all of them.
void Promoter::insertDriveBlocks() {
  // Find all blocks that have any reaching definition that can't propagate
  // beyond one of its successors. If that's the case, we need an additional
  // drive block before that successor.
  SmallDenseSet<std::pair<BlockExit *, BlockEntry *>, 1> worklist;
  for (auto *node : lattice.nodes) {
    if (auto *exit = dyn_cast<BlockExit>(node)) {
      SmallVector<DefSlot> partialSlots;
      for (auto [slot, reachingDef] : exit->valueBefore->reachingDefs) {
        if (reachingDef->condition.isNever())
          continue;
        unsigned numContinues = 0;
        for (auto *successor : exit->successors)
          if (successor->valueAfter->reachingDefs.contains(slot))
            ++numContinues;
        if (numContinues != 0 && numContinues != exit->successors.size())
          partialSlots.push_back(slot);
      }
      for (auto *successor : exit->successors)
        if (llvm::any_of(partialSlots, [&](auto slot) {
              return !successor->valueAfter->reachingDefs.contains(slot);
            }))
          worklist.insert({exit, successor});
    }
  }

  // Insert drive blocks before all blocks we have identified.
  for (auto [predecessor, successor] : worklist) {
    LLVM_DEBUG(llvm::dbgs() << "- Inserting drive block towards " << successor
                            << " after " << *predecessor->terminator << "\n");
    OpBuilder builder(predecessor->terminator);
    auto *newBlock = builder.createBlock(successor->block);
    for (auto oldArg : successor->block->getArguments())
      newBlock->addArgument(oldArg.getType(), oldArg.getLoc());
    builder.create<cf::BranchOp>(predecessor->terminator->getLoc(),
                                 successor->block, newBlock->getArguments());
    for (auto &blockOp : predecessor->terminator->getBlockOperands())
      if (blockOp.get() == successor->block)
        blockOp.set(newBlock);

    // Create new nodes in the lattice for the added block.
    auto *value = lattice.createValue();
    value->neededDefs = successor->valueAfter->neededDefs;
    value->reachingDefs = predecessor->valueBefore->reachingDefs;
    auto *newEntry = lattice.createNode<BlockEntry>(newBlock, value);
    auto *newExit = lattice.createNode<BlockExit>(newBlock, value);
    newEntry->predecessors.push_back(predecessor);
    newExit->successors.push_back(successor);
    llvm::replace(successor->predecessors, predecessor, newExit);
    llvm::replace(predecessor->successors, successor, newEntry);
  }
}

/// Insert drives wherever a reaching definition can no longer propagate. This
/// is the before any suspensions and before operations that have unknown
/// effects on memory slots.
void Promoter::insertDrives() {
  for (auto *node : lattice.nodes) {
    if (auto *exit = dyn_cast<BlockExit>(node))
      insertDrives(exit);
    else if (auto *drive = dyn_cast<DriveNode>(node))
      insertDrives(drive);
  }
}

/// Insert drives at block terminators for definitions that do not propagate
/// into successors.
void Promoter::insertDrives(BlockExit *node) {
  auto builder = OpBuilder::atBlockTerminator(node->block);

  ConstantTimeOp epsilonTime;
  ConstantTimeOp deltaTime;
  auto getTime = [&](bool delta) {
    if (delta) {
      if (!deltaTime)
        deltaTime = builder.create<ConstantTimeOp>(node->terminator->getLoc(),
                                                   0, "ns", 1, 0);
      return deltaTime;
    }
    if (!epsilonTime)
      epsilonTime = builder.create<ConstantTimeOp>(node->terminator->getLoc(),
                                                   0, "ns", 0, 1);
    return epsilonTime;
  };

  auto insertDriveForSlot = [&](DefSlot slot) {
    auto reachingDef = node->valueBefore->reachingDefs.lookup(slot);
    if (!reachingDef || reachingDef->condition.isNever())
      return;
    if (!node->suspends && !node->successors.empty() &&
        llvm::all_of(node->successors, [&](auto *successor) {
          return successor->valueAfter->reachingDefs.contains(slot);
        }))
      return;
    LLVM_DEBUG(llvm::dbgs() << "- Inserting drive for " << getSlot(slot) << " "
                            << (isDelayed(slot) ? "(delayed)" : "(blocking)")
                            << " before " << *node->terminator << "\n");
    auto time = getTime(isDelayed(slot));
    auto value = reachingDef->getValueOrPlaceholder();
    auto enable = reachingDef->condition.isConditional()
                      ? reachingDef->getConditionOrPlaceholder()
                      : Value{};
    builder.create<DrvOp>(getLoc(slot), getSlot(slot), value, time, enable);
  };

  for (auto slot : slots)
    insertDriveForSlot(blockingSlot(slot));
  for (auto slot : slots)
    insertDriveForSlot(delayedSlot(slot));
}

/// Remove drives to slots that we are promoting. These have been replaced with
/// new drives at block exits.
void Promoter::insertDrives(DriveNode *node) {
  if (!slotOrder.contains(getSlot(node->slot)))
    return;
  LLVM_DEBUG(llvm::dbgs() << "- Removing drive " << *node->op << "\n");
  auto *delayOp = cast<DrvOp>(node->op).getTime().getDefiningOp();
  node->op->erase();
  node->op = nullptr;
  if (delayOp && isOpTriviallyDead(delayOp))
    delayOp->erase();
}

//===----------------------------------------------------------------------===//
// Drive-to-Probe Forwarding
//===----------------------------------------------------------------------===//

/// Forward definitions throughout the IR.
void Promoter::resolveDefinitions() {
  for (auto *node : lattice.nodes)
    if (auto *probe = dyn_cast<ProbeNode>(node))
      resolveDefinitions(probe);
}

/// Replace probes with the corresponding reaching definition.
void Promoter::resolveDefinitions(ProbeNode *node) {
  if (!slotOrder.contains(node->slot))
    return;
  auto *def = node->valueBefore->reachingDefs.lookup(blockingSlot(node->slot));
  assert(def && "no definition reaches probe");
  LLVM_DEBUG(llvm::dbgs() << "- Replacing " << *node->op << "\n");
  replaceValueWith(node->op->getResult(0), def->getValueOrPlaceholder());
  node->op->erase();
  node->op = nullptr;
}

//===----------------------------------------------------------------------===//
// Block Argument Insertion
//===----------------------------------------------------------------------===//

/// Insert block arguments into the IR.
void Promoter::insertBlockArgs() {
  bool anyArgsInserted = true;
  while (anyArgsInserted) {
    anyArgsInserted = false;
    for (auto *node : lattice.nodes)
      if (auto *entry = dyn_cast<BlockEntry>(node))
        anyArgsInserted |= insertBlockArgs(entry);
  }
}

/// Insert block arguments for any merging definitions for which a placeholder
/// value has been created. Also insert corresponding successor operands to any
/// ops branching here. Returns true if any arguments were inserted.
///
/// This function may create additional placeholders in predecessor blocks.
/// Creating block arguments in a later block may uncover additional arguments
/// to be inserted in a previous one. Therefore this function must be called
/// until no more block arguments are inserted.
bool Promoter::insertBlockArgs(BlockEntry *node) {
  // Determine which slots require a merging definition. Use the `slots` array
  // for this to have a deterministic order for the block arguments. We only
  // insert block arguments for the def's value or drive condition if
  // placeholders have been created for them, indicating that they are actually
  // used.
  enum class Which { Value, Condition };
  SmallVector<std::pair<DefSlot, Which>> neededSlots;
  auto addNeededSlot = [&](DefSlot slot) {
    if (auto *def = node->mergedDefs.lookup(slot)) {
      if (node->valueAfter->reachingDefs.contains(slot)) {
        if (def->valueIsPlaceholder)
          neededSlots.push_back({slot, Which::Value});
        if (def->conditionIsPlaceholder)
          neededSlots.push_back({slot, Which::Condition});
      }
    }
  };
  for (auto slot : slots) {
    addNeededSlot(blockingSlot(slot));
    addNeededSlot(delayedSlot(slot));
  }
  if (neededSlots.empty())
    return false;
  LLVM_DEBUG(llvm::dbgs() << "- Adding " << neededSlots.size()
                          << " args to block " << node->block << "\n");

  // Add the block arguments.
  for (auto [slot, which] : neededSlots) {
    auto *def = node->mergedDefs.lookup(slot);
    assert(def);
    switch (which) {
    case Which::Value: {
      // Create an argument for the definition's value and replace any
      // placeholder we might have created earlier.
      auto *placeholder = def->value.getDefiningOp();
      assert(isa_and_nonnull<UnrealizedConversionCastOp>(placeholder) &&
             "placeholder replaced but valueIsPlaceholder still set");
      auto arg = node->block->addArgument(getStoredType(slot), getLoc(slot));
      replaceValueWith(placeholder->getResult(0), arg);
      placeholder->erase();
      def->value = arg;
      def->valueIsPlaceholder = false;
      break;
    }
    case Which::Condition: {
      // If the definition's drive mode is conditional, create an argument for
      // the drive condition and replace any placeholder we might have created
      // earlier.
      auto *placeholder = def->condition.getCondition().getDefiningOp();
      assert(isa_and_nonnull<UnrealizedConversionCastOp>(placeholder) &&
             "placeholder replaced but conditionIsPlaceholder still set");
      auto conditionArg = node->block->addArgument(
          IntegerType::get(region.getContext(), 1), getLoc(slot));
      replaceValueWith(placeholder->getResult(0), conditionArg);
      placeholder->erase();
      def->condition.setCondition(conditionArg);
      def->conditionIsPlaceholder = false;
      break;
    }
    }
  }

  // Add successor operands to the predecessor terminators.
  for (auto *predecessor : node->predecessors) {
    // Collect the interesting reaching definitions in the predecessor.
    SmallVector<Value> args;
    for (auto [slot, which] : neededSlots) {
      auto *def = predecessor->valueBefore->reachingDefs.lookup(slot);
      auto builder = OpBuilder::atBlockTerminator(predecessor->block);
      switch (which) {
      case Which::Value:
        if (def) {
          args.push_back(def->getValueOrPlaceholder());
        } else {
          auto type = getStoredType(slot);
          auto flatType = builder.getIntegerType(hw::getBitWidth(type));
          Value value =
              builder.create<hw::ConstantOp>(getLoc(slot), flatType, 0);
          if (type != flatType)
            value = builder.create<hw::BitcastOp>(getLoc(slot), type, value);
          args.push_back(value);
        }
        break;
      case Which::Condition:
        if (def) {
          args.push_back(def->getConditionOrPlaceholder());
        } else {
          args.push_back(builder.create<hw::ConstantOp>(
              getLoc(slot), builder.getI1Type(), 0));
        }
        break;
      }
    }

    // Add the reaching definitions to the branch op.
    auto branchOp = cast<BranchOpInterface>(predecessor->terminator);
    for (auto &blockOperand : branchOp->getBlockOperands())
      if (blockOperand.get() == node->block)
        branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
            .append(args);
  }

  return true;
}

/// Replace all uses of an old value with a new value in the IR, and update all
/// mentions of the old value in the lattice to the new value.
void Promoter::replaceValueWith(Value oldValue, Value newValue) {
  oldValue.replaceAllUsesWith(newValue);
  for (auto *def : lattice.defs) {
    if (def->value == oldValue)
      def->value = newValue;
    if (def->condition.isConditional() &&
        def->condition.getCondition() == oldValue)
      def->condition.setCondition(newValue);
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct Mem2RegPass : public llhd::impl::Mem2RegPassBase<Mem2RegPass> {
  void runOnOperation() override;
};
} // namespace

void Mem2RegPass::runOnOperation() {
  SmallVector<Region *> regions;
  getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ProcessOp, FinalOp>(op)) {
      auto &region = op->getRegion(0);
      if (!region.empty())
        regions.push_back(&region);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  for (auto *region : regions)
    if (failed(Promoter(*region).promote()))
      return signalPassFailure();
}
