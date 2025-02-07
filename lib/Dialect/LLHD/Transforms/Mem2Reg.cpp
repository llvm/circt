//===- Mem2Reg.cpp - Promote signal/memory into values --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"

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
using llvm::SmallMapVector;
using llvm::SmallSetVector;
using llvm::SpecificBumpPtrAllocator;

static bool isEpsilonDelay(Value value) {
  if (auto timeOp = value.getDefiningOp<ConstantTimeOp>()) {
    auto t = timeOp.getValue();
    return t.getTime() == 0 && t.getDelta() == 0 && t.getEpsilon() == 1;
  }
  return false;
}

static bool isUnconditionalBlockingDrive(Operation *op) {
  if (auto driveOp = dyn_cast<DrvOp>(op))
    return !driveOp.getEnable() && isEpsilonDelay(driveOp.getTime());
  return false;
}

static Value lookThroughSubaccesses(Value mem) {
  // TODO
  return mem;
}

namespace {

struct Lattice;
struct LatticeNode;
struct BlockExit;
struct ProbeNode;
struct DriveNode;

struct DefSlot {
  Block *block;
  Type type;
  Value value;

  DefSlot(Block *block, Type type) : block(block), type(type) {}
};

using ReachingDef = PointerUnion<Value, DefSlot *>;

static Value getValue(ReachingDef reachingDef) {
  if (auto value = dyn_cast<Value>(reachingDef))
    return value;
  auto *slot = cast<DefSlot *>(reachingDef);
  if (!slot->value) {
    auto builder = OpBuilder::atBlockBegin(slot->block);
    slot->value = builder
                      .create<UnrealizedConversionCastOp>(
                          UnknownLoc::get(slot->type.getContext()), slot->type,
                          ValueRange{})
                      .getResult(0);
  }
  return slot->value;
}

static void setValue(DefSlot *slot, Value value) {
  if (slot->value) {
    auto placeholder = slot->value.getDefiningOp<UnrealizedConversionCastOp>();
    assert(placeholder && "slot is already set to a non-placeholder value");
    placeholder.getResult(0).replaceAllUsesWith(value);
    placeholder.erase();
  }
  slot->value = value;
}

struct LatticeValue {
  LatticeNode *nodeBefore = nullptr;
  LatticeNode *nodeAfter = nullptr;
  SmallDenseSet<Value, 1> neededDefs;
  SmallDenseMap<Value, ReachingDef, 1> reachingDefs;
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
  SmallVector<std::pair<Value, ReachingDef>, 0> insertedProbes;
  SmallDenseMap<Value, ReachingDef, 1> mergedDefs;

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
  Value slot;
  Value value;

  DriveNode(DrvOp op, LatticeValue *valueBefore, LatticeValue *valueAfter)
      : OpNode(Kind::Drive, op, valueBefore, valueAfter), slot(op.getSignal()),
        value(op.getValue()) {
    assert(isUnconditionalBlockingDrive(op));
  }

  static bool classof(const LatticeNode *n) { return n->kind == Kind::Drive; }
};

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
  DefSlot *createDef(Args... args) {
    return new (defAllocator.Allocate()) DefSlot(std::forward<Args>(args)...);
  }

  void dump(llvm::raw_ostream &os = llvm::dbgs());
  void propagateBackward();
  void propagateBackward(LatticeNode *node);
  void propagateForward();
  void propagateForward(LatticeNode *node);

  /// Mark a lattice node to be updated during propagation.
  void markDirty(LatticeNode *node) {
    assert(node);
    dirtyNodes.insert(node);
  }

  std::vector<LatticeNode *> nodes;
  std::vector<LatticeValue *> values;
  SmallPtrSet<LatticeNode *, 4> dirtyNodes;

private:
  SpecificBumpPtrAllocator<LatticeValue> valueAllocator;
  SpecificBumpPtrAllocator<DefSlot> defAllocator;
  SpecificBumpPtrAllocator<BlockEntry> blockEntryAllocator;
  SpecificBumpPtrAllocator<BlockExit> blockExitAllocator;
  SpecificBumpPtrAllocator<ProbeNode> probeAllocator;
  SpecificBumpPtrAllocator<DriveNode> driveAllocator;

  // Helper function to get the correct allocator given a lattice node class.
  template <class T>
  SpecificBumpPtrAllocator<T> &getAllocator();
  template <>
  SpecificBumpPtrAllocator<BlockEntry> &getAllocator() {
    return blockEntryAllocator;
  }
  template <>
  SpecificBumpPtrAllocator<BlockExit> &getAllocator() {
    return blockExitAllocator;
  }
  template <>
  SpecificBumpPtrAllocator<ProbeNode> &getAllocator() {
    return probeAllocator;
  }
  template <>
  SpecificBumpPtrAllocator<DriveNode> &getAllocator() {
    return driveAllocator;
  }
};

} // namespace

/// Print the lattice in human-readable form. Useful for debugging.
void Lattice::dump(llvm::raw_ostream &os) {
  // Helper functions to quickly come up with unique names for things.
  llvm::MapVector<Block *, unsigned> blockNames;
  llvm::MapVector<Value, unsigned> memNames;
  llvm::MapVector<ReachingDef, unsigned> defNames;

  auto blockName = [&](Block *block) {
    unsigned id = blockNames.insert({block, blockNames.size()}).first->second;
    return std::string("bb") + llvm::utostr(id);
  };

  auto memName = [&](Value value) {
    unsigned id = memNames.insert({value, memNames.size()}).first->second;
    return std::string("mem") + llvm::utostr(id);
  };

  auto defName = [&](ReachingDef def) {
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
          os << " " << memName(mem);
        os << "\n";
      }
      if (!value->reachingDefs.empty()) {
        os << "    -> def";
        for (auto [mem, def] : value->reachingDefs)
          os << " " << memName(mem) << "=" << defName(def);
        os << "\n";
      }
      if (isa<BlockExit>(value->nodeAfter))
        break;

      // Print the node.
      if (auto *node = dyn_cast<ProbeNode>(value->nodeAfter))
        os << "    probe " << memName(node->slot) << "\n";
      else if (auto *node = dyn_cast<DriveNode>(value->nodeAfter))
        os << "    drive " << memName(node->slot) << "\n";
      else
        os << "    unknown\n";

      // Advance to the next node.
      value = cast<OpNode>(value->nodeAfter)->valueAfter;
    }

    // Print the closing braces and successors for the block.
    auto *exit = cast<BlockExit>(value->nodeAfter);
    if (exit->successors.empty()) {
      os << "    halt";
    } else {
      os << "    goto";
      for (auto *node : exit->successors)
        os << " " << blockName(node->block);
    }
    if (exit->suspends)
      os << "  // suspends";
    os << "\n";
  }
  os << "}\n";
}

/// Propagate the lattice values backwards against control flow until a fixed
/// point is reached.
void Lattice::propagateBackward() {
  for (auto *node : nodes)
    propagateBackward(node);
  while (!dirtyNodes.empty()) {
    auto *node = *dirtyNodes.begin();
    dirtyNodes.erase(node);
    propagateBackward(node);
  }
}

void Lattice::propagateBackward(LatticeNode *node) {
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

  // Drives kill the need for a definition to be available, since they provide a
  // definition themselves.
  if (auto *drive = dyn_cast<DriveNode>(node)) {
    auto needed = drive->valueAfter->neededDefs;
    needed.erase(drive->slot);
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
/// point is reached.
void Lattice::propagateForward() {
  for (auto *node : nodes)
    propagateForward(node);
  while (!dirtyNodes.empty()) {
    auto *node = *dirtyNodes.begin();
    dirtyNodes.erase(node);
    propagateForward(node);
  }
}

void Lattice::propagateForward(LatticeNode *node) {
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

  // Drives propagate the driven value as a reaching def.
  if (auto *drive = dyn_cast<DriveNode>(node)) {
    auto reaching = drive->valueBefore->reachingDefs;
    reaching[drive->slot] = drive->value;
    update(drive->valueAfter, reaching);
    return;
  }

  // Block entry points propagate any reaching definitions available in all
  // predecessors, plus any probes inserted locally.
  if (auto *entry = dyn_cast<BlockEntry>(node)) {
    // Propagate reaching definitions for each inserted probe.
    SmallDenseMap<Value, ReachingDef, 1> reaching;
    for (auto [slot, insertedProbe] : entry->insertedProbes)
      reaching[slot] = insertedProbe;

    // Propagate reaching definitions from predecessors, creating new
    // definitions in case of a merge.
    SmallDenseMap<Value, ReachingDef, 1> reachingDefs;
    for (auto *predecessor : entry->predecessors) {
      if (!predecessor->suspends) {
        reachingDefs = predecessor->valueBefore->reachingDefs;
        break;
      }
    }
    for (auto pair : reachingDefs) {
      Value slot = pair.first;
      ReachingDef reachingDef = pair.second;

      // Do not override inserted probes.
      if (reaching.contains(slot))
        continue;

      // Check if all predecessors provide a definition for this slot. If any
      // multiple definitions for the same slot reach us, simply set the
      // `reachingDef` to null such that we can insert a new merge definition.
      if (!llvm::all_of(entry->predecessors, [&](auto *predecessor) {
            if (predecessor->suspends)
              return false;
            auto otherReachingDef =
                predecessor->valueBefore->reachingDefs.lookup(slot);
            if (!otherReachingDef)
              return false;
            if (reachingDef != otherReachingDef)
              reachingDef = nullptr;
            return true;
          }))
        continue;

      // Create a merge definition if different definitions reach us from our
      // predecessors.
      if (!reachingDef)
        reachingDef = entry->mergedDefs.lookup(slot);
      if (!reachingDef) {
        reachingDef = createDef(
            entry->block, cast<hw::InOutType>(slot.getType()).getElementType());
        entry->mergedDefs.insert({slot, reachingDef});
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

namespace {
struct Node;

// struct LatticeValue {
//   unsigned id;
//   SmallVector<Node *, 1> predecessors;
//   SmallVector<Node *, 1> successors;
// };

using Def = PointerUnion<Node *, Value>;

struct Node {
  unsigned id;
  Operation *op = nullptr;
  Block *block = nullptr;
  SmallVector<Node *, 1> before;
  SmallVector<Node *, 1> after;
  SmallVector<Value, 1> escapedRefs;
  SmallMapVector<Value, Def, 2> reachingDefs;
  SmallSetVector<Value, 2> neededDefs;
  SmallDenseMap<Value, PointerIntPair<Value, 1>> localDefs;
};
} // namespace

namespace {
struct Promoter {
  Promoter(Region &region) : region(region) {}
  LogicalResult promote();

  void findPromotableSlots();
  void constructLattice2();

  void insertProbeBlocks();
  void insertProbes();
  void insertProbes(BlockEntry *node);

  void insertDriveBlocks();
  void insertDrives();
  void insertDrives(BlockExit *node);
  void insertDrives(DriveNode *node);

  void resolveDefinitions();
  void resolveDefinitions(BlockEntry *node);
  void resolveDefinitions(ProbeNode *node);

  void constructLattice();
  void propagateLattice();
  bool updateNode(Node *node);
  void findBlockedRefs();
  void updateIR();
  void updateIR(Node *node);
  void debugLattice();

  Region &region;
  Lattice lattice;
  SmallVector<Node *> nodes;
  DenseMap<Block *, Node *> nodeByBlock;
  DenseMap<Operation *, Node *> nodeByOp;
  SmallSetVector<Value, 2> blockedRefs;

  SmallVector<Value> slots;
  SmallDenseMap<Value, unsigned> slotOrder;
};
} // namespace

LogicalResult Promoter::promote() {
  if (region.empty())
    return success();

  findPromotableSlots();
  if (slots.empty())
    return success();

  constructLattice2();
  LLVM_DEBUG({
    llvm::dbgs() << "Initial lattice:\n";
    lattice.dump();
  });

  // Propagate the needed definitions backward across the lattice.
  lattice.propagateBackward();

  // Insert probes wherever a def is needed for the first time.
  insertProbeBlocks();
  insertProbes();
  LLVM_DEBUG({
    llvm::dbgs() << "Backward propagation:\n";
    lattice.dump();
  });

  // Propagate the reaching definitions forward across the lattice.
  lattice.propagateForward();

  // Resolve definitions.
  resolveDefinitions();

  // Insert drives wherever a def can no longer propagate.
  insertDriveBlocks();
  insertDrives();
  LLVM_DEBUG({
    llvm::dbgs() << "Forward propagation:\n";
    lattice.dump();
  });

  // constructLattice();
  // propagateLattice();
  // findBlockedRefs();
  // debugLattice();
  // updateIR();
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
            return isa<PrbOp>(user) || isUnconditionalBlockingDrive(user);
          }))
        continue;

      slotOrder.insert({operand, slots.size()});
      slots.push_back(operand);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << slots.size() << " promotable slots\n");
}

void Promoter::constructLattice2() {
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
        if (!isUnconditionalBlockingDrive(&op))
          continue;
        if (!slotOrder.contains(driveOp.getSignal()))
          continue;
        auto *node = lattice.createNode<DriveNode>(driveOp, valueBefore,
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

/// Insert additional probe blocks where needed. This can happen if a definition
/// is needed in a block which has a suspending and non-suspending predecessor.
/// In that case we would like to insert probes in the predecessor blocks, but
/// cannot do so because of the suspending predecessor.
void Promoter::insertProbeBlocks() {
  // Find all blocks that have any needed definition that can't propagate beyond
  // one of its predecessors. If that's the case, we need an additional probe
  // block after that predecessor.
  SmallDenseSet<std::pair<BlockExit *, BlockEntry *>, 1> worklist;
  for (auto *node : lattice.nodes)
    if (auto *entry = dyn_cast<BlockEntry>(node))
      for (auto *predecessor : entry->predecessors)
        if (llvm::any_of(entry->valueAfter->neededDefs, [&](auto neededDef) {
              return !predecessor->valueBefore->neededDefs.contains(neededDef);
            }))
          worklist.insert({predecessor, entry});

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
    Value value = builder.create<PrbOp>(neededDef.getLoc(), neededDef);
    node->insertedProbes.push_back({neededDef, value});
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
      SmallVector<Value> partialSlots;
      for (auto [slot, reachingDef] : exit->valueBefore->reachingDefs) {
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
  ConstantTimeOp time;
  auto builder = OpBuilder::atBlockTerminator(node->block);
  for (auto slot : slots) {
    auto reachingDef = node->valueBefore->reachingDefs.lookup(slot);
    if (!reachingDef)
      continue;
    if (!node->suspends && !node->successors.empty() &&
        llvm::all_of(node->successors, [&](auto *successor) {
          return successor->valueAfter->reachingDefs.contains(slot);
        }))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "- Inserting drive for " << slot << " before "
                            << *node->terminator << "\n");
    if (!time)
      time = builder.create<ConstantTimeOp>(node->terminator->getLoc(), 0, "ns",
                                            0, 1);
    builder.create<DrvOp>(slot.getLoc(), slot, getValue(reachingDef), time,
                          Value{});
  }
}

/// Remove drives to slots that we are promoting. These have been replaced with
/// new drives at block exits.
void Promoter::insertDrives(DriveNode *node) {
  if (!slotOrder.contains(node->slot))
    return;
  LLVM_DEBUG(llvm::dbgs() << "- Removing drive " << *node->op << "\n");
  auto *delayOp = cast<DrvOp>(node->op).getTime().getDefiningOp();
  node->op->erase();
  node->op = nullptr;
  if (delayOp && isOpTriviallyDead(delayOp))
    delayOp->erase();
}

/// Forward definitions throughout the IR.
void Promoter::resolveDefinitions() {
  for (auto *node : lattice.nodes) {
    if (auto *entry = dyn_cast<BlockEntry>(node))
      resolveDefinitions(entry);
    else if (auto *probe = dyn_cast<ProbeNode>(node))
      resolveDefinitions(probe);
  }
}

/// Forward reaching definitions across a block entry point. This inserts block
/// arguments for any merging definitions, and will also insert successor
/// operands to any ops branching here.
void Promoter::resolveDefinitions(BlockEntry *node) {
  // Determine which slots require a merging definition. Use the `slots` array
  // for this to have a deterministic order for the block arguments.
  SmallVector<Value> neededSlots;
  for (auto slot : slots)
    if (node->mergedDefs.contains(slot))
      neededSlots.push_back(slot);
  if (neededSlots.empty())
    return;
  LLVM_DEBUG(llvm::dbgs() << "- Adding " << neededSlots.size()
                          << " arguments to block " << node->block << "\n");

  // Add the block arguments.
  for (auto slot : neededSlots) {
    auto arg = node->block->addArgument(
        cast<hw::InOutType>(slot.getType()).getElementType(), slot.getLoc());
    setValue(cast<DefSlot *>(node->mergedDefs.lookup(slot)), arg);
  }

  // Add successor operands to the predecessor terminators.
  for (auto *predecessor : node->predecessors) {
    // Collect the interesting reaching definitions in the predecessor.
    SmallVector<Value> reachingDefs;
    for (auto slot : neededSlots) {
      auto reachingDef = predecessor->valueBefore->reachingDefs.lookup(slot);
      assert(reachingDef && "no definition reaches terminator");
      reachingDefs.push_back(getValue(reachingDef));
    }

    // Add the reaching definitions to the branch op.
    auto branchOp = cast<BranchOpInterface>(predecessor->terminator);
    for (auto &blockOperand : branchOp->getBlockOperands())
      if (blockOperand.get() == node->block)
        branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
            .append(reachingDefs);
  }
}

/// Replace probes with the corresponding reaching definition.
void Promoter::resolveDefinitions(ProbeNode *node) {
  if (!slotOrder.contains(node->slot))
    return;
  auto reachingDef = node->valueBefore->reachingDefs.lookup(node->slot);
  assert(reachingDef && "no definition reaches probe");
  LLVM_DEBUG(llvm::dbgs() << "- Replacing " << *node->op << "\n");
  node->op->getResult(0).replaceAllUsesWith(getValue(reachingDef));
  node->op->erase();
  node->op = nullptr;
}

void Promoter::constructLattice() {
  // Create entry nodes for each block.
  SmallDenseMap<Block *, Node *, 8> blockEntries;
  for (auto &block : region) {
    auto *entry = new Node;
    entry->id = nodes.size();
    entry->block = &block;
    nodes.push_back(entry);
    blockEntries[&block] = entry;
    nodeByBlock[&block] = entry;
  }

  // Create nodes for each operation that is relevant for the pass.
  SmallSetVector<Value, 8> escapedRefs;
  for (auto &block : region) {
    auto *nodeBefore = blockEntries.lookup(&block);
    for (auto &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>() || isa<PrbOp>(op) ||
          isUnconditionalBlockingDrive(&op)) {
        auto *node = new Node;
        node->id = nodes.size();
        node->op = &op;
        node->before.push_back(nodeBefore);
        nodeBefore->after.push_back(node);
        nodes.push_back(node);
        for (auto *succ : op.getSuccessors()) {
          auto *nodeAfter = blockEntries.lookup(succ);
          nodeAfter->before.push_back(node);
          node->after.push_back(nodeAfter);
        }
        nodeByOp[&op] = node;
        nodeBefore = node;
        continue;
      }
      op.walk([&](Operation *innerOp) {
        for (auto operand : innerOp->getOperands())
          if (isa<hw::InOutType>(operand.getType()))
            escapedRefs.insert(lookThroughSubaccesses(operand));
      });
      if (!escapedRefs.empty()) {
        auto *node = new Node;
        node->id = nodes.size();
        node->op = &op;
        node->escapedRefs = escapedRefs.takeVector();
        escapedRefs.clear();
        node->before.push_back(nodeBefore);
        nodeBefore->after.push_back(node);
        nodes.push_back(node);
        nodeByOp[&op] = node;
        nodeBefore = node;
        continue;
      }
    }
  }
}

void Promoter::propagateLattice() {
  bool anyChanges = true;
  while (anyChanges) {
    anyChanges = false;
    for (auto *node : nodes)
      if (updateNode(node))
        anyChanges = true;
  }
}

bool Promoter::updateNode(Node *node) {
  // Handle block entry points which merge reaching defs of the nodes before.
  if (node->block) {
    assert(node->after.size() == 1);
    auto *nodeAfter = node->after[0];
    bool anyChanges = node->neededDefs.set_union(nodeAfter->neededDefs);
    if (node->before.empty())
      return anyChanges;

    auto *nodeBefore = node->before.front();
    for (auto [mem, def] : nodeBefore->reachingDefs) {
      for (auto *otherBefore : ArrayRef(node->before).drop_front()) {
        auto otherDef = otherBefore->reachingDefs.lookup(mem);
        if (!otherDef) {
          mem = {};
          def = nullptr;
          break;
        }
        if (otherDef != def)
          def = nullptr;
      }
      if (mem) {
        if (!def)
          def = node;
        auto &slot = node->reachingDefs[mem];
        if (slot != def) {
          slot = def;
          anyChanges = true;
        }
      }
    }
    return anyChanges;
  }
  assert(node->op);
  auto *op = node->op;

  // Handle block exit points which merge needed defs of the nodes after.
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    if (isa<WaitOp, HaltOp>(op))
      return false;

    assert(node->before.size() == 1);
    auto *nodeBefore = node->before[0];
    bool anyChanges = false;
    if (!llvm::equal(node->reachingDefs, nodeBefore->reachingDefs)) {
      node->reachingDefs = nodeBefore->reachingDefs;
      anyChanges = true;
    }

    for (auto *nodeAfter : node->after)
      for (auto mem : nodeAfter->neededDefs)
        if (node->neededDefs.insert(mem))
          anyChanges = true;

    return anyChanges;
  }

  // Handle regular ops.
  assert(node->before.size() == 1);
  assert(node->after.size() == 1);
  auto *nodeBefore = node->before[0];
  auto *nodeAfter = node->after[0];

  // Drives create a new definition for the driven signal, and if only part of
  // the signal is driven, they require an incoming definition to be mutated.
  if (auto driveOp = dyn_cast<DrvOp>(node->op)) {
    auto mem = lookThroughSubaccesses(driveOp.getSignal());
    bool complete = mem == driveOp.getSignal();

    bool anyChanges = false;
    for (auto otherMem : nodeAfter->neededDefs)
      if (mem != otherMem)
        anyChanges |= node->neededDefs.insert(otherMem);
    if (!complete)
      anyChanges |= node->neededDefs.insert(mem);

    for (auto [otherMem, otherDef] : nodeBefore->reachingDefs) {
      if (otherMem != mem) {
        auto &slot = node->reachingDefs[otherMem];
        if (slot != otherDef) {
          slot = otherDef;
          anyChanges = true;
        }
      }
    }

    auto &slot = node->reachingDefs[mem];
    auto def = complete ? Def(driveOp.getValue()) : Def(node);
    if (slot != def) {
      slot = def;
      anyChanges = true;
    }

    return anyChanges;
  }

  // Probes require an incoming definition.
  if (auto probeOp = dyn_cast<PrbOp>(node->op)) {
    auto mem = lookThroughSubaccesses(probeOp.getSignal());

    bool anyChanges = node->neededDefs.set_union(nodeAfter->neededDefs);
    anyChanges |= node->neededDefs.insert(mem);

    if (!llvm::equal(node->reachingDefs, nodeBefore->reachingDefs)) {
      node->reachingDefs = nodeBefore->reachingDefs;
      anyChanges = true;
    }

    return anyChanges;
  }

  // Otherwise, if any refs escape in this op, kill their reaching defs and
  // needed defs.
  bool anyChanges = false;

  for (auto [otherMem, otherDef] : nodeBefore->reachingDefs) {
    if (llvm::is_contained(node->escapedRefs, otherMem))
      continue;
    auto &slot = node->reachingDefs[otherMem];
    if (slot != otherDef) {
      slot = otherDef;
      anyChanges = true;
    }
  }

  for (auto mem : nodeAfter->neededDefs)
    if (!llvm::is_contained(node->escapedRefs, mem))
      if (node->neededDefs.insert(mem))
        anyChanges = true;

  return anyChanges;
}

void Promoter::findBlockedRefs() {
  LLVM_DEBUG(llvm::dbgs() << "Determine blocked refs\n");

  // Ensure that a definition reaches all points that need one.
  for (auto *node : nodes) {
    for (auto mem : node->neededDefs) {
      if (blockedRefs.contains(mem) || node->reachingDefs.contains(mem))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "- Missing definition of " << mem << "\n");
      blockedRefs.insert(mem);
    }
  }

  // Ensure that the reaching definitions at every terminator continue into all
  // its successors, or none of its sucessors. If it continues into none, this
  // is the point where we materialize a final drive. If it continues into all,
  // the final drive will be materialized in the successors.
  for (auto *node : nodes) {
    if (!node->op || !node->op->hasTrait<OpTrait::IsTerminator>())
      continue;
    for (auto [mem, def] : node->reachingDefs) {
      if (blockedRefs.contains(mem))
        continue;
      unsigned count = 0;
      for (auto *nodeAfter : node->after)
        if (nodeAfter->reachingDefs.contains(mem))
          ++count;
      if (count == 0 || count == node->after.size())
        continue;
      LLVM_DEBUG(llvm::dbgs() << "- Some successors lose " << mem << "\n  in "
                              << *node->op << "\n");
      blockedRefs.insert(mem);
    }
  }
}

void Promoter::updateIR() {
  LLVM_DEBUG(llvm::dbgs() << "Updating IR\n");
  for (auto *node : nodes)
    updateIR(node);
}

static Value getDefOrPlaceholder(Value mem, Def def) {
  if (auto value = dyn_cast<Value>(def))
    return value;
  auto *node = cast<Node *>(def);
  auto &slot = node->localDefs[mem];
  if (!slot.getPointer()) {
    auto builder =
        node->op ? OpBuilder(node->op) : OpBuilder::atBlockBegin(node->block);
    slot.setPointer(builder
                        .create<UnrealizedConversionCastOp>(
                            mem.getLoc(),
                            cast<hw::InOutType>(mem.getType()).getElementType(),
                            ValueRange{})
                        .getResult(0));
    slot.setInt(1);
  }
  return slot.getPointer();
}

void Promoter::updateIR(Node *node) {
  // TODO: Materialize probes for needed defs that stop here.

  // Materialize drives for reaching defs that stop here.
  if (auto *op = node->op) {
    assert(node->before.size() == 1);
    auto *nodeBefore = node->before[0];
    OpBuilder builder(op);
    ConstantTimeOp time;
    for (auto [mem, def] : nodeBefore->reachingDefs) {
      if (blockedRefs.contains(mem) || node->reachingDefs.contains(mem))
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "- Driving " << mem << " before " << *op << "\n");
      auto reachingDef = getDefOrPlaceholder(mem, def);
      if (!time)
        time =
            builder.create<ConstantTimeOp>(reachingDef.getLoc(), 0, "ns", 0, 1);
      builder.create<DrvOp>(reachingDef.getLoc(), mem, reachingDef, time,
                            Value{});
    }
  }

  // Insert block arguments.
  if (auto *block = node->block) {
    for (auto [mem, def] : node->reachingDefs) {
      if (blockedRefs.contains(mem) || !node->neededDefs.contains(mem) ||
          def != Def(node))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "- Adding block arg for " << mem << "\n");
      auto arg = block->addArgument(
          cast<hw::InOutType>(mem.getType()).getElementType(), mem.getLoc());

      // Update local definition, replacing the placeholder if there is one.
      auto &localDef = node->localDefs[mem];
      if (localDef.getInt()) {
        localDef.getPointer().replaceAllUsesWith(arg);
        localDef.getPointer().getDefiningOp()->erase();
      }
      localDef.setPointer(arg);
      localDef.setInt(0);

      // Add operands to the predecessor branch ops.
      for (auto *nodeBefore : node->before) {
        auto branchOp = cast<BranchOpInterface>(nodeBefore->op);
        LLVM_DEBUG(llvm::dbgs() << "  - Updating " << branchOp << "\n");
        auto defBefore = nodeBefore->reachingDefs.lookup(mem);
        assert(defBefore);
        auto defValue = getDefOrPlaceholder(mem, defBefore);
        for (auto &blockOperand : branchOp->getBlockOperands())
          if (blockOperand.get() == block)
            branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
                .append(defValue);
      }
    }
    return;
  }

  auto *op = node->op;
  assert(op);

  // Apply drives.
  if (auto driveOp = dyn_cast<DrvOp>(op)) {
    auto mem = lookThroughSubaccesses(driveOp.getSignal());
    if (blockedRefs.contains(mem))
      return;
    LLVM_DEBUG(llvm::dbgs() << "- Removing " << driveOp << "\n");
    assert(node->reachingDefs[mem] == Def(driveOp.getValue()) &&
           "drive to subaccess not supported");
    driveOp.erase();
    return;
  }

  // Replace probes with the reaching def.
  if (auto probeOp = dyn_cast<PrbOp>(op)) {
    auto mem = lookThroughSubaccesses(probeOp.getSignal());
    if (blockedRefs.contains(mem))
      return;
    LLVM_DEBUG(llvm::dbgs() << "- Replacing " << probeOp << "\n");
    assert(mem == probeOp.getSignal() && "probe from subaccess not supported");
    auto reachingDef = node->reachingDefs.lookup(mem);
    assert(reachingDef);
    probeOp.replaceAllUsesWith(getDefOrPlaceholder(mem, reachingDef));
    probeOp.erase();
    return;
  }
}

void Promoter::debugLattice() {
  llvm::MapVector<Block *, unsigned> blockNames;
  llvm::MapVector<Value, unsigned> memNames;
  llvm::MapVector<Def, unsigned> defNames;

  auto blockName = [&](Block *block) {
    unsigned id = blockNames.insert({block, blockNames.size()}).first->second;
    return std::string("block") + llvm::utostr(id);
  };

  auto memName = [&](Value value) {
    unsigned id = memNames.insert({value, memNames.size()}).first->second;
    return std::string("mem") + llvm::utostr(id);
  };

  auto defName = [&](Def def) {
    unsigned id = defNames.insert({def, defNames.size()}).first->second;
    return std::string("def") + llvm::utostr(id);
  };

  for (auto &block : region) {
    llvm::dbgs() << "\n" << blockName(&block) << ":\n";
    auto *node = nodeByBlock.lookup(&block);
    for (auto [mem, def] : node->reachingDefs)
      llvm::dbgs() << "// reaching " << memName(mem) << " = " << defName(def)
                   << "\n";

    for (auto &op : block) {
      auto *node = nodeByOp.lookup(&op);
      if (!node)
        continue;
      for (auto mem : node->neededDefs)
        llvm::dbgs() << "// needs " << memName(mem) << "\n";
      for (auto mem : node->escapedRefs)
        llvm::dbgs() << "// escaped " << memName(mem) << "\n";
      llvm::dbgs() << op << "\n";
      for (auto block : op.getSuccessors())
        llvm::dbgs() << "// successor " << blockName(block) << "\n";
      for (auto [mem, def] : node->reachingDefs)
        llvm::dbgs() << "// reaching " << memName(mem) << " = " << defName(def)
                     << "\n";
    }
  }

  llvm::dbgs() << "\n";
  for (auto [mem, id] : memNames)
    llvm::dbgs() << "// mem" << id << ": " << mem << "\n";
  for (auto [def, id] : defNames) {
    llvm::dbgs() << "// def" << id << ": ";
    if (auto *node = dyn_cast<Node *>(def)) {
      llvm::dbgs() << "node ";
      if (node->block)
        llvm::dbgs() << blockName(node->block) << "\n";
      else
        llvm::dbgs() << *node->op << "\n";
    } else if (auto value = dyn_cast<Value>(def)) {
      llvm::dbgs() << "value " << value << "\n";
    }
  }

  // std::error_code error;
  // llvm::raw_fd_ostream os("debug.dot", error);
  // os << "digraph G {\n";
  // for (auto *node : nodes) {
  //   os << "n" << node->id << " [label='";
  //   for (auto mem : node->neededDefs)
  //     os << "needs " << mem << "\n";
  //   if (node->block)
  //     os << "block-entry";
  //   if (node->op)
  //     os << *node->op;
  //   for (auto [mem, def] : node->reachingDefs)
  //     os << "\nreaching " << mem << " = " << def;
  //   os << "']\n";
  // }
  // for (auto *node : nodes) {
  //   for (auto *nodeAfter : node->after) {
  //     os << "n" << node->id << " -> n" << nodeAfter->id << "\n";
  //   }
  // }
  // os << "}\n";
}

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
