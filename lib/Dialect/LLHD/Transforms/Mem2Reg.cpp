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

/// Check whether an operation is a `llhd.drive` with no enable condition and an
/// epsilon delay. This corresponds to a blocking assignment in Verilog.
static bool isUnconditionalBlockingDrive(Operation *op) {
  if (auto driveOp = dyn_cast<DrvOp>(op))
    return !driveOp.getEnable() && isEpsilonDelay(driveOp.getTime());
  return false;
}

//===----------------------------------------------------------------------===//
// Reaching Definitions and Placeholders
//===----------------------------------------------------------------------===//

namespace {
/// A definition for a memory slot that does not yet have a concrete SSA value.
/// These are created for blocks which need to merge distinct definitions for
/// the same slot coming from its predecssors, as a standin before block
/// arguments are created.
struct DefSlot {
  Block *block;
  Type type;
  Value value;
  DefSlot(Block *block, Type type) : block(block), type(type) {}
};
} // namespace

/// A definition for a memory slot. Either a concrete SSA `Value`, or a
/// placeholder `DefSlot` for a value yet to be created.
using Def = PointerUnion<Value, DefSlot *>;

/// Return the SSA `Value` for a reaching definition, or create a placeholder
/// value if it is a `DefSlot` for which no value exists yet.
static Value getValue(Def reachingDef) {
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

/// Assign a concrete SSA `Value` to a `DefSlot`. This replaces the placeholder
/// value that might have been created for the slot. This is used by blocks once
/// they construct actual block arguments, to make the block argument provide
/// the actual value for a memory slot.
static void setValue(DefSlot *slot, Value value) {
  if (slot->value) {
    auto placeholder = slot->value.getDefiningOp<UnrealizedConversionCastOp>();
    assert(placeholder && "slot is already set to a non-placeholder value");
    placeholder.getResult(0).replaceAllUsesWith(value);
    placeholder.erase();
  }
  slot->value = value;
}

//===----------------------------------------------------------------------===//
// Lattice to Propagate Needed and Reaching Definitions
//===----------------------------------------------------------------------===//

namespace {

struct LatticeNode;
struct BlockExit;
struct ProbeNode;
struct DriveNode;

struct LatticeValue {
  LatticeNode *nodeBefore = nullptr;
  LatticeNode *nodeAfter = nullptr;
  SmallDenseSet<Value, 1> neededDefs;
  SmallDenseMap<Value, Def, 1> reachingDefs;
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
  SmallVector<std::pair<Value, Def>, 0> insertedProbes;
  SmallDenseMap<Value, Def, 1> mergedDefs;

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
  DefSlot *createDef(Args... args) {
    return new (defAllocator.Allocate()) DefSlot(std::forward<Args>(args)...);
  }

  void dump(llvm::raw_ostream &os = llvm::dbgs());

  /// All nodes in the lattice.
  std::vector<LatticeNode *> nodes;
  /// All values in the lattice.
  std::vector<LatticeValue *> values;

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
  llvm::MapVector<Def, unsigned> defNames;

  auto blockName = [&](Block *block) {
    unsigned id = blockNames.insert({block, blockNames.size()}).first->second;
    return std::string("bb") + llvm::utostr(id);
  };

  auto memName = [&](Value value) {
    unsigned id = memNames.insert({value, memNames.size()}).first->second;
    return std::string("mem") + llvm::utostr(id);
  };

  auto defName = [&](Def def) {
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

//===----------------------------------------------------------------------===//
// Drive/Probe to SSA Value Promotion
//===----------------------------------------------------------------------===//

namespace {
/// The main promoter forwarding drives to probes within a region.
struct Promoter {
  Promoter(Region &region) : region(region) {}
  LogicalResult promote();

  void findPromotableSlots();

  void constructLattice();
  void propagateBackward();
  void propagateBackward(LatticeNode *node);
  void propagateForward();
  void propagateForward(LatticeNode *node);
  void markDirty(LatticeNode *node);

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

  constructLattice();
  LLVM_DEBUG({
    llvm::dbgs() << "Initial lattice:\n";
    lattice.dump();
  });

  // Propagate the needed definitions backward across the lattice.
  propagateBackward();

  // Insert probes wherever a def is needed for the first time.
  insertProbeBlocks();
  insertProbes();
  LLVM_DEBUG({
    llvm::dbgs() << "Backward propagation:\n";
    lattice.dump();
  });

  // Propagate the reaching definitions forward across the lattice.
  propagateForward();

  // Resolve definitions.
  resolveDefinitions();

  // Insert drives wherever a reaching def can no longer propagate.
  insertDriveBlocks();
  insertDrives();
  LLVM_DEBUG({
    llvm::dbgs() << "Forward propagation:\n";
    lattice.dump();
  });

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
void Promoter::propagateForward() {
  for (auto *node : lattice.nodes)
    propagateForward(node);
  while (!dirtyNodes.empty()) {
    auto *node = *dirtyNodes.begin();
    dirtyNodes.erase(node);
    propagateForward(node);
  }
}

/// Propagate the lattice value before a node forward to the value after a node.
void Promoter::propagateForward(LatticeNode *node) {
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
    SmallDenseMap<Value, Def, 1> reaching;
    for (auto [slot, insertedProbe] : entry->insertedProbes)
      reaching[slot] = insertedProbe;

    // Propagate reaching definitions from predecessors, creating new
    // definitions in case of a merge.
    SmallDenseMap<Value, Def, 1> reachingDefs;
    for (auto *predecessor : entry->predecessors) {
      if (!predecessor->suspends) {
        reachingDefs = predecessor->valueBefore->reachingDefs;
        break;
      }
    }
    for (auto pair : reachingDefs) {
      Value slot = pair.first;
      Def reachingDef = pair.second;

      // Do not override inserted probes.
      if (reaching.contains(slot))
        continue;

      // Check if all predecessors provide a definition for this slot. If any
      // multiple definitions for the same slot reach us, simply set the
      // `reachingDef` to null such that we can insert a new merge definition.
      if (!llvm::all_of(entry->predecessors, [&](auto *predecessor) {
            if (predecessor->suspends)
              return false;
            auto otherDef = predecessor->valueBefore->reachingDefs.lookup(slot);
            if (!otherDef)
              return false;
            if (reachingDef != otherDef)
              reachingDef = nullptr;
            return true;
          }))
        continue;

      // Create a merge definition if different definitions reach us from our
      // predecessors.
      if (!reachingDef)
        reachingDef = entry->mergedDefs.lookup(slot);
      if (!reachingDef) {
        reachingDef = lattice.createDef(
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

//===----------------------------------------------------------------------===//
// Drive-to-Probe Forwarding
//===----------------------------------------------------------------------===//

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
