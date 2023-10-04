//===- LegalizeStateUpdate.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-legalize-state-update"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LEGALIZESTATEUPDATE
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

/// Check if an operation partakes in state accesses.
static bool isOpInteresting(Operation *op) {
  if (isa<StateReadOp, StateWriteOp, CallOpInterface, CallableOpInterface>(op))
    return true;
  if (op->getNumRegions() > 0)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Access Analysis
//===----------------------------------------------------------------------===//

namespace {

enum class AccessType { Read = 0, Write = 1 };

/// A read or write access to a state value.
using Access = llvm::PointerIntPair<Value, 1, AccessType>;

struct BlockAccesses;
struct OpAccesses;

/// A block's access analysis information and graph edges.
struct BlockAccesses {
  BlockAccesses(Block *block) : block(block) {}

  /// The block.
  Block *const block;
  /// The parent op lattice node.
  OpAccesses *parent = nullptr;
  /// The accesses from ops within this block to the block arguments.
  SmallPtrSet<Access, 1> argAccesses;
  /// The accesses from ops within this block to values defined outside the
  /// block.
  SmallPtrSet<Access, 1> aboveAccesses;
};

/// An operation's access analysis information and graph edges.
struct OpAccesses {
  OpAccesses(Operation *op) : op(op) {}

  /// The operation.
  Operation *const op;
  /// The parent block lattice node.
  BlockAccesses *parent = nullptr;
  /// If this is a callable op, `callers` is the set of ops calling it.
  SmallPtrSet<OpAccesses *, 1> callers;
  /// The accesses performed by this op.
  SmallPtrSet<Access, 1> accesses;
};

/// An analysis that determines states read and written by operations and
/// blocks. Looks through calls and handles nested operations properly. Does not
/// follow state values returned from functions and modified by operations.
struct AccessAnalysis {
  LogicalResult analyze(Operation *op);
  OpAccesses *lookup(Operation *op);
  BlockAccesses *lookup(Block *block);

  /// A global order assigned to state values. These allow us to not care about
  /// ordering during the access analysis and only establish a determinstic
  /// order once we insert additional operations later on.
  DenseMap<Value, unsigned> stateOrder;

  /// A symbol table cache.
  SymbolTableCollection symbolTable;

private:
  llvm::SpecificBumpPtrAllocator<OpAccesses> opAlloc;
  llvm::SpecificBumpPtrAllocator<BlockAccesses> blockAlloc;

  DenseMap<Operation *, OpAccesses *> opAccesses;
  DenseMap<Block *, BlockAccesses *> blockAccesses;

  SetVector<OpAccesses *> opWorklist;
  bool anyInvalidStateAccesses = false;

  // Get the node for an operation, creating one if necessary.
  OpAccesses &get(Operation *op) {
    auto &slot = opAccesses[op];
    if (!slot)
      slot = new (opAlloc.Allocate()) OpAccesses(op);
    return *slot;
  }

  // Get the node for a block, creating one if necessary.
  BlockAccesses &get(Block *block) {
    auto &slot = blockAccesses[block];
    if (!slot)
      slot = new (blockAlloc.Allocate()) BlockAccesses(block);
    return *slot;
  }

  // NOLINTBEGIN(misc-no-recursion)
  void addOpAccess(OpAccesses &op, Access access);
  void addBlockAccess(BlockAccesses &block, Access access);
  // NOLINTEND(misc-no-recursion)
};
} // namespace

LogicalResult AccessAnalysis::analyze(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing accesses in " << op->getName() << "\n");

  // Create the lattice nodes for all blocks and operations.
  llvm::SmallSetVector<OpAccesses *, 16> initWorklist;
  initWorklist.insert(&get(op));
  while (!initWorklist.empty()) {
    OpAccesses &opNode = *initWorklist.pop_back_val();

    // First create lattice nodes for all nested blocks and operations.
    for (auto &region : opNode.op->getRegions()) {
      for (auto &block : region) {
        BlockAccesses &blockNode = get(&block);
        blockNode.parent = &opNode;
        for (auto &subOp : block) {
          if (!isOpInteresting(&subOp))
            continue;
          OpAccesses &subOpNode = get(&subOp);
          if (!subOp.hasTrait<OpTrait::IsIsolatedFromAbove>()) {
            subOpNode.parent = &blockNode;
          }
          initWorklist.insert(&subOpNode);
        }
      }
    }

    // Track the relationship between callers and callees.
    if (auto callOp = dyn_cast<CallOpInterface>(opNode.op))
      if (auto *calleeOp = callOp.resolveCallable(&symbolTable))
        get(calleeOp).callers.insert(&opNode);

    // Create the seed accesses.
    if (auto readOp = dyn_cast<StateReadOp>(opNode.op))
      addOpAccess(opNode, Access(readOp.getState(), AccessType::Read));
    else if (auto writeOp = dyn_cast<StateWriteOp>(opNode.op))
      addOpAccess(opNode, Access(writeOp.getState(), AccessType::Write));
  }
  LLVM_DEBUG(llvm::dbgs() << "- Prepared " << blockAccesses.size()
                          << " block and " << opAccesses.size()
                          << " op lattice nodes\n");
  LLVM_DEBUG(llvm::dbgs() << "- Worklist has " << opWorklist.size()
                          << " initial ops\n");

  // Propagate accesses through calls.
  while (!opWorklist.empty()) {
    if (anyInvalidStateAccesses)
      return failure();
    auto &opNode = *opWorklist.pop_back_val();
    if (opNode.callers.empty())
      continue;
    auto calleeOp = dyn_cast<CallableOpInterface>(opNode.op);
    if (!calleeOp)
      return opNode.op->emitOpError(
          "does not implement CallableOpInterface but has callers");
    LLVM_DEBUG(llvm::dbgs() << "- Updating callable " << opNode.op->getName()
                            << " " << opNode.op->getAttr("sym_name") << "\n");

    auto &calleeRegion = *calleeOp.getCallableRegion();
    auto *blockNode = lookup(&calleeRegion.front());
    if (!blockNode)
      continue;
    auto calleeArgs = blockNode->block->getArguments();

    for (auto *callOpNode : opNode.callers) {
      LLVM_DEBUG(llvm::dbgs() << "  - Updating " << *callOpNode->op << "\n");
      auto callArgs = cast<CallOpInterface>(callOpNode->op).getArgOperands();
      for (auto [calleeArg, callArg] : llvm::zip(calleeArgs, callArgs)) {
        if (blockNode->argAccesses.contains({calleeArg, AccessType::Read}))
          addOpAccess(*callOpNode, {callArg, AccessType::Read});
        if (blockNode->argAccesses.contains({calleeArg, AccessType::Write}))
          addOpAccess(*callOpNode, {callArg, AccessType::Write});
      }
    }
  }

  return failure(anyInvalidStateAccesses);
}

OpAccesses *AccessAnalysis::lookup(Operation *op) {
  return opAccesses.lookup(op);
}

BlockAccesses *AccessAnalysis::lookup(Block *block) {
  return blockAccesses.lookup(block);
}

// NOLINTBEGIN(misc-no-recursion)
void AccessAnalysis::addOpAccess(OpAccesses &op, Access access) {
  // We don't support state pointers flowing among ops and blocks. Check that
  // the accessed state is either directly passed down through a block argument
  // (no defining op), or is trivially a local state allocation.
  auto *defOp = access.getPointer().getDefiningOp();
  if (defOp && !isa<AllocStateOp, RootInputOp, RootOutputOp>(defOp)) {
    auto d = op.op->emitOpError("accesses non-trivial state value defined by `")
             << defOp->getName()
             << "`; only block arguments and `arc.alloc_state` results are "
                "supported";
    d.attachNote(defOp->getLoc()) << "state defined here";
    anyInvalidStateAccesses = true;
  }

  // HACK: Do not propagate accesses outside of `arc.passthrough` to prevent
  // reads from being legalized. Ideally we'd be able to more precisely specify
  // on read ops whether they should read the initial or the final value.
  if (isa<PassThroughOp>(op.op))
    return;

  // Propagate to the parent block and operation if the access escapes the block
  // or targets a block argument.
  if (op.accesses.insert(access).second && op.parent) {
    stateOrder.insert({access.getPointer(), stateOrder.size()});
    addBlockAccess(*op.parent, access);
  }
}

void AccessAnalysis::addBlockAccess(BlockAccesses &block, Access access) {
  Value value = access.getPointer();

  // If the accessed value is defined outside the block, add it to the set of
  // outside accesses.
  if (value.getParentBlock() != block.block) {
    if (block.aboveAccesses.insert(access).second)
      addOpAccess(*block.parent, access);
    return;
  }

  // If the accessed value is defined within the block, and it is a block
  // argument, add it to the list of block argument accesses.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    assert(blockArg.getOwner() == block.block);
    if (!block.argAccesses.insert(access).second)
      return;

    // Adding block argument accesses affects calls to the surrounding ops. Add
    // the op to the worklist such that the access can propagate to callers.
    opWorklist.insert(block.parent);
  }
}
// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// Legalization
//===----------------------------------------------------------------------===//

namespace {
struct Legalizer {
  Legalizer(AccessAnalysis &analysis) : analysis(analysis) {}
  LogicalResult run(MutableArrayRef<Region> regions);
  LogicalResult visitBlock(Block *block);

  AccessAnalysis &analysis;

  unsigned numLegalizedWrites = 0;
  unsigned numUpdatedReads = 0;

  /// A mapping from pre-existing states to temporary states for read
  /// operations, created during legalization to remove read-after-write
  /// hazards.
  DenseMap<Value, Value> legalizedStates;
};
} // namespace

LogicalResult Legalizer::run(MutableArrayRef<Region> regions) {
  for (auto &region : regions)
    for (auto &block : region)
      if (failed(visitBlock(&block)))
        return failure();
  assert(legalizedStates.empty() && "should be balanced within block");
  return success();
}

LogicalResult Legalizer::visitBlock(Block *block) {
  // In a first reverse pass over the block, find the first write that occurs
  // before the last read of a state, if any.
  SmallPtrSet<Value, 4> readStates;
  DenseMap<Value, Operation *> illegallyWrittenStates;
  for (Operation &op : llvm::reverse(*block)) {
    const auto *accesses = analysis.lookup(&op);
    if (!accesses)
      continue;

    // Determine the states written by this op for which we have already seen a
    // read earlier. These writes need to be legalized.
    SmallVector<Value, 1> affectedStates;
    for (auto access : accesses->accesses)
      if (access.getInt() == AccessType::Write)
        if (readStates.contains(access.getPointer()))
          illegallyWrittenStates[access.getPointer()] = &op;

    // Determine the states read by this op. This comes after handling of the
    // writes, such that a block that contains both reads and writes to a state
    // doesn't mark itself as illegal. Instead, we will descend into that block
    // further down and do a more fine-grained legalization.
    for (auto access : accesses->accesses)
      if (access.getInt() == AccessType::Read)
        readStates.insert(access.getPointer());
  }

  // Create a mapping from operations that create a read-after-write hazard to
  // the states that they modify. Don't consider states that have already been
  // legalized. This is important since we may have already created a temporary
  // in a parent block which we can just reuse.
  DenseMap<Operation *, SmallVector<Value, 1>> illegalWrites;
  for (auto [state, op] : illegallyWrittenStates)
    if (!legalizedStates.count(state))
      illegalWrites[op].push_back(state);

  // In a second forward pass over the block, insert the necessary temporary
  // state to legalize the writes and recur into subblocks while providing the
  // necessary rewrites.
  SmallVector<Value> locallyLegalizedStates;

  auto handleIllegalWrites =
      [&](Operation *op, SmallVector<Value, 1> &states) -> LogicalResult {
    LLVM_DEBUG(llvm::dbgs() << "Visiting illegal " << op->getName() << "\n");

    // Sort the states we need to legalize by a determinstic order established
    // during the access analysis. Without this the exact order in which states
    // were moved into a temporary would be non-deterministic.
    llvm::sort(states, [&](Value a, Value b) {
      return analysis.stateOrder.lookup(a) < analysis.stateOrder.lookup(b);
    });

    // Legalize each state individually.
    for (auto state : states) {
      LLVM_DEBUG(llvm::dbgs() << "- Legalizing " << state << "\n");

      // HACK: This is ugly, but we need a storage reference to allocate a state
      // into. Ideally we'd materialize this later on, but the current impl of
      // the alloc op requires a storage immediately. So try to find one.
      auto storage = TypeSwitch<Operation *, Value>(state.getDefiningOp())
                         .Case<AllocStateOp, RootInputOp, RootOutputOp>(
                             [&](auto allocOp) { return allocOp.getStorage(); })
                         .Default([](auto) { return Value{}; });
      if (!storage) {
        mlir::emitError(
            state.getLoc(),
            "cannot find storage pointer to allocate temporary into");
        return failure();
      }

      // Allocate a temporary state, read the current value of the state we are
      // legalizing, and write it to the temporary.
      ++numLegalizedWrites;
      ImplicitLocOpBuilder builder(state.getLoc(), op);
      auto tmpState =
          builder.create<AllocStateOp>(state.getType(), storage, nullptr);
      auto stateValue = builder.create<StateReadOp>(state);
      builder.create<StateWriteOp>(tmpState, stateValue, Value{});
      locallyLegalizedStates.push_back(state);
      legalizedStates.insert({state, tmpState});
    }
    return success();
  };

  for (Operation &op : *block) {
    if (isOpInteresting(&op)) {
      if (auto it = illegalWrites.find(&op); it != illegalWrites.end())
        if (failed(handleIllegalWrites(&op, it->second)))
          return failure();
    }
    // BUG: This is insufficient. Actually only reads should have their state
    // updated, since we want writes to still affect the original state. This
    // works for `state_read`, but in the case of a function that both reads and
    // writes a state we only have a single operand to change but we would need
    // one for reads and one for writes instead.
    // HACKY FIX: Assume that there is ever only a single write to a state. In
    // that case it is safe to assume that when an op is marked as writing a
    // state it wants the original state, not the temporary one for reads.
    const auto *accesses = analysis.lookup(&op);
    for (auto &operand : op.getOpOperands()) {
      if (accesses &&
          accesses->accesses.contains({operand.get(), AccessType::Read}) &&
          accesses->accesses.contains({operand.get(), AccessType::Write})) {
        auto d = op.emitWarning("operation reads and writes state; "
                                "legalization may be insufficient");
        d.attachNote()
            << "state update legalization does not properly handle operations "
               "that both read and write states at the same time; runtime data "
               "races between the read and write behavior are possible";
        d.attachNote(operand.get().getLoc()) << "state defined here:";
      }
      if (!accesses ||
          !accesses->accesses.contains({operand.get(), AccessType::Write})) {
        if (auto tmpState = legalizedStates.lookup(operand.get())) {
          operand.set(tmpState);
          ++numUpdatedReads;
        }
      }
    }
    for (auto &region : op.getRegions())
      for (auto &block : region)
        if (failed(visitBlock(&block)))
          return failure();
  }

  // Since we're leaving this block's scope, remove all the locally-legalized
  // states which are no longer accessible outside.
  for (auto state : locallyLegalizedStates)
    legalizedStates.erase(state);
  return success();
}

static LogicalResult getAncestorOpsInCommonDominatorBlock(
    Operation *write, Operation **writeAncestor, Operation *read,
    Operation **readAncestor, DominanceInfo *domInfo) {
  Block *commonDominator =
      domInfo->findNearestCommonDominator(write->getBlock(), read->getBlock());
  if (!commonDominator)
    return write->emitOpError(
        "cannot find a common dominator block with all read operations");

  // Path from writeOp to commmon dominator must only contain IfOps with no
  // return values
  Operation *writeParent = write;
  while (writeParent->getBlock() != commonDominator) {
    if (!isa<scf::IfOp, ClockTreeOp>(writeParent->getParentOp()))
      return write->emitOpError("memory write operations in arbitrarily nested "
                                "regions not supported");
    writeParent = writeParent->getParentOp();
  }
  Operation *readParent = read;
  while (readParent->getBlock() != commonDominator)
    readParent = readParent->getParentOp();

  *writeAncestor = writeParent;
  *readAncestor = readParent;
  return success();
}

static LogicalResult
moveMemoryWritesAfterLastRead(Region &region, const DenseSet<Value> &memories,
                              DominanceInfo *domInfo) {
  // Collect memory values and their reads
  DenseMap<Value, SetVector<Operation *>> readOps;
  auto result = region.walk([&](Operation *op) {
    if (isa<MemoryWriteOp>(op))
      return WalkResult::advance();
    SmallVector<Value> memoriesReadFrom;
    if (auto readOp = dyn_cast<MemoryReadOp>(op)) {
      memoriesReadFrom.push_back(readOp.getMemory());
    } else {
      for (auto operand : op->getOperands())
        if (isa<MemoryType>(operand.getType()))
          memoriesReadFrom.push_back(operand);
    }
    for (auto memVal : memoriesReadFrom) {
      if (!memories.contains(memVal))
        return op->emitOpError("uses memory value not directly defined by a "
                               "arc.alloc_memory operation"),
               WalkResult::interrupt();
      readOps[memVal].insert(op);
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  // Collect all writes
  SmallVector<MemoryWriteOp> writes;
  region.walk([&](MemoryWriteOp writeOp) { writes.push_back(writeOp); });

  // Move the writes
  for (auto writeOp : writes) {
    if (!memories.contains(writeOp.getMemory()))
      return writeOp->emitOpError("uses memory value not directly defined by a "
                                  "arc.alloc_memory operation");
    for (auto *readOp : readOps[writeOp.getMemory()]) {
      // (1) If the last read and the write are in the same block, just move the
      // write after the read.
      // (2) If the write is directly in the clock tree region and the last read
      // in some nested region, move the write after the operation with the
      // nested region. (3) If the write is nested in if-statements (arbitrarily
      // deep) without return value, move the whole if operation after the last
      // read or the operation that defines the region if the read is inside a
      // nested region. (4) Number (3) may move more memory operations with the
      // write op, thus messing up the order of previously moved memory writes,
      // we check in a second walk-through if that is the case and just emit an
      // error for now. We could instead move reads in a parent region, split if
      // operations such that the memory write has its own, etc. Alternatively,
      // rewrite this to insert temporaries which is more difficult for memories
      // than simple states because the memory addresses have to be considered
      // (we cannot just copy the whole memory each time).
      Operation *readAncestor, *writeAncestor;
      if (failed(getAncestorOpsInCommonDominatorBlock(
              writeOp, &writeAncestor, readOp, &readAncestor, domInfo)))
        return failure();
      // FIXME: the 'isBeforeInBlock` + 'moveAfter' compination can be
      // computationally very expensive.
      if (writeAncestor->isBeforeInBlock(readAncestor))
        writeAncestor->moveAfter(readAncestor);
    }
  }

  // Double check that all writes happen after all reads to the same memory.
  for (auto writeOp : writes) {
    for (auto *readOp : readOps[writeOp.getMemory()]) {
      Operation *readAncestor, *writeAncestor;
      if (failed(getAncestorOpsInCommonDominatorBlock(
              writeOp, &writeAncestor, readOp, &readAncestor, domInfo)))
        return failure();

      if (writeAncestor->isBeforeInBlock(readAncestor))
        return writeOp
                   ->emitOpError("could not be moved to be after all reads to "
                                 "the same memory")
                   .attachNote(readOp->getLoc())
               << "could not be moved after this read";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LegalizeStateUpdatePass
    : public arc::impl::LegalizeStateUpdateBase<LegalizeStateUpdatePass> {
  LegalizeStateUpdatePass() = default;
  LegalizeStateUpdatePass(const LegalizeStateUpdatePass &pass)
      : LegalizeStateUpdatePass() {}

  void runOnOperation() override;

  Statistic numLegalizedWrites{
      this, "legalized-writes",
      "Writes that required temporary state for later reads"};
  Statistic numUpdatedReads{this, "updated-reads", "Reads that were updated"};
};
} // namespace

void LegalizeStateUpdatePass::runOnOperation() {
  auto module = getOperation();
  auto *domInfo = &getAnalysis<DominanceInfo>();

  for (auto model : module.getOps<ModelOp>()) {
    DenseSet<Value> memories;
    for (auto memOp : model.getOps<AllocMemoryOp>())
      memories.insert(memOp.getResult());
    for (auto ct : model.getOps<ClockTreeOp>())
      if (failed(
              moveMemoryWritesAfterLastRead(ct.getBody(), memories, domInfo)))
        return signalPassFailure();
  }

  AccessAnalysis analysis;
  if (failed(analysis.analyze(module)))
    return signalPassFailure();

  Legalizer legalizer(analysis);
  if (failed(legalizer.run(module->getRegions())))
    return signalPassFailure();
  numLegalizedWrites += legalizer.numLegalizedWrites;
  numUpdatedReads += legalizer.numUpdatedReads;
}

std::unique_ptr<Pass> arc::createLegalizeStateUpdatePass() {
  return std::make_unique<LegalizeStateUpdatePass>();
}
