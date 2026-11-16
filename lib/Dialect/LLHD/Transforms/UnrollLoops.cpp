//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-unroll-loops"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_UNROLLLOOPSPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Clone a list of blocks into a region before the given block.
///
/// See `Region::cloneInto` for the original code that clones an entire region.
static void cloneBlocks(ArrayRef<Block *> blocks, Region &region,
                        Region::iterator before, IRMapping &mapper) {
  // If the list is empty there is nothing to clone.
  if (blocks.empty())
    return;

  // First clone all the blocks and block arguments and map them, but don't yet
  // clone the operations, as they may otherwise add a use to a block that has
  // not yet been mapped
  SmallVector<Block *> newBlocks;
  newBlocks.reserve(blocks.size());
  for (auto *block : blocks) {
    auto *newBlock = new Block();
    mapper.map(block, newBlock);
    for (auto arg : block->getArguments())
      mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));
    region.getBlocks().insert(before, newBlock);
    newBlocks.push_back(newBlock);
  }

  // Now follow up with creating the operations, but don't yet clone their
  // regions, nor set their operands. Setting the successors is safe as all have
  // already been mapped. We are essentially just creating the operation results
  // to be able to map them. Cloning the operands and region as well would lead
  // to uses of operations not yet mapped.
  auto cloneOptions =
      Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  for (auto [oldBlock, newBlock] : llvm::zip(blocks, newBlocks))
    for (auto &op : *oldBlock)
      newBlock->push_back(op.clone(mapper, cloneOptions));

  // Finally now that all operation results have been mapped, set the operands
  // and clone the regions.
  SmallVector<Value> operands;
  for (auto [oldBlock, newBlock] : llvm::zip(blocks, newBlocks)) {
    for (auto [oldOp, newOp] : llvm::zip(*oldBlock, *newBlock)) {
      operands.resize(oldOp.getNumOperands());
      llvm::transform(
          oldOp.getOperands(), operands.begin(),
          [&](Value operand) { return mapper.lookupOrDefault(operand); });
      newOp.setOperands(operands);
      for (auto [oldRegion, newRegion] :
           llvm::zip(oldOp.getRegions(), newOp.getRegions()))
        oldRegion.cloneInto(&newRegion, mapper);
    }
  }
}

//===----------------------------------------------------------------------===//
// Loop Unroller
//===----------------------------------------------------------------------===//

namespace {
/// A data structure tracking information on a single loop.
struct Loop {
  Loop(unsigned loopId, CFGLoop &cfgLoop) : loopId(loopId), cfgLoop(cfgLoop) {}
  bool failMatch(const Twine &msg) const;
  bool match();
  bool matchExit(BlockOperand *edge);
  bool routeLiveOutsThroughArguments();
  void unroll(CFGLoopInfo &cfgLoopInfo);

  /// A numeric identifier for debugging purposes.
  unsigned loopId;
  /// Loop analysis information about this specific loop.
  CFGLoop &cfgLoop;
  /// The CFG edge exiting the loop.
  BlockOperand *exitEdge = nullptr;
  /// The SSA value holding the exit condition.
  Value exitCondition;
  /// Whether the exit condition is inverted, i.e. the contination condition.
  bool exitInverted;
  /// The induction variable.
  Value indVar;
  /// The updated induction variable passed into the next loop iteration.
  Value indVarNext;
  /// Whether the exit condition tests the updated induction variable (a
  /// `do ... while` loop) instead of the header block argument.
  bool comparesNext = false;
  /// The continuation predicate. The loop continues until the induction
  /// variable compared against the end bound no longer matches this predicate.
  comb::ICmpPredicate predicate;
  /// The induction variable increment.
  APInt indVarIncrement;
  /// The initial value for the induction variable.
  APInt beginBound;
  /// The final value for the induction variable.
  APInt endBound;
  /// The number of iterations of the loop.
  unsigned tripCount = 0;
};
} // namespace

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Loop &loop) {
  os << "#" << loop.loopId << " from ";
  loop.cfgLoop.getHeader()->printAsOperand(os);
  os << " to ";
  loop.cfgLoop.getLoopLatch()->printAsOperand(os);
  return os;
}

/// Helper to print a debug message on match failure and return false.
bool Loop::failMatch(const Twine &msg) const {
  LLVM_DEBUG(llvm::dbgs() << "- Ignoring loop " << *this << ": " << msg
                          << "\n");
  return false;
}

/// Evaluate a comparison predicate on two constants. Returns `std::nullopt` for
/// the four-valued predicates, which never control a loop we unroll.
static std::optional<bool> evaluatePredicate(comb::ICmpPredicate predicate,
                                             const APInt &lhs,
                                             const APInt &rhs) {
  switch (predicate) {
  case comb::ICmpPredicate::eq:
    return lhs == rhs;
  case comb::ICmpPredicate::ne:
    return lhs != rhs;
  case comb::ICmpPredicate::slt:
    return lhs.slt(rhs);
  case comb::ICmpPredicate::sle:
    return lhs.sle(rhs);
  case comb::ICmpPredicate::sgt:
    return lhs.sgt(rhs);
  case comb::ICmpPredicate::sge:
    return lhs.sge(rhs);
  case comb::ICmpPredicate::ult:
    return lhs.ult(rhs);
  case comb::ICmpPredicate::ule:
    return lhs.ule(rhs);
  case comb::ICmpPredicate::ugt:
    return lhs.ugt(rhs);
  case comb::ICmpPredicate::uge:
    return lhs.uge(rhs);
  default:
    return std::nullopt;
  }
}

/// The largest number of iterations the unroller copies a loop body for.
static constexpr unsigned kMaxTripCount = 1024;

/// Check that the loop matches the specific pattern we understand, and extract
/// the loop condition and induction variable.
///
/// The loop may have more than one exit (a `break`, or a `return` from a
/// function inlined into the process): exactly one of them must be the
/// counting exit that compares the induction variable against a constant, and
/// the others stay in every unrolled copy of the body as ordinary branches out
/// of the loop. Such early exits make the exit blocks reachable from every
/// copy, so the loop's values must leave it only through branch operands
/// (see routeLiveOutsThroughArguments).
bool Loop::match() {
  SmallVector<BlockOperand *> exits;
  for (auto *block : cfgLoop.getBlocks())
    for (auto &edge : block->getTerminator()->getBlockOperands())
      if (!cfgLoop.contains(edge.get()))
        exits.push_back(&edge);
  if (exits.empty())
    return failMatch("no exit");

  // A single exit may be anywhere in the loop. With several, the counting exit
  // must be in the header, which every iteration runs: that is where a `for`
  // loop tests its bound, and a `break` leaves through a later block. An exit
  // test that an iteration can skip would make the trip count wrong.
  if (exits.size() == 1)
    return matchExit(exits.front());
  auto *header = cfgLoop.getHeader();
  bool found = false;
  for (auto *edge : exits)
    if (edge->getOwner()->getBlock() == header && matchExit(edge)) {
      found = true;
      break;
    }
  if (!found)
    return failMatch("no counting exit in the header of a loop with several "
                     "exits");
  if (!routeLiveOutsThroughArguments())
    return failMatch("a value of the loop is used after it where it cannot be "
                     "passed as a block argument");
  return true;
}

/// Make every use of a loop value outside the loop go through a block
/// argument. Unrolling a loop with several exits clones the exiting blocks, so
/// a block after the loop gets one predecessor per copy and a loop value it
/// uses directly would no longer dominate it. A `break` block (outside the
/// loop, entered from one loop block) commonly reads the induction variable;
/// such a block, and any chain of single-predecessor blocks after it, gets the
/// value as a new argument from its predecessor instead; so does a block whose
/// predecessors are all in the loop (where a `break` from an inner loop joins
/// its normal exit). A use in a block also entered from outside the loop
/// cannot be rewritten this way and fails the match. The rewrite does not
/// change what the region computes, so a failure part-way leaves valid,
/// equivalent IR.
bool Loop::routeLiveOutsThroughArguments() {
  Region *region = cfgLoop.getHeader()->getParent();

  // A header argument that the loop passes back unchanged and that enters with
  // a single value is that value. Replacing it keeps a use after the loop (a
  // variable the loop does not write, read where the exits merge) from
  // counting as a loop value.
  auto *header = cfgLoop.getHeader();
  for (auto arg : header->getArguments()) {
    Value incoming;
    bool invariant = true;
    for (auto &pred : header->getUses()) {
      auto branch = dyn_cast<BranchOpInterface>(pred.getOwner());
      if (!branch) {
        invariant = false;
        break;
      }
      Value value = branch.getSuccessorOperands(
          pred.getOperandNumber())[arg.getArgNumber()];
      if (value == arg)
        continue;
      if (!value || (incoming && incoming != value)) {
        invariant = false;
        break;
      }
      incoming = value;
    }
    if (invariant && incoming)
      arg.replaceAllUsesWith(incoming);
  }

  SmallVector<Value> worklist;
  for (auto *block : cfgLoop.getBlocks()) {
    for (auto arg : block->getArguments())
      worklist.push_back(arg);
    block->walk([&](Operation *op) {
      for (auto result : op->getResults())
        worklist.push_back(result);
    });
  }
  unsigned budget = 100000;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      auto *block =
          region->findAncestorBlockInRegion(*use.getOwner()->getBlock());
      if (!block)
        return false;
      if (cfgLoop.contains(block))
        continue;
      // The block must be entered by a single edge, or only from inside the
      // loop: the value then dominates every predecessor (it dominated the
      // block), so each edge can pass it.
      if (block->use_empty() || --budget == 0)
        return false;
      SmallVector<std::pair<BranchOpInterface, unsigned>> edges;
      for (auto &edge : block->getUses()) {
        auto branch = dyn_cast<BranchOpInterface>(edge.getOwner());
        auto *pred = edge.getOwner()->getBlock();
        if (!branch || pred == block ||
            (!block->hasOneUse() && !cfgLoop.contains(pred)))
          return false;
        edges.push_back({branch, edge.getOperandNumber()});
      }
      auto arg = block->addArgument(value.getType(), value.getLoc());
      value.replaceUsesWithIf(arg, [&](OpOperand &other) {
        return region->findAncestorBlockInRegion(
                   *other.getOwner()->getBlock()) == block;
      });
      for (auto [branch, index] : edges)
        branch.getSuccessorOperands(index).append(value);
      // The value is now used by the predecessor's branch; if that block is
      // outside the loop too, the next round moves the use up once more.
      worklist.push_back(value);
      break;
    }
  }
  return true;
}

/// Check whether `edge` is a counting exit: a conditional branch on a
/// comparison of a header block argument against a constant, where the argument
/// starts at a constant and advances by a constant step. Compute the trip count
/// by stepping the induction variable from its initial value.
bool Loop::matchExit(BlockOperand *edge) {
  exitEdge = edge;
  indVarNext = {};
  comparesNext = false;

  // The terminator doing the exit must be a conditional branch whose other
  // successor stays in the loop.
  auto exitBranch = dyn_cast<cf::CondBranchOp>(exitEdge->getOwner());
  if (!exitBranch)
    return failMatch("unsupported exit branch");
  if (!cfgLoop.contains(
          exitBranch->getSuccessor(1 - exitEdge->getOperandNumber())))
    return failMatch("both successors of the exit branch leave the loop");
  exitCondition = exitBranch.getCondition();
  exitInverted = exitEdge->getOperandNumber() == 1;

  // Determine one of the loop bounds and the induction variable based on the
  // exit condition.
  if (auto icmpOp = exitCondition.getDefiningOp<comb::ICmpOp>()) {
    IntegerAttr boundAttr;
    if (!matchPattern(icmpOp.getRhs(), m_Constant(&boundAttr)))
      return failMatch("non-constant loop bound");
    indVar = icmpOp.getLhs();
    predicate = icmpOp.getPredicate();
    endBound = boundAttr.getValue();
  } else {
    return failMatch("unsupported exit condition");
  }

  // If the exit condition is not inverted, the predicate is the exit predicate.
  // Negate it such that we have a continuation predicate.
  if (!exitInverted)
    predicate = comb::ICmpOp::getNegatedPredicate(predicate);

  // Determine the initial and next value of the induction variable.
  auto *header = cfgLoop.getHeader();
  auto *latch = cfgLoop.getLoopLatch();
  auto indVarArg = dyn_cast<BlockArgument>(indVar);
  if (!indVarArg) {
    // A `do ... while` loop tests the stepped value, `i + step < bound`; the
    // add is checked against the back-edge value below.
    if (auto addOp = indVar.getDefiningOp<comb::AddOp>();
        addOp && addOp.getNumOperands() == 2) {
      indVarArg = dyn_cast<BlockArgument>(addOp.getOperand(0));
      comparesNext = true;
    }
  }
  if (!indVarArg || indVarArg.getOwner() != header)
    return failMatch("induction variable is not a header block argument");
  Value compared = indVar;
  indVar = indVarArg;
  IntegerAttr beginBoundAttr;
  for (auto &pred : header->getUses()) {
    auto branchOp = dyn_cast<BranchOpInterface>(pred.getOwner());
    if (!branchOp)
      return failMatch("header predecessor terminator is not a branch op");
    auto indVarValue = branchOp.getSuccessorOperands(
        pred.getOperandNumber())[indVarArg.getArgNumber()];
    IntegerAttr boundAttr;
    if (pred.getOwner()->getBlock() == latch) {
      indVarNext = indVarValue;
    } else if (matchPattern(indVarValue, m_Constant(&boundAttr))) {
      if (!beginBoundAttr)
        beginBoundAttr = boundAttr;
      else if (boundAttr != beginBoundAttr)
        return failMatch("multiple initial bounds");
    } else {
      return failMatch("unsupported induction variable value");
    }
  }
  if (!beginBoundAttr)
    return failMatch("no initial bound");
  beginBound = beginBoundAttr.getValue();

  // Pattern match the increment operation on the induction variable: an add
  // of a constant (a down-counting loop adds -1), or a subtract of one.
  if (!indVarNext)
    return failMatch("no back-edge value for the induction variable");
  IntegerAttr incAttr;
  if (auto addOp = indVarNext.getDefiningOp<comb::AddOp>();
      addOp && addOp.getNumOperands() == 2) {
    if (addOp.getOperand(0) != indVarArg)
      return failMatch("increment LHS not the induction variable");
    if (!matchPattern(addOp.getOperand(1), m_Constant(&incAttr)))
      return failMatch("increment RHS non-constant");
    indVarIncrement = incAttr.getValue();
  } else if (auto subOp = indVarNext.getDefiningOp<comb::SubOp>()) {
    if (subOp.getLhs() != indVarArg)
      return failMatch("decrement LHS not the induction variable");
    if (!matchPattern(subOp.getRhs(), m_Constant(&incAttr)))
      return failMatch("decrement RHS non-constant");
    indVarIncrement = -incAttr.getValue();
  } else {
    return failMatch("unsupported increment");
  }
  if (indVarIncrement.isZero())
    return failMatch("zero increment");
  if (comparesNext && compared != indVarNext)
    return failMatch("exit condition tests a value other than the next "
                     "induction variable");

  // Determine the trip count by stepping the induction variable from its
  // initial value until the continuation predicate fails (tested on the
  // stepped value for a `do ... while`). This covers up- and down-counting
  // loops with any constant start, step and comparison, also across the
  // signed and unsigned wrap-around points. A loop that runs longer than
  // kMaxTripCount iterations, or never ends, is left alone.
  if (beginBound.getBitWidth() != endBound.getBitWidth() ||
      beginBound.getBitWidth() != indVarIncrement.getBitWidth())
    return failMatch("mismatched induction variable widths");
  APInt value = beginBound;
  unsigned trips = 0;
  while (true) {
    auto proceed = evaluatePredicate(
        predicate, comparesNext ? value + indVarIncrement : value, endBound);
    if (!proceed)
      return failMatch("unsupported loop predicate");
    if (!*proceed)
      break;
    if (++trips > kMaxTripCount)
      return failMatch("unsupported loop bounds");
    value += indVarIncrement;
  }
  tripCount = trips;
  return true;
}

/// Unroll the loop by cloning its body blocks and replacing the induction
/// variable with constant iteration indices.
void Loop::unroll(CFGLoopInfo &cfgLoopInfo) {
  LLVM_DEBUG(llvm::dbgs() << "- Unrolling loop " << *this << "\n");
  UnusedOpPruner pruner;

  // Sort the blocks in the body. This is not strictly necessary, but makes the
  // pass a lot easier to reason about in tests.
  auto *header = cfgLoop.getHeader();
  SmallVector<Block *> orderedBody;
  for (auto &block : *header->getParent())
    if (cfgLoop.contains(&block))
      orderedBody.push_back(&block);

  // Copy the loop body for every iteration of the loop.
  auto *latch = cfgLoop.getLoopLatch();
  OpBuilder builder(indVar.getContext());
  auto indValue = beginBound;
  for (unsigned trip = 0; trip < tripCount; ++trip) {
    // Clone the loop body.
    IRMapping mapper;
    cloneBlocks(orderedBody, *header->getParent(), header->getIterator(),
                mapper);
    auto *clonedHeader = mapper.lookup(header);
    auto *clonedTail = mapper.lookup(latch);

    // Replace the induction variable with the concrete value.
    auto iterIndVar = mapper.lookup(indVar);
    pruner.eraseLaterIfUnused(iterIndVar);
    builder.setInsertionPointAfterValue(iterIndVar);
    iterIndVar.replaceAllUsesWith(
        hw::ConstantOp::create(builder, iterIndVar.getLoc(), indValue));

    // Update all edges to the original loop header to point to the cloned loop
    // header. Leave the original back-edge untouched.
    for (auto &blockOperand : llvm::make_early_inc_range(header->getUses()))
      if (blockOperand.getOwner()->getBlock() != latch)
        blockOperand.set(clonedHeader);

    // Update the back-edge in the cloned latch to point to the original loop
    // header, i.e. the next iteration, instead of the cloned loop header.
    for (auto &blockOperand : clonedTail->getTerminator()->getBlockOperands())
      if (blockOperand.get() == clonedHeader)
        blockOperand.set(header);

    // Remove the exit edge in the cloned body, since we statically know that
    // the loop will continue.
    auto exitBranchOp =
        cast<cf::CondBranchOp>(mapper.lookup(exitEdge->getOwner()));
    Block *continueDest = exitBranchOp.getTrueDest();
    ValueRange continueDestOperands = exitBranchOp.getTrueDestOperands();
    if (exitEdge->getOperandNumber() == 0) {
      continueDest = exitBranchOp.getFalseDest();
      continueDestOperands = exitBranchOp.getFalseDestOperands();
    }
    builder.setInsertionPoint(exitBranchOp);
    cf::BranchOp::create(builder, exitBranchOp.getLoc(), continueDest,
                         continueDestOperands);
    pruner.eraseLaterIfUnused(exitBranchOp.getOperands());
    exitBranchOp.erase();

    // Add the new blocks to the loop body.
    for (auto *block : orderedBody) {
      auto *newBlock = mapper.lookup(block);
      cfgLoop.addBasicBlockToLoop(newBlock, cfgLoopInfo);
    }

    // Increment the induction variable value.
    indValue += indVarIncrement;
  }

  // Now that the loop body has been cloned once for each trip throughout the
  // loop, we can clean up the final iteration by always breaking out of the
  // loop. Start by replacing the induction variable with the final value.
  pruner.eraseLaterIfUnused(indVar);
  builder.setInsertionPointAfterValue(indVar);
  indVar.replaceAllUsesWith(
      hw::ConstantOp::create(builder, indVar.getLoc(), indValue));
  indVar = {};

  // Remove the continue edge of the exit branch in the loop body, since we
  // statically know that the loop will exit.
  auto exitBranchOp = cast<cf::CondBranchOp>(exitEdge->getOwner());
  Block *exitDest = exitBranchOp.getTrueDest();
  ValueRange exitDestOperands = exitBranchOp.getTrueDestOperands();
  if (exitEdge->getOperandNumber() == 1) {
    exitDest = exitBranchOp.getFalseDest();
    exitDestOperands = exitBranchOp.getFalseDestOperands();
  }
  builder.setInsertionPoint(exitBranchOp);
  cf::BranchOp::create(builder, exitBranchOp.getLoc(), exitDest,
                       exitDestOperands);
  pruner.eraseLaterIfUnused(exitBranchOp.getOperands());
  exitBranchOp.erase();
  exitEdge = nullptr;

  // Prune any body blocks that have become unreachable.
  auto &region = *header->getParent();
  SmallPtrSet<Block *, 8> reachable;
  SmallVector<Block *> worklist;
  reachable.insert(&region.front());
  worklist.push_back(&region.front());
  while (!worklist.empty())
    for (auto *succ : worklist.pop_back_val()->getSuccessors())
      if (reachable.insert(succ).second)
        worklist.push_back(succ);

  SmallVector<Block *> deadBlocks;
  for (auto &block : region) {
    if (reachable.contains(&block))
      continue;
    block.dropAllDefinedValueUses();
    deadBlocks.push_back(&block);
  }

  for (auto *block : deadBlocks) {
    cfgLoopInfo.removeBlock(block);
    block->erase();
  }

  // Remove any unused operations and block arguments.
  pruner.eraseNow();

  // Collapse trivial branches to avoid carrying a lot of useless blocks around
  // especially when unrolling nested loops.
  for (auto &block : *header->getParent()) {
    if (!cfgLoop.contains(&block))
      continue;
    while (true) {
      auto branchOp = dyn_cast<cf::BranchOp>(block.getTerminator());
      if (!branchOp)
        break;
      auto *otherBlock = branchOp.getDest();
      if (!cfgLoop.contains(otherBlock) || !otherBlock->getSinglePredecessor())
        break;
      for (auto [blockArg, branchArg] :
           llvm::zip(otherBlock->getArguments(), branchOp.getDestOperands()))
        blockArg.replaceAllUsesWith(branchArg);
      block.getOperations().splice(branchOp->getIterator(),
                                   otherBlock->getOperations());
      branchOp.erase();
      cfgLoopInfo.removeBlock(otherBlock);
      otherBlock->erase();
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct UnrollLoopsPass
    : public llhd::impl::UnrollLoopsPassBase<UnrollLoopsPass> {
  void runOnOperation() override;
  void runOnOperation(CombinationalOp op);
  bool unrollLoops(CombinationalOp op);
};
} // namespace

void UnrollLoopsPass::runOnOperation() {
  for (auto op : getOperation().getOps<CombinationalOp>())
    runOnOperation(op);
}

void UnrollLoopsPass::runOnOperation(CombinationalOp op) {
  // Unrolling a loop may open up opportunities to unroll its nested loops.
  // Iterate until all possible loops are unrolled or the limit is reached.
  for (unsigned i = 0; i < 16; ++i)
    if (!unrollLoops(op))
      break;
}

bool UnrollLoopsPass::unrollLoops(CombinationalOp op) {
  // There's nothing to do if we only have a single block. MLIR even refuses to
  // compute a dominator tree in that case.
  if (op.getBody().hasOneBlock())
    return false;

  // Find the loops.
  LLVM_DEBUG(llvm::dbgs() << "Unrolling loops in " << op.getLoc() << "\n");
  DominanceInfo domInfo(op);
  CFGLoopInfo cfgLoopInfo(domInfo.getDomTree(&op.getBody()));

  // We only support simple loops where there is a single back-edge to the
  // header, and the latch block has a back-edge to a single header. Create a
  // data structure for each loop we can potentially unroll. The loops are in
  // preorder, with outer loops appearing before their child loops.
  SmallVector<Loop> loops;
  bool retry = false;
  for (auto *cfgLoop : cfgLoopInfo.getLoopsInPreorder()) {
    // To simplify unrolling we need a unique latch block branching back to the
    // header.
    auto *header = cfgLoop->getHeader();
    auto *latch = cfgLoop->getLoopLatch();
    if (!latch)
      continue;

    LLVM_DEBUG({
      llvm::dbgs() << "- ";
      cfgLoop->print(llvm::dbgs(), false, false);
      llvm::dbgs() << "\n";
    });
    Loop loop(loops.size(), *cfgLoop);

    // Ensure that the header block is only a header for the current loop. This
    // simplifies unrolling.
    auto *parent = cfgLoop->getParentLoop();
    while (parent && parent->getHeader() != header)
      parent = parent->getParentLoop();
    if (parent) {
      loop.failMatch("header block shared across multiple loops");
      continue;
    }

    // Ensure that the latch block is only a latch for the current loop. This
    // simplifies unrolling.
    parent = cfgLoop->getParentLoop();
    while (parent && !parent->isLoopLatch(latch))
      parent = parent->getParentLoop();
    if (parent) {
      loop.failMatch("latch block shared across multiple loops");
      continue;
    }

    // Check if the loop body matches the pattern we can unroll.
    if (loop.match()) {
      loops.push_back(std::move(loop));
      continue;
    }

    // If an enclosing loop is unrolled, we retry the unrolling.
    retry |= llvm::any_of(loops, [&](const Loop &other) {
      return other.cfgLoop.contains(cfgLoop);
    });
  }

  if (loops.empty())
    return false;

  // Dump some debugging information about the loops we've found.
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    for (auto &loop : loops) {
      os << "- Loop " << loop << ":\n";
      os << "  - ";
      loop.cfgLoop.print(os, false, false);
      os << "\n";
      os << "  - Exit: ";
      loop.exitEdge->get()->printAsOperand(os);
      os << " if ";
      if (loop.exitInverted)
        os << "not ";
      os << loop.exitCondition;
      os << "\n";
      os << "  - Induction variable: ";
      loop.indVar.printAsOperand(os, OpPrintingFlags().useLocalScope());
      os << ", from " << loop.beginBound << ", while " << loop.predicate << " "
         << loop.endBound << ", increment " << loop.indVarIncrement << "\n";
      os << "  - Trip count: " << loop.tripCount << "\n";
    }
  });

  // Unroll the loops. Handling the loops in reverse unrolls inner loops before
  // their parent loops.
  for (auto &loop : llvm::reverse(loops))
    loop.unroll(cfgLoopInfo);

  return retry;
}
