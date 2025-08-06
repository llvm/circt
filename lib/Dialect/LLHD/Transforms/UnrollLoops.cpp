//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
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
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
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

/// Check that the loop matches the specific pattern we understand, and extract
/// the loop condition and induction variable.
bool Loop::match() {
  // Ensure that there is a unique exit point and condition for the loop.
  SmallVector<BlockOperand *> exits;
  for (auto *block : cfgLoop.getBlocks())
    for (auto &edge : block->getTerminator()->getBlockOperands())
      if (!cfgLoop.contains(edge.get()))
        exits.push_back(&edge);
  if (exits.size() != 1)
    return failMatch("multiple exits");
  exitEdge = exits.back();

  // The terminator doing the exit must be a conditional branch.
  auto exitBranch = dyn_cast<cf::CondBranchOp>(exitEdge->getOwner());
  if (!exitBranch)
    return failMatch("unsupported exit branch");
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
  if (!indVarArg || indVarArg.getOwner() != header)
    return failMatch("induction variable is not a header block argument");
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

  // Pattern match the increment operation on the induction variable.
  if (auto addOp = indVarNext.getDefiningOp<comb::AddOp>();
      addOp && addOp.getNumOperands() == 2) {
    if (addOp.getOperand(0) != indVarArg)
      return failMatch("increment LHS not the induction variable");
    IntegerAttr incAttr;
    if (!matchPattern(addOp.getOperand(1), m_Constant(&incAttr)))
      return failMatch("increment RHS non-constant");
    indVarIncrement = incAttr.getValue();
  } else {
    return failMatch("unsupported increment");
  }

  // Determine the trip count and loop behavior. We're very picky for now.
  if (predicate == comb::ICmpPredicate::slt && indVarIncrement == 1 &&
      beginBound == 0 && !endBound.isNegative() && endBound.slt(1024)) {
    tripCount = endBound.getZExtValue();
  } else {
    return failMatch("unsupported loop bounds");
  }

  return true;
}

/// Unroll the loop by cloning its body blocks and replacing the induction
/// variable with constant iteration indices.
void Loop::unroll(CFGLoopInfo &cfgLoopInfo) {
  assert(beginBound == 0 && !endBound.isNegative() && indVarIncrement == 1);
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
  SmallPtrSet<Block *, 8> blocksToPrune;
  for (auto *block : cfgLoop.getBlocks())
    if (block->use_empty())
      blocksToPrune.insert(block);
  while (!blocksToPrune.empty()) {
    auto *block = *blocksToPrune.begin();
    blocksToPrune.erase(block);
    if (!block->use_empty())
      continue;
    for (auto *succ : block->getSuccessors())
      if (cfgLoop.contains(succ))
        blocksToPrune.insert(succ);
    block->dropAllDefinedValueUses();
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
};
} // namespace

void UnrollLoopsPass::runOnOperation() {
  for (auto op : getOperation().getOps<CombinationalOp>())
    runOnOperation(op);
}

void UnrollLoopsPass::runOnOperation(CombinationalOp op) {
  // There's nothing to do if we only have a single block. MLIR even refuses to
  // compute a dominator tree in that case.
  if (op.getBody().hasOneBlock())
    return;

  // Find the loops.
  LLVM_DEBUG(llvm::dbgs() << "Unrolling loops in " << op.getLoc() << "\n");
  DominanceInfo domInfo(op);
  CFGLoopInfo cfgLoopInfo(domInfo.getDomTree(&op.getBody()));

  // We only support simple loops where there is a single back-edge to the
  // header, and the latch block has a back-edge to a single header. Create a
  // data structure for each loop we can potentially unroll. The loops are in
  // preorder, with outer loops appearing before their child loops.
  SmallVector<Loop> loops;
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
    if (loop.match())
      loops.push_back(std::move(loop));
  }

  if (loops.empty())
    return;

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
}
