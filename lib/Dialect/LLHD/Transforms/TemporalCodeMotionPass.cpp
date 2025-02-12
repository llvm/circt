//===- TemporalCodeMotionPass.cpp - Implement Temporal Code Motion Pass ---===//
//
// Implement Pass to move all signal drives in a unique exiting block per
// temporal region and coalesce drives to the same signal.
//
//===----------------------------------------------------------------------===//

#include "TemporalRegions.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_TEMPORALCODEMOTION
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace circt;
using namespace mlir;

/// Explore all paths from the 'driveBlock' to the 'dominator' block and
/// construct a boolean expression at the current insertion point of 'builder'
/// to represent all those paths.
static Value
getBranchDecisionsFromDominatorToTarget(OpBuilder &builder, Block *driveBlock,
                                        Block *dominator,
                                        DenseMap<Block *, Value> &mem) {
  Location loc = driveBlock->getTerminator()->getLoc();
  if (mem.count(driveBlock))
    return mem[driveBlock];

  SmallVector<Block *> worklist;
  worklist.push_back(driveBlock);

  while (!worklist.empty()) {
    Block *curr = worklist.back();

    if (curr == dominator || curr->getPredecessors().empty()) {
      if (!mem.count(curr))
        mem[curr] = builder.create<hw::ConstantOp>(loc, APInt(1, 1));

      worklist.pop_back();
      continue;
    }

    bool addedSomething = false;
    for (auto *predBlock : curr->getPredecessors()) {
      if (!mem.count(predBlock)) {
        worklist.push_back(predBlock);
        addedSomething = true;
      }
    }

    if (addedSomething)
      continue;

    Value runner = builder.create<hw::ConstantOp>(loc, APInt(1, 0));
    for (auto *predBlock : curr->getPredecessors()) {
      if (predBlock->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = cast<cf::CondBranchOp>(predBlock->getTerminator());
        Value cond = condBr.getCondition();
        if (condBr.getFalseDest() == curr) {
          Value trueVal = builder.create<hw::ConstantOp>(loc, APInt(1, 1));
          cond = builder.create<comb::XorOp>(loc, cond, trueVal);
        }
        Value next = builder.create<comb::AndOp>(loc, mem[predBlock], cond);
        runner = builder.create<comb::OrOp>(loc, runner, next);
      } else {
        runner = builder.create<comb::OrOp>(loc, runner, mem[predBlock]);
      }
    }
    mem[curr] = runner;
    worklist.pop_back();
  }

  return mem[driveBlock];
}

/// More a 'llhd.drv' operation before the 'moveBefore' operation by adjusting
/// the 'enable' operand.
static void moveDriveOpBefore(llhd::DrvOp drvOp, Block *dominator,
                              Operation *moveBefore,
                              DenseMap<Block *, Value> &mem) {
  OpBuilder builder(drvOp);
  builder.setInsertionPoint(moveBefore);
  Block *drvParentBlock = drvOp->getBlock();

  // Find sequence of branch decisions and add them as a sequence of
  // instructions to the TR exiting block
  Value finalValue = getBranchDecisionsFromDominatorToTarget(
      builder, drvParentBlock, dominator, mem);

  if (drvOp.getEnable())
    finalValue = builder.create<comb::AndOp>(drvOp.getLoc(), drvOp.getEnable(),
                                             finalValue);

  drvOp.getEnableMutable().assign(finalValue);
  drvOp->moveBefore(moveBefore);
}

namespace {
struct TemporalCodeMotionPass
    : public llhd::impl::TemporalCodeMotionBase<TemporalCodeMotionPass> {
  void runOnOperation() override;
  LogicalResult runOnProcess(llhd::ProcessOp procOp);
};
} // namespace

void TemporalCodeMotionPass::runOnOperation() {
  for (auto proc : getOperation().getOps<llhd::ProcessOp>())
    (void)runOnProcess(proc); // Ignore processes that could not be lowered
}

static LogicalResult checkForCFGLoop(llhd::ProcessOp procOp) {
  SmallVector<Block *> toCheck(
      llvm::map_range(procOp.getOps<llhd::WaitOp>(), [](llhd::WaitOp waitOp) {
        return waitOp.getSuccessor();
      }));
  toCheck.push_back(&procOp.getBody().front());

  SmallVector<Block *> cfStack;
  DenseSet<Block *> visited;
  for (auto *block : toCheck) {
    cfStack.clear();
    visited.clear();
    cfStack.push_back(block);

    while (!cfStack.empty()) {
      Block *curr = cfStack.back();
      if (isa<llhd::WaitOp>(curr->getTerminator())) {
        visited.insert(curr);
        cfStack.pop_back();
        continue;
      }

      bool allSuccessorsVisited = true;
      for (auto *succ : curr->getSuccessors()) {
        if (llvm::is_contained(cfStack, succ))
          return failure();

        if (!visited.contains(succ)) {
          cfStack.push_back(succ);
          allSuccessorsVisited = false;
          break;
        }
      }

      if (allSuccessorsVisited) {
        visited.insert(curr);
        cfStack.pop_back();
      }
    }
  }

  return success();
}

LogicalResult TemporalCodeMotionPass::runOnProcess(llhd::ProcessOp procOp) {
  // Make sure there are no CFG loops that don't contain a block with a wait
  // terminator in the cycle because that's currently not supported by the
  // temporal region analysis and this pass.
  if (failed(checkForCFGLoop(procOp)))
    return failure();

  llhd::TemporalRegionAnalysis trAnalysis =
      llhd::TemporalRegionAnalysis(procOp);
  unsigned numTRs = trAnalysis.getNumTemporalRegions();

  // Only support processes with max. 2 temporal regions and one wait terminator
  // as this is enough to represent flip-flops, registers, etc.
  // NOTE: there always has to be either a wait or halt terminator in a process
  // If the wait block creates the backwards edge, we only have one TR,
  // otherwise we have 2 TRs
  // NOTE: as the wait instruction needs to be on every path around the loop,
  // it has to be the only exiting block of its TR
  // NOTE: the other TR can either have only one exiting block, then we do not
  // need to add an auxillary block, otherwise we have to add one
  // NOTE: All drive operations have to be moved to the single exiting block of
  // its TR. To do so, add the condition under which its block is reached from
  // the TR entry block as a gating condition to the 'llhd.drv' operation
  // NOTE: the entry blocks that are not part of the infinite loop do not count
  // as TR and have TR number -1
  // TODO: need to check that entry blocks that are note part of the loop to not
  // have any instructions that have side effects that should not be allowed
  // outside of the loop (drv, prb, ...)
  // TODO: add support for more TRs and wait terminators (e.g., to represent
  // FSMs)
  if (numTRs > 2)
    return failure();

  bool seenWait = false;
  WalkResult walkResult = procOp.walk([&](llhd::WaitOp op) -> WalkResult {
    if (seenWait)
      return failure();

    // Check that the block containing the wait is the only exiting block of
    // that TR
    int trId = trAnalysis.getBlockTR(op->getBlock());
    if (!trAnalysis.hasSingleExitBlock(trId))
      return failure();

    seenWait = true;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  //===--------------------------------------------------------------------===//
  // Create unique exit block per TR
  //===--------------------------------------------------------------------===//

  // TODO: consider the case where a wait brances to itself
  for (unsigned currTR = 0; currTR < numTRs; ++currTR) {
    unsigned numTRSuccs = trAnalysis.getNumTRSuccessors(currTR);
    (void)numTRSuccs;
    // NOTE: Above error checks make this impossible to trigger, but the above
    // are changed this one might have to be promoted to a proper error message.
    assert((numTRSuccs == 1 ||
            (numTRSuccs == 2 && trAnalysis.isOwnTRSuccessor(currTR))) &&
           "only TRs with a single TR as possible successor are "
           "supported for now.");

    if (trAnalysis.hasSingleExitBlock(currTR))
      continue;

    // Get entry block of successor TR
    Block *succTREntry =
        trAnalysis.getTREntryBlock(*trAnalysis.getTRSuccessors(currTR).begin());

    // Create the auxillary block as we currently don't have a single exiting
    // block and give it the same arguments as the entry block of the
    // successor TR
    Block *auxBlock = new Block();
    auxBlock->addArguments(
        succTREntry->getArgumentTypes(),
        SmallVector<Location>(succTREntry->getNumArguments(), procOp.getLoc()));

    // Insert the auxillary block after the last block of the current TR
    procOp.getBody().getBlocks().insertAfter(
        Region::iterator(trAnalysis.getExitingBlocksInTR(currTR).back()),
        auxBlock);

    // Let all current exit blocks branch to the auxillary block instead.
    for (Block *exit : trAnalysis.getExitingBlocksInTR(currTR))
      for (auto [i, succ] : llvm::enumerate(exit->getSuccessors()))
        if (trAnalysis.getBlockTR(succ) != static_cast<int>(currTR))
          exit->getTerminator()->setSuccessor(auxBlock, i);

    // Let the auxiallary block branch to the entry block of the successor
    // temporal region entry block
    OpBuilder b(procOp);
    b.setInsertionPointToEnd(auxBlock);
    b.create<cf::BranchOp>(procOp.getLoc(), succTREntry,
                           auxBlock->getArguments());
  }

  //===--------------------------------------------------------------------===//
  // Move drive instructions
  //===--------------------------------------------------------------------===//

  DenseMap<Operation *, Block *> drvPos;

  // Force a new analysis as we have changed the CFG
  trAnalysis = llhd::TemporalRegionAnalysis(procOp);
  numTRs = trAnalysis.getNumTemporalRegions();
  OpBuilder builder(procOp);

  for (unsigned currTR = 0; currTR < numTRs; ++currTR) {
    DenseMap<Block *, Value> mem;

    // We ensured this in the previous phase above.
    assert(trAnalysis.getExitingBlocksInTR(currTR).size() == 1);

    Block *exitingBlock = trAnalysis.getExitingBlocksInTR(currTR)[0];
    Block *entryBlock = trAnalysis.getTREntryBlock(currTR);

    DominanceInfo dom(procOp);
    Block *dominator = exitingBlock;

    // Collect all 'llhd.drv' operations in the process and compute their common
    // dominator block.
    procOp.walk([&](llhd::DrvOp op) {
      if (trAnalysis.getBlockTR(op.getOperation()->getBlock()) ==
          static_cast<int>(currTR)) {
        Block *parentBlock = op.getOperation()->getBlock();
        drvPos[op] = parentBlock;
        dominator = dom.findNearestCommonDominator(dominator, parentBlock);
      }
    });

    // Set insertion point before first 'llhd.drv' op in the exiting block
    Operation *moveBefore = exitingBlock->getTerminator();
    exitingBlock->walk([&](llhd::DrvOp op) { moveBefore = op; });

    assert(dominator &&
           "could not find nearest common dominator for TR exiting "
           "block and the block containing drv");

    // If the dominator isn't already a TR entry block, set it to the nearest
    // dominating TR entry block.
    if (trAnalysis.getBlockTR(dominator) != static_cast<int>(currTR))
      dominator = trAnalysis.getTREntryBlock(currTR);

    std::queue<Block *> workQueue;
    SmallPtrSet<Block *, 32> workDone;

    if (entryBlock != exitingBlock)
      workQueue.push(entryBlock);

    while (!workQueue.empty()) {
      Block *block = workQueue.front();
      workQueue.pop();
      workDone.insert(block);

      builder.setInsertionPoint(moveBefore);
      SmallVector<llhd::DrvOp> drives(block->getOps<llhd::DrvOp>());
      for (auto drive : drives)
        moveDriveOpBefore(drive, dominator, moveBefore, mem);

      for (Block *succ : block->getSuccessors()) {
        if (succ == exitingBlock ||
            trAnalysis.getBlockTR(succ) != static_cast<int>(currTR))
          continue;

        if (llvm::all_of(succ->getPredecessors(), [&](Block *block) {
              return workDone.contains(block);
            }))
          workQueue.push(succ);
      }
    }

    // Merge entry and exit block of each TR, remove all other blocks
    if (entryBlock != exitingBlock) {
      entryBlock->getTerminator()->erase();
      entryBlock->getOperations().splice(entryBlock->end(),
                                         exitingBlock->getOperations());
    }
  }

  IRRewriter rewriter(procOp);
  (void)mlir::eraseUnreachableBlocks(rewriter, procOp->getRegions());

  //===--------------------------------------------------------------------===//
  // Coalesce multiple drives to the same signal
  //===--------------------------------------------------------------------===//

  trAnalysis = llhd::TemporalRegionAnalysis(procOp);
  DominanceInfo dom(procOp);
  for (unsigned currTR = 0; currTR < numTRs; ++currTR) {
    // We ensured this in the previous phase above.
    assert(trAnalysis.getExitingBlocksInTR(currTR).size() == 1);

    Block *exitingBlock = trAnalysis.getExitingBlocksInTR(currTR)[0];
    DenseMap<std::pair<Value, Value>, llhd::DrvOp> sigToDrv;

    SmallVector<llhd::DrvOp> drives(exitingBlock->getOps<llhd::DrvOp>());
    for (auto op : drives) {
      auto sigTimePair = std::make_pair(op.getSignal(), op.getTime());
      if (!sigToDrv.count(sigTimePair)) {
        sigToDrv[sigTimePair] = op;
        continue;
      }

      OpBuilder builder(op);
      if (op.getEnable()) {
        // Multiplex value to be driven
        auto firstDrive = sigToDrv[sigTimePair];
        Value muxValue = builder.create<comb::MuxOp>(
            op.getLoc(), op.getEnable(), op.getValue(), firstDrive.getValue());
        op.getValueMutable().assign(muxValue);

        // Take the disjunction of the enable conditions
        if (firstDrive.getEnable()) {
          Value orVal = builder.create<comb::OrOp>(op.getLoc(), op.getEnable(),
                                                   firstDrive.getEnable());
          op.getEnableMutable().assign(orVal);
        } else {
          // No enable is equivalent to a constant 'true' enable
          op.getEnableMutable().clear();
        }
      }

      sigToDrv[sigTimePair]->erase();
      sigToDrv[sigTimePair] = op;
    }
  }

  return success();
}
