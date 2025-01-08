//===- TemporalCodeMotionPass.cpp - Implement Temporal Code Motion Pass ---===//
//
// Implement Pass to move all signal drives in a unique exiting block per
// temporal region and coalesce drives to the same signal.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_TEMPORALCODEMOTIONPASS
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
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
                              OpBuilder &builder,
                              DenseMap<Block *, Value> &mem) {
  Block *drvParentBlock = drvOp->getBlock();

  // Find sequence of branch decisions and add them as a sequence of
  // instructions to the TR exiting block
  Value finalValue = getBranchDecisionsFromDominatorToTarget(
      builder, drvParentBlock, dominator, mem);

  if (drvOp.getEnable())
    finalValue = builder.create<comb::AndOp>(drvOp.getLoc(), drvOp.getEnable(),
                                             finalValue);

  drvOp.getEnableMutable().assign(finalValue);
  drvOp->remove();
  builder.insert(drvOp);
}

namespace {
struct TemporalCodeMotionPass
    : public llhd::impl::TemporalCodeMotionPassBase<TemporalCodeMotionPass> {
  void runOnOperation() override;
  LogicalResult runOnProcess(llhd::ProcessOp procOp);
};
} // namespace

void TemporalCodeMotionPass::runOnOperation() {
  for (auto proc : getOperation().getOps<llhd::ProcessOp>())
    (void)runOnProcess(proc); // Ignore processes that could not be lowered
}

static FailureOr<Block *> createUniqueExitBlock(llhd::ProcessOp procOp) {
  SmallVector<llhd::YieldOp> yields(procOp.getBody().getOps<llhd::YieldOp>());

  // Return failure if there is an infinite loop.
  if (yields.empty())
    return failure();

  // No need to do anything if there is already a unique exit block.
  if (yields.size() == 1)
    return yields[0]->getBlock();

  // Create the auxillary block as we currently don't have a single exiting
  // block and give it the same argument types as the process result types.
  Block *auxBlock = &procOp.getBody().emplaceBlock();
  auxBlock->addArguments(
      procOp.getResults().getTypes(),
      SmallVector<Location>(procOp.getResults().size(), procOp.getLoc()));

  IRRewriter rewriter(OpBuilder::atBlockEnd(auxBlock));
  rewriter.create<llhd::YieldOp>(procOp.getLoc(), auxBlock->getArguments());

  // Let all current exit blocks branch to the auxillary block instead.
  for (auto yieldOp : yields) {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(yieldOp, auxBlock,
                                              yieldOp.getOperands());
  }

  return auxBlock;
}

static void moveDriveOperationsToExitBlock(llhd::ProcessOp procOp,
                                           Block *exitBlock) {
  DenseMap<Operation *, Block *> drvPos;

  DenseMap<Block *, Value> mem;
  std::queue<Block *> workQueue;
  SmallPtrSet<Block *, 32> workDone;

  Block *entryBlock = &procOp.getBody().front();

  if (entryBlock != exitBlock)
    workQueue.push(entryBlock);

  OpBuilder builder = OpBuilder::atBlockBegin(exitBlock);

  while (!workQueue.empty()) {
    Block *block = workQueue.front();
    workQueue.pop();
    workDone.insert(block);

    SmallVector<llhd::DrvOp> drives(block->getOps<llhd::DrvOp>());
    for (auto drive : drives)
      moveDriveOpBefore(drive, entryBlock, builder, mem);

    for (Block *succ : block->getSuccessors()) {
      if (llvm::all_of(
              succ->getPredecessors(),
              [&](Block *block) { return workDone.contains(block); }) &&
          succ != exitBlock)
        workQueue.push(succ);
    }
  }

  // Merge entry and exit block of each TR, remove all other blocks
  if (entryBlock != exitBlock) {
    entryBlock->getTerminator()->erase();
    entryBlock->getOperations().splice(entryBlock->end(),
                                       exitBlock->getOperations());
  }
}

LogicalResult TemporalCodeMotionPass::runOnProcess(llhd::ProcessOp procOp) {
  auto result = createUniqueExitBlock(procOp);

  // If there is no exit block in the process we don't have a place to move
  // drives to, thus just fail.
  if (failed(result))
    return failure();

  // Move drive instructions
  IRRewriter rewriter(procOp);
  moveDriveOperationsToExitBlock(procOp, *result);
  (void)mlir::eraseUnreachableBlocks(rewriter, procOp->getRegions());

  // Coalesce multiple drives to the same signal
  DenseMap<std::pair<Value, Value>, llhd::DrvOp> sigToDrv;

  SmallVector<llhd::DrvOp> drives((*result)->getOps<llhd::DrvOp>());
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

  return success();
}
