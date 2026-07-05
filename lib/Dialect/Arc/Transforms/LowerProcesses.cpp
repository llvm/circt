//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower `llhd.process` ops into `arc.coroutine.define` and
// `arc.coroutine.instance` pairs so the rest of the arcilator pipeline can
// schedule them like any other coroutine.
//
// For each process, the lowering:
//
// - Captures values defined outside the body that are referenced inside.
//   Constant-like ops are cloned into the body; all other values are threaded
//   as coroutine arguments.
//
// - Lifts the body into a new top-level `arc.coroutine.define` whose function
//   type appends an `i64` "now" argument after the captures, and appends an
//   observe bitmask and an `i64` wakeup time result after the process results.
//   The entry block and every block targeted by an `llhd.wait` receive the
//   coroutine's argument types as leading block arguments, and a per-value SSA
//   renaming pass threads each coroutine argument through the CFG so uses see
//   the value bound on the most recent re-entry.
//
// - Rewrites each `llhd.wait` into an `arc.coroutine.yield` (or, for a wait
//   that can never resume, an `arc.coroutine.halt`) whose trailing yield
//   operands are an observe bitmask and a wakeup time. The bitmask has one bit
//   per coroutine argument, set for each observed value; the instance re-enters
//   the coroutine when a set argument changes. The wakeup is `now + delay`
//   (with the delay converted to `i64` via `llhd.time_to_int`, leaving the time
//   unit opaque to this pass), or the unit-independent sentinel `-1` when the
//   wait has no delay. Each `llhd.halt` becomes an `arc.coroutine.halt` with an
//   empty mask and the `-1` wakeup.
//
// - Replaces the original process with an `arc.coroutine.instance` that
//   reads the current simulation time and forwards the captured values.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERPROCESSESPASS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;
using llvm::SmallDenseMap;
using llvm::SmallDenseSet;

namespace {

/// Worker that lowers a single `llhd.process` op.
struct ProcessLowering {
  ProcessLowering(llhd::ProcessOp processOp, SymbolTable &symbolTable)
      : processOp(processOp), symbolTable(symbolTable) {}

  LogicalResult run();

private:
  void collectCaptures();
  void createCoroutine();
  void addEntryAndResumeBlockArguments();
  LogicalResult rewriteTerminators();
  void renameCoroutineArgument(unsigned argIdx, Liveness &liveness,
                               DominanceInfo &dominance);
  void buildInstance();

  llhd::ProcessOp processOp;
  SymbolTable &symbolTable;

  /// The coroutine created for this process. Populated by `createCoroutine`.
  CoroutineDefineOp coroOp;

  /// External SSA values referenced inside the body that need to be threaded
  /// as coroutine arguments.
  SmallVector<Value> captures;

  /// Block arguments of the coroutine entry block, in order: indices `0..N-1`
  /// mirror `captures`, and the final index holds `%now`.
  SmallVector<Value> entryArgs;

  /// Blocks targeted by an `llhd.wait`. Each receives a copy of the
  /// coroutine's argument-type prefix as its leading block arguments. (The
  /// entry block cannot appear here: MLIR entry blocks have no predecessors.)
  SetVector<Block *> resumeBlocks;

  /// Maps each block argument that stands in for a coroutine argument to its
  /// argument index `0..N-1`. This covers the entry block's leading arguments
  /// and every resume block's leading prefix arguments, so multiple block
  /// arguments map to the same index. Populated by
  /// `addEntryAndResumeBlockArguments` and used by `rewriteTerminators` to
  /// build the observe bitmask.
  SmallDenseMap<Value, uint32_t> argIndices;

  /// Indicates which arguments are relevant for the sensitivity check. This
  /// matches the bit-wise OR of all dynamically returned sensitivity masks.
  SmallVector<bool> staticSensitivityMask;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

void ProcessLowering::collectCaptures() {
  auto &body = processOp.getBody();
  auto builder = OpBuilder::atBlockBegin(&body.front());
  SmallDenseSet<Value> captureSet;
  DenseMap<Operation *, Operation *> clonedConstants;
  body.walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      auto value = operand.get();

      // Skip values defined inside the body.
      if (body.isAncestor(value.getParentRegion()))
        continue;

      // Remap references to outside-defined constants to clones of the
      // constants in the entry block.
      auto result = dyn_cast<OpResult>(value);
      if (result && result.getOwner()->hasTrait<OpTrait::ConstantLike>()) {
        auto *&cloned = clonedConstants[result.getOwner()];
        if (!cloned)
          cloned = builder.clone(*result.getOwner());
        operand.set(cloned->getResult(result.getResultNumber()));
        continue;
      }

      // Otherwise this is an outside-defined SSA value we need to capture as an
      // argument of the coroutine.
      if (captureSet.insert(value).second)
        captures.push_back(value);
    }
  });
}

void ProcessLowering::createCoroutine() {
  // Find the parent module and use it as the insertion anchor.
  auto hwModule = processOp->getParentOfType<hw::HWModuleOp>();
  assert(hwModule);
  OpBuilder builder(hwModule);

  // Build the function type:
  // `(captureTypes..., i64) -> (procResults..., iN mask, i64 wakeup)`, where
  // the observe bitmask `iN` carries one bit per coroutine argument and the
  // trailing `i64` is the next wakeup time.
  auto i64Type = builder.getIntegerType(64);
  SmallVector<Type> argTypes;
  for (auto cap : captures)
    argTypes.push_back(cap.getType());
  argTypes.push_back(i64Type);
  auto maskType = builder.getIntegerType(argTypes.size());
  SmallVector<Type> resultTypes(processOp.getResultTypes());
  resultTypes.push_back(maskType);
  resultTypes.push_back(i64Type);
  auto funcType = builder.getFunctionType(argTypes, resultTypes);

  // Pick a symbol name based on the parent hw.module's name. Insertion into
  // the symbol table uniquifies the name.
  auto funcName =
      builder.getStringAttr(hwModule.getSymName() + ".llhd.process");
  coroOp = CoroutineDefineOp::create(builder, processOp.getLoc(), funcName,
                                     funcType);
  symbolTable.insert(coroOp);

  // Move the process body wholesale into the coroutine. Block arguments are
  // added in a separate step.
  coroOp.getBody().takeBody(processOp.getBody());
}

/// Prepend the coroutine function-type prefix as leading block arguments to
/// the entry block and to every block targeted by an `llhd.wait`. After this
/// step the resume blocks satisfy `arc.coroutine.yield`'s contract (its
/// destination's first `N` arguments must match the coroutine's function
/// type) and the entry block's first `N` arguments become the SSA values
/// that the per-argument SSA renaming step will thread through the CFG.
///
/// Predecessor branches of resume blocks that are not `llhd.wait`s (e.g.,
/// `cf.br` from inside the body looping back to a wait target) have their
/// successor operands extended with placeholder values — the entry block
/// arguments themselves — at positions 0..N-1, so that the operand count
/// keeps matching the resume block's new argument count. The SSA renaming
/// step run afterwards walks the body and rewrites these references to the
/// correct reaching definition for the source block.
///
/// Uses of the original outside captures inside the body are also rewritten
/// to the entry block's new capture arguments here, so by the time renaming
/// runs every reference is to an internal SSA value defined at the entry.
void ProcessLowering::addEntryAndResumeBlockArguments() {
  auto &body = coroOp.getBody();
  auto loc = processOp.getLoc();

  // Assemble the initial resume block argument types that will be provided by
  // callers entering or re-entering the coroutine.
  SmallVector<Type> prefixTypes;
  for (auto capture : captures)
    prefixTypes.push_back(capture.getType());
  prefixTypes.push_back(IntegerType::get(coroOp.getContext(), 64));

  // Add arguments to the entry block, recording each one's coroutine argument
  // index so `rewriteTerminators` can map observed values to bitmask bits.
  auto &entry = body.front();
  for (auto [index, type] : llvm::enumerate(prefixTypes)) {
    entryArgs.push_back(entry.insertArgument(index, type, loc));
    argIndices[entryArgs.back()] = index;
  }

  // Collect the resume blocks, which are targets of `llhd.wait` ops.
  for (auto &block : body)
    if (auto wait = dyn_cast<llhd::WaitOp>(block.getTerminator()))
      resumeBlocks.insert(wait.getDest());

  // Pre-add prefix args to each resume block, then extend non-wait predecessor
  // branches with placeholder operands so the destination operand counts keep
  // matching.
  auto prefixSize = prefixTypes.size();
  for (auto *resumeBlock : resumeBlocks) {
    for (auto [index, type] : llvm::enumerate(prefixTypes))
      argIndices[resumeBlock->insertArgument(index, type, loc)] = index;

    // Fix non-wait predecessors. The wait predecessors will become
    // `arc.coroutine.yield`s in the next step; the yield's
    // `getSuccessorOperands` treats the prefix as produced (not forwarded) so
    // its operand count already matches.
    for (auto *pred : resumeBlock->getPredecessors()) {
      auto *term = pred->getTerminator();
      if (isa<llhd::WaitOp>(term))
        continue;
      auto branchOp = cast<BranchOpInterface>(term);
      for (unsigned succIdx = 0, succNum = term->getNumSuccessors();
           succIdx < succNum; ++succIdx) {
        if (term->getSuccessor(succIdx) != resumeBlock)
          continue;
        auto succOperands = branchOp.getSuccessorOperands(succIdx);
        SmallVector<Value> newOperands(entryArgs.begin(),
                                       entryArgs.begin() + prefixSize);
        for (auto v : succOperands.getForwardedOperands())
          newOperands.push_back(v);
        succOperands.getMutableForwardedOperands().assign(newOperands);
      }
    }
  }

  // Replace in-body uses of the original outside captures with the entry
  // block's new arguments. The per-argument SSA renaming step then refines
  // these uses to the right reaching definitions, since each resume block
  // provides its own set of resume arguments, and multiple resume blocks may
  // converge on a single block.
  for (auto [capture, entryArg] : llvm::zip(captures, entryArgs))
    capture.replaceUsesWithIf(entryArg, [&](OpOperand &use) {
      return body.isAncestor(use.getOwner()->getParentRegion());
    });
}

/// Convert `llhd.wait`/`llhd.halt` terminators to their `arc.coroutine`
/// equivalents. The wakeup operand is computed against the entry block's
/// `%now` argument; the per-argument SSA renaming step run afterwards
/// rewrites that reference to the correct block-local definition. A `llhd.wait`
/// with an `observed` clause yields an observe bitmask with the bit of each
/// observed coroutine argument set; ops without a delay and without observed
/// values can never resume and are treated as halts.
LogicalResult ProcessLowering::rewriteTerminators() {
  // The observe bitmask has one bit per coroutine argument. `argIndices` maps
  // the block arguments that stand in for coroutine arguments to their bit
  // index; block arguments beyond that prefix (a wait's forwarded destination
  // operands) and values defined inside the body are absent from the map and
  // therefore cannot be observed for changes.
  auto maskWidth = entryArgs.size();
  staticSensitivityMask.resize(maskWidth, false);
  for (auto &block : coroOp.getBody()) {
    auto *term = block.getTerminator();
    auto loc = term->getLoc();
    OpBuilder builder(term);

    // Handle `llhd.wait` ops.
    if (auto wait = dyn_cast<llhd::WaitOp>(term)) {
      // Build the observe bitmask from the observed operands. Only operands
      // that map to a coroutine argument set a bit; internally-defined values
      // (which can never change) and forwarded block arguments contribute none.
      APInt maskBits(maskWidth, 0);
      for (auto observed : wait.getObserved())
        if (auto it = argIndices.find(observed); it != argIndices.end()) {
          maskBits.setBit(it->second);
          staticSensitivityMask[it->second] = true;
        }
      auto mask = hw::ConstantOp::create(builder, loc, maskBits);

      // A wait without a delay never resumes on time; use the sentinel `-1` as
      // its wakeup. It is only a real suspension if it observes something --
      // otherwise it can never resume and is treated as a halt.
      if (!wait.getDelay()) {
        auto never =
            hw::ConstantOp::create(builder, loc, APInt::getAllOnes(64));
        SmallVector<Value> yieldOperands(wait.getYieldOperands());
        yieldOperands.push_back(mask);
        yieldOperands.push_back(never);
        if (wait.getObserved().empty())
          CoroutineHaltOp::create(builder, loc, yieldOperands);
        else
          CoroutineYieldOp::create(builder, loc, yieldOperands,
                                   wait.getDestOperands(), wait.getDest());
        wait.erase();
        continue;
      }

      auto now = entryArgs.back();
      auto delay = llhd::TimeToIntOp::create(builder, loc, wait.getDelay());
      auto wakeup = comb::AddOp::create(builder, loc, now, delay);
      SmallVector<Value> yieldOperands(wait.getYieldOperands());
      yieldOperands.push_back(mask);
      yieldOperands.push_back(wakeup);
      CoroutineYieldOp::create(builder, loc, yieldOperands,
                               wait.getDestOperands(), wait.getDest());
      wait.erase();
      continue;
    }

    // Handle `llhd.halt` ops.
    if (auto halt = dyn_cast<llhd::HaltOp>(term)) {
      auto mask = hw::ConstantOp::create(builder, loc, APInt(maskWidth, 0));
      auto never = hw::ConstantOp::create(builder, loc, APInt::getAllOnes(64));
      SmallVector<Value> yieldOperands(halt.getYieldOperands());
      yieldOperands.push_back(mask);
      yieldOperands.push_back(never);
      CoroutineHaltOp::create(builder, loc, yieldOperands);
      halt.erase();
      continue;
    }
  }

  return success();
}

/// SSA-rename uses of the coroutine's `argIdx`-th entry-block argument so
/// that every use observes the reaching definition for its block.
///
/// The entry block's argument and the corresponding pre-added argument on
/// each resume block are the per-block "redefinitions" of this value. Any
/// block where two such reaching definitions converge — the iterated
/// dominance frontier of the redefining blocks, restricted to blocks where
/// the value is live-in — receives a fresh merge block argument and any
/// branches reaching it gain the appropriate successor operand. Inside each
/// block we then rewrite uses to that block's reaching definition.
///
/// This mirrors the capture rewriting we do in `LowerCoroutines` as part of the
/// `captureValue` function.
void ProcessLowering::renameCoroutineArgument(unsigned argIdx,
                                              Liveness &liveness,
                                              DominanceInfo &dominance) {
  auto &body = coroOp.getBody();
  auto entryArg = entryArgs[argIdx];
  auto *entryBlock = &body.front();

  // Defining blocks of the value: the entry block plus every resume block.
  SmallPtrSet<Block *, 8> definingBlocks;
  definingBlocks.insert(entryBlock);
  for (auto *resumeBlock : resumeBlocks)
    definingBlocks.insert(resumeBlock);

  // Restrict the IDF to blocks where the value is live-in.
  SmallPtrSet<Block *, 16> liveInBlocks;
  for (auto &block : body)
    if (liveness.getLiveness(&block)->isLiveIn(entryArg))
      liveInBlocks.insert(&block);

  auto &domTree = dominance.getDomTree(&body);
  // Second template arg `false` means forward (not post-) dominance frontier.
  llvm::IDFCalculatorBase<Block, false> idfCalculator(domTree);
  idfCalculator.setDefiningBlocks(definingBlocks);
  idfCalculator.setLiveInBlocks(liveInBlocks);
  SmallVector<Block *> mergeBlocks;
  idfCalculator.calculate(mergeBlocks);

  SmallPtrSet<Block *, 16> mergeBlockSet(mergeBlocks.begin(),
                                         mergeBlocks.end());

  // Walk the dominator tree from the entry block, threading the reaching
  // definition through each block.
  struct WorklistItem {
    DominanceInfoNode *domNode;
    Value reachingDef;
  };
  SmallVector<WorklistItem> worklist;
  worklist.push_back({domTree.getNode(entryBlock), entryArg});

  while (!worklist.empty()) {
    auto item = worklist.pop_back_val();
    auto *block = item.domNode->getBlock();

    if (resumeBlocks.contains(block)) {
      item.reachingDef = block->getArgument(argIdx);
    } else if (mergeBlockSet.contains(block)) {
      item.reachingDef =
          block->addArgument(entryArg.getType(), entryArg.getLoc());
    }

    // Rewrite uses in this block (including nested regions and successor
    // operands of the terminator) to the reaching definition.
    if (item.reachingDef != entryArg)
      block->walk([&](Operation *nested) {
        nested->replaceUsesOfWith(entryArg, item.reachingDef);
      });

    // For each edge leaving this block into a merge block, forward the
    // reaching definition by appending it to the branch op's successor
    // operands. Resume blocks are skipped even when the IDF also flags them
    // as merge blocks: their prefix slots are filled by the runtime on the
    // yield path and by the placeholder operands fixed up by the RAUW above
    // on the non-yield paths, so an append would push an extra operand past
    // the block's argument count.
    auto *terminator = block->getTerminator();
    if (auto branchOp = dyn_cast<BranchOpInterface>(terminator)) {
      for (auto &blockOperand : terminator->getBlockOperands()) {
        auto *succ = blockOperand.get();
        if (!mergeBlockSet.contains(succ) || resumeBlocks.contains(succ))
          continue;
        branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
            .append(item.reachingDef);
      }
    }

    for (auto *child : item.domNode->children())
      worklist.push_back({child, item.reachingDef});
  }
}

/// Create an `arc.coroutine.instance` that instantiates the outlined coroutine
/// in place of the old `llhd.process`.
void ProcessLowering::buildInstance() {
  OpBuilder builder(processOp);
  auto loc = processOp.getLoc();
  auto now = llhd::CurrentTimeOp::create(builder, loc);
  auto nowI64 = llhd::TimeToIntOp::create(builder, loc, now);

  SmallVector<Value> args(captures.begin(), captures.end());
  args.push_back(nowI64);

  assert(args.size() == staticSensitivityMask.size());
  auto instanceOp = CoroutineInstanceOp::create(
      builder, loc, processOp.getResultTypes(),
      FlatSymbolRefAttr::get(coroOp.getSymNameAttr()), args,
      builder.getDenseBoolArrayAttr(staticSensitivityMask));

  processOp.replaceAllUsesWith(instanceOp.getResults());
  processOp.erase();
}

LogicalResult ProcessLowering::run() {
  collectCaptures();
  createCoroutine();
  addEntryAndResumeBlockArguments();
  if (failed(rewriteTerminators()))
    return failure();

  // SSA-rename each coroutine entry argument through the CFG. With a single
  // block, every use is already in the entry block and refers to the entry
  // arg directly, so there is nothing to thread; `DominanceInfo` also asserts
  // on single-block regions.
  if (!coroOp.getBody().hasOneBlock()) {
    Liveness liveness(coroOp);
    DominanceInfo dominance(coroOp);
    for (unsigned argIdx = 0, argNum = entryArgs.size(); argIdx != argNum;
         ++argIdx)
      renameCoroutineArgument(argIdx, liveness, dominance);
  }

  buildInstance();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerProcessesPass
    : public arc::impl::LowerProcessesPassBase<LowerProcessesPass> {
  void runOnOperation() override;
};
} // namespace

void LowerProcessesPass::runOnOperation() {
  auto module = getOperation();
  auto &symbolTable = getAnalysis<SymbolTable>();

  bool anyFailed = false;
  module.walk([&](llhd::ProcessOp op) {
    ProcessLowering lowering(op, symbolTable);
    if (failed(lowering.run()))
      anyFailed = true;
  });
  if (anyFailed)
    signalPassFailure();
}
