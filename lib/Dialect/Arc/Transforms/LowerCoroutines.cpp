//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower `arc.coroutine.define` ops into state machine functions. Coroutines
// are functions that can suspend execution at `arc.coroutine.yield` ops and
// be resumed at a later point. This pass converts each coroutine definition
// into a plain `func.func` that dispatches on an explicit integer program
// counter (PC) argument, and that persists all values live across suspension
// points in an explicit state argument.
//
// The lowering proceeds in two steps. The first step lowers each coroutine
// definition independently of all others:
//
// - Rewrite the body such that every resume block receives the values that
//   are live across its suspension points as trailing block arguments, which
//   makes the state that has to be persisted explicit. Values that do not
//   cross a suspension point are left untouched, and constants are cloned
//   into the resumed blocks instead, since they are cheaper to rematerialize
//   than to persist.
//
// - Assign a PC value to each resume block and derive the concrete PC and
//   state types. The PC type is an integer just wide enough to encode the
//   start PC (0), one resume PC per resume block (1 to N), and the return and
//   halt sentinels (the two largest values). The state type is a union with
//   one struct variant per resume block that has values to persist.
//
// - Replace the definition with a `func.func` that takes the state and PC as
//   leading arguments, dispatches on the PC to either the original entry
//   block or one of the resume blocks (unpacking the corresponding state
//   variant), and returns the new state, resume PC, and yielded values at
//   each suspension point.
//
// At this point, the state and PC types of *other* coroutines may still occur
// as opaque `!arc.coroutine_state` and `!arc.coroutine_pc` types within the
// lowered functions, for example on calls to nested coroutines. The second
// step performs a single global sweep over the module that concretizes all
// such occurrences:
//
// - `arc.coroutine.call` ops become plain `func.call` ops, and the auxiliary
//   coroutine ops become integer constants, comparisons, and poison values.
//
// - All remaining occurrences of the opaque types -- in block arguments,
//   results of unrelated ops, function signatures, and nested within
//   aggregate types -- are replaced by the concrete types computed in step
//   one. Since a coroutine's persistent state may contain the state of the
//   coroutines it calls, this replacement is recursive. Cyclic state
//   containment, i.e. recursive coroutines, is detected and rejected.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERCOROUTINESPASS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Suspension Value Capture
//===----------------------------------------------------------------------===//
//
// The following functions rewrite the body of a coroutine such that every
// resume block receives the values that are live across the suspension points
// it resumes from as trailing block arguments. After the lowering, control
// re-enters a resume block directly from the dispatch logic, so the
// definitions of these values no longer dominate their uses. Capturing them as
// block arguments makes the state that has to be persisted explicit, and the
// lowering picks the arguments up as the contents of the resume block's state
// variant.
//
// Values that do not cross a suspension point are left untouched and keep
// using their original definition through dominance. Where a control flow
// path carrying a captured value rejoins a path carrying the original
// definition, the join block receives a merging block argument as well.
// Constants are not captured; they are cloned into the using blocks that are
// reachable from a resume block instead, since they are cheaper to
// rematerialize on re-entry than to persist across suspension points.

/// Collect the resume blocks of a coroutine body, i.e. the blocks targeted by
/// yield ops, in region order to make the lowering deterministic.
static SmallVector<Block *> collectResumeBlocks(Region &region) {
  SmallPtrSet<Block *, 8> resumeBlockSet;
  for (auto &block : region)
    if (auto yieldOp = dyn_cast<CoroutineYieldOp>(block.getTerminator()))
      resumeBlockSet.insert(yieldOp.getDest());
  SmallVector<Block *> resumeBlocks;
  for (auto &block : region)
    if (resumeBlockSet.contains(&block))
      resumeBlocks.push_back(&block);
  return resumeBlocks;
}

/// Capture a single value as a trailing block argument of each of the given
/// resume blocks and rewrite all affected uses of the value.
static LogicalResult captureValue(Value value, Region &region,
                                  ArrayRef<Block *> captureBlocks,
                                  Liveness &liveness,
                                  DominanceInfo &dominance) {
  auto *defBlock = value.getParentBlock();

  // Determine the blocks that receive an argument for the value. The resume
  // blocks capture it directly. In addition, wherever a path carrying the
  // captured value rejoins a path carrying the original definition, the join
  // block needs a merging argument. These join blocks are the iterated
  // dominance frontier of the capture blocks plus the original definition,
  // pruned to the blocks where the value is live-in.
  auto &domTree = dominance.getDomTree(&region);
  llvm::IDFCalculatorBase<Block, false> idfCalculator(domTree);

  SmallPtrSet<Block *, 8> definingBlocks(captureBlocks.begin(),
                                         captureBlocks.end());
  definingBlocks.insert(defBlock);
  idfCalculator.setDefiningBlocks(definingBlocks);

  SmallPtrSet<Block *, 16> liveInBlocks;
  for (auto &block : region)
    if (liveness.getLiveness(&block)->isLiveIn(value))
      liveInBlocks.insert(&block);
  idfCalculator.setLiveInBlocks(liveInBlocks);

  SmallVector<Block *> mergeBlocks;
  idfCalculator.calculate(mergeBlocks);

  SmallPtrSet<Block *, 16> argBlocks(mergeBlocks.begin(), mergeBlocks.end());
  argBlocks.insert(captureBlocks.begin(), captureBlocks.end());

  // Since the value is an SSA value, its defining block dominates its entire
  // live range, and with it all argument blocks and their predecessors. Walk
  // the dominator tree from there, tracking the reaching definition of the
  // value, which is the original value until an argument block redefines it.
  // Rewrite all uses to the reaching definition of their block, and pass the
  // definition at the end of each block into any argument block successors.
  struct WorklistItem {
    DominanceInfoNode *domNode;
    Value reachingDef;
  };
  SmallVector<WorklistItem> worklist;
  worklist.push_back({domTree.getNode(defBlock), value});

  while (!worklist.empty()) {
    auto item = worklist.pop_back_val();
    auto *block = item.domNode->getBlock();

    if (argBlocks.contains(block))
      item.reachingDef = block->addArgument(value.getType(), value.getLoc());

    // Rewrite the uses in this block, including in nested regions.
    if (item.reachingDef != value)
      block->walk([&](Operation *nestedOp) {
        nestedOp->replaceUsesOfWith(value, item.reachingDef);
      });

    // Append the reaching definition to the successor operands of every edge
    // into an argument block.
    auto *terminator = block->getTerminator();
    auto branchOp = dyn_cast<BranchOpInterface>(terminator);
    for (auto &blockOperand : terminator->getBlockOperands()) {
      if (!argBlocks.contains(blockOperand.get()))
        continue;
      if (!branchOp)
        return terminator->emitOpError()
               << "does not implement `BranchOpInterface`; cannot pass value "
                  "into successor block";
      branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
          .append(item.reachingDef);
    }

    for (auto *child : item.domNode->children())
      worklist.push_back({child, item.reachingDef});
  }

  return success();
}

/// Clone a constant into the using blocks that are reachable from a capturing
/// resume block, where the original definition no longer dominates its uses
/// after the lowering. All other uses keep using the original.
static void rematerializeConstant(Value value, Region &region,
                                  ArrayRef<Block *> captureBlocks,
                                  Liveness &liveness) {
  // Collect the blocks reachable from a capturing resume block in which the
  // value is live-in.
  DenseSet<Block *> resumedBlocks(captureBlocks.begin(), captureBlocks.end());
  SmallVector<Block *> worklist(captureBlocks.begin(), captureBlocks.end());
  while (!worklist.empty())
    for (auto *successor : worklist.pop_back_val()->getSuccessors())
      if (liveness.getLiveIn(successor).contains(value) &&
          resumedBlocks.insert(successor).second)
        worklist.push_back(successor);

  // Redirect the uses in these blocks to a per-block clone of the constant.
  auto *defOp = value.getDefiningOp();
  DenseMap<Block *, Value> clones;
  for (auto &use : llvm::make_early_inc_range(value.getUses())) {
    auto *block = region.findAncestorBlockInRegion(*use.getOwner()->getBlock());
    if (!resumedBlocks.contains(block))
      continue;
    auto &clone = clones[block];
    if (!clone) {
      OpBuilder builder(block, block->begin());
      clone = builder.clone(*defOp)->getResult(0);
    }
    use.set(clone);
  }
  if (defOp->use_empty())
    defOp->erase();
}

/// Rewrite the body of a coroutine such that every resume block captures the
/// values that are live across the suspension points it resumes from as
/// trailing block arguments.
static LogicalResult captureValuesAcrossSuspension(CoroutineDefineOp defineOp) {
  Region &region = defineOp.getBody();
  if (region.hasOneBlock())
    return success();

  auto resumeBlocks = collectResumeBlocks(region);
  if (resumeBlocks.empty())
    return success();

  // Compute the liveness and dominance of the original body once. The
  // per-value rewrites below only ever change the liveness of the value
  // currently being processed, never that of the other collected values, and
  // they add no blocks or control flow edges, so both analyses remain valid
  // throughout the loop.
  Liveness liveness(defineOp);
  DominanceInfo dominance(defineOp);

  // Collect the candidate values up front, since the rewriting below adds new
  // block arguments and constants which need no further treatment.
  SmallVector<Value> values;
  for (auto &block : region) {
    llvm::append_range(values, block.getArguments());
    for (auto &op : block)
      llvm::append_range(values, op.getResults());
  }

  for (auto value : values) {
    // Collect the resume blocks where the value is live-in. Values that are
    // not live across any suspension point still dominate their uses after
    // the lowering and need no rewriting at all.
    SmallVector<Block *> captureBlocks;
    for (auto *block : resumeBlocks)
      if (liveness.getLiveIn(block).contains(value))
        captureBlocks.push_back(block);
    if (captureBlocks.empty())
      continue;

    // Rematerialize constants in the resumed blocks instead of capturing
    // them; capture all other values as trailing resume block arguments.
    auto *defOp = value.getDefiningOp();
    if (defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
      rematerializeConstant(value, region, captureBlocks, liveness);
      continue;
    }
    if (failed(captureValue(value, region, captureBlocks, liveness, dominance)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Coroutine Analysis
//===----------------------------------------------------------------------===//

namespace {
/// Per-coroutine lowering info: the resume blocks and the concrete PC and
/// state types derived from the body after the values live across suspension
/// points have been captured.
struct CoroutineLowering {
  /// The original coroutine definition.
  CoroutineDefineOp defineOp;
  /// The blocks targeted by yield ops, in region order. The PC value of a
  /// resume block is its index in this list plus one.
  SmallVector<Block *> resumeBlocks;
  /// The union field index holding the persisted state of each resume block,
  /// or none if the resume block persists no state.
  SmallVector<std::optional<unsigned>> variantIndices;
  /// The concrete PC type.
  IntegerType pcType;
  /// The concrete state type. This is a union with one struct variant per
  /// state-persisting resume block, and may still contain the opaque state
  /// and PC types of other coroutines.
  Type stateType;

  /// Returns the sentinel PC value indicating that the coroutine returned.
  uint64_t getReturnPC() { return getHaltPC() - 1; }
  /// Returns the sentinel PC value indicating that the coroutine halted.
  uint64_t getHaltPC() {
    return APInt::getAllOnes(pcType.getWidth()).getZExtValue();
  }
  /// Returns the union variant persisting the state of the resume block with
  /// the given index, or none if the resume block persists no state.
  std::optional<hw::UnionType::FieldInfo> getVariant(unsigned resumeIndex) {
    if (auto fieldIndex = variantIndices[resumeIndex])
      return cast<hw::UnionType>(stateType).getElements()[*fieldIndex];
    return std::nullopt;
  }
};
} // namespace

/// Determine the resume blocks of a coroutine and derive its concrete PC and
/// state types. To be called after the values live across suspension points
/// have been captured, such that the trailing block arguments of each resume
/// block are exactly the values that have to be persisted.
static CoroutineLowering analyzeDefinition(CoroutineDefineOp defineOp) {
  auto *context = defineOp.getContext();
  CoroutineLowering lowering;
  lowering.defineOp = defineOp;

  lowering.resumeBlocks = collectResumeBlocks(defineOp.getBody());

  // The PC type must be wide enough to encode the start PC (0), one resume PC
  // per resume block (1 to N), and the return and halt sentinels (the two
  // largest values).
  unsigned pcWidth = llvm::Log2_64_Ceil(lowering.resumeBlocks.size() + 3);
  lowering.pcType = IntegerType::get(context, pcWidth);

  // Build the state type as a union with one struct variant per resume block
  // that has values to persist. The variants are named after the resume PC,
  // and the struct fields after the resume block's trailing arguments.
  unsigned numCoroutineArgs = defineOp.getArgumentTypes().size();
  SmallVector<hw::UnionType::FieldInfo> variants;
  for (auto [index, block] : llvm::enumerate(lowering.resumeBlocks)) {
    auto persistedArgs = block->getArguments().drop_front(numCoroutineArgs);
    if (persistedArgs.empty()) {
      lowering.variantIndices.push_back(std::nullopt);
      continue;
    }
    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto [fieldIndex, arg] : llvm::enumerate(persistedArgs))
      fields.push_back(
          {StringAttr::get(context, "f" + Twine(fieldIndex)), arg.getType()});
    lowering.variantIndices.push_back(variants.size());
    variants.push_back({StringAttr::get(context, "r" + Twine(index + 1)),
                        hw::StructType::get(context, fields), 0});
  }
  if (variants.empty())
    lowering.stateType = hw::StructType::get(context, {});
  else
    lowering.stateType = hw::UnionType::get(context, variants);

  return lowering;
}

//===----------------------------------------------------------------------===//
// Coroutine Lowering
//===----------------------------------------------------------------------===//

/// Replace a coroutine definition with a state machine function. The function
/// takes the persistent state and PC as leading arguments and dispatches on
/// the PC to either the original entry block or, through a trampoline block
/// that unpacks the corresponding state variant, to one of the resume blocks.
/// The coroutine terminators become function returns that produce the new
/// state, resume PC, and yielded values.
static void lowerDefinition(CoroutineLowering &lowering) {
  auto defineOp = lowering.defineOp;
  auto loc = defineOp.getLoc();
  auto *context = defineOp.getContext();
  unsigned pcWidth = lowering.pcType.getWidth();

  // Create the replacement function. The signature wraps the coroutine's
  // function type with the state and PC as leading arguments and results.
  SmallVector<Type> inputTypes{lowering.stateType, lowering.pcType};
  llvm::append_range(inputTypes, defineOp.getArgumentTypes());
  SmallVector<Type> resultTypes{lowering.stateType, lowering.pcType};
  llvm::append_range(resultTypes, defineOp.getResultTypes());
  OpBuilder builder(defineOp);
  auto funcOp =
      func::FuncOp::create(builder, loc, defineOp.getSymName(),
                           builder.getFunctionType(inputTypes, resultTypes));

  // Move the body over and create the dispatch block in front of it.
  funcOp.getBody().getBlocks().splice(funcOp.getBody().end(),
                                      defineOp.getBody().getBlocks());
  auto *entryBlock = &funcOp.getBody().front();
  auto *dispatchBlock =
      builder.createBlock(&funcOp.getBody(), funcOp.getBody().begin());
  Value stateArg = dispatchBlock->addArgument(lowering.stateType, loc);
  Value pcArg = dispatchBlock->addArgument(lowering.pcType, loc);
  SmallVector<Value> callerArgs;
  for (auto arg : entryBlock->getArguments())
    callerArgs.push_back(
        dispatchBlock->addArgument(arg.getType(), arg.getLoc()));

  // Create a trampoline block for each resume block that unpacks the
  // persisted values from the corresponding state variant and passes them to
  // the resume block, alongside the fresh caller-supplied arguments.
  SmallVector<APInt> caseValues;
  SmallVector<Block *> caseBlocks;
  for (auto [index, resumeBlock] : llvm::enumerate(lowering.resumeBlocks)) {
    auto *trampolineBlock = builder.createBlock(resumeBlock);
    SmallVector<Value> operands = callerArgs;
    if (auto variant = lowering.getVariant(index)) {
      Value variantValue =
          hw::UnionExtractOp::create(builder, loc, stateArg, variant->name);
      auto explodeOp = hw::StructExplodeOp::create(builder, loc, variantValue);
      llvm::append_range(operands, explodeOp.getResults());
    }
    cf::BranchOp::create(builder, loc, resumeBlock, operands);
    caseValues.push_back(APInt(pcWidth, index + 1));
    caseBlocks.push_back(trampolineBlock);
  }

  // Dispatch on the PC. The start PC enters the original entry block;
  // passing a return or halt PC is undefined behavior, so those simply fall
  // into the default case alongside the start PC.
  builder.setInsertionPointToEnd(dispatchBlock);
  if (lowering.resumeBlocks.empty()) {
    cf::BranchOp::create(builder, loc, entryBlock, callerArgs);
  } else {
    SmallVector<ValueRange> caseOperands(caseBlocks.size());
    cf::SwitchOp::create(builder, loc, pcArg, entryBlock, callerArgs,
                         caseValues, caseBlocks, caseOperands);
  }

  // Replace the coroutine terminators with function returns producing the new
  // state, resume PC, and yielded values.
  DenseMap<Block *, unsigned> resumePCs;
  for (auto [index, block] : llvm::enumerate(lowering.resumeBlocks))
    resumePCs[block] = index + 1;

  auto lowerTerminator = [&](Operation *op, Value state, uint64_t pc,
                             ValueRange yieldOperands) {
    OpBuilder builder(op);
    if (!state)
      state = ub::PoisonOp::create(builder, op->getLoc(), lowering.stateType,
                                   ub::PoisonAttr::get(context));
    Value pcValue =
        hw::ConstantOp::create(builder, op->getLoc(), APInt(pcWidth, pc));
    SmallVector<Value> operands{state, pcValue};
    llvm::append_range(operands, yieldOperands);
    func::ReturnOp::create(builder, op->getLoc(), operands);
    op->erase();
  };

  for (auto &block : funcOp.getBody()) {
    TypeSwitch<Operation *>(block.getTerminator())
        .Case<CoroutineYieldOp>([&](CoroutineYieldOp op) {
          // Pack the values to persist into the state variant corresponding
          // to the destination block. Resume blocks without persisted state
          // have no variant and return a poison state.
          unsigned pc = resumePCs.lookup(op.getDest());
          Value state;
          if (auto variant = lowering.getVariant(pc - 1)) {
            OpBuilder builder(op);
            Value variantValue = hw::StructCreateOp::create(
                builder, op.getLoc(), variant->type, op.getDestOperands());
            state = hw::UnionCreateOp::create(builder, op.getLoc(),
                                              lowering.stateType, variant->name,
                                              variantValue);
          }
          lowerTerminator(op, state, pc, op.getYieldOperands());
        })
        .Case<CoroutineReturnOp>([&](CoroutineReturnOp op) {
          lowerTerminator(op, Value{}, lowering.getReturnPC(),
                          op.getYieldOperands());
        })
        .Case<CoroutineHaltOp>([&](CoroutineHaltOp op) {
          lowerTerminator(op, Value{}, lowering.getHaltPC(),
                          op.getYieldOperands());
        });
  }

  defineOp.erase();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerCoroutinesPass
    : public arc::impl::LowerCoroutinesPassBase<LowerCoroutinesPass> {
  void runOnOperation() override;
};
} // namespace

void LowerCoroutinesPass::runOnOperation() {
  auto module = getOperation();

  // Collect the coroutine definitions to lower, and reject any leftover
  // coroutine instances in the same pass over the module. Instances are
  // expected to be lowered into explicit storage and calls beforehand;
  // rejecting them here avoids breaking their symbol references.
  SmallVector<CoroutineDefineOp> defineOps;
  auto walkResult = module->walk([&](Operation *op) {
    if (auto instanceOp = dyn_cast<CoroutineInstanceOp>(op)) {
      instanceOp.emitOpError("must be lowered before LowerCoroutines");
      return WalkResult::interrupt();
    }
    if (auto defineOp = dyn_cast<CoroutineDefineOp>(op))
      defineOps.push_back(defineOp);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();

  // Step 1: Lower each coroutine definition independently. Capture the values
  // live across suspension points as trailing resume block arguments and
  // derive the concrete PC and state types. The state types may still contain
  // opaque types of other coroutines, which the global sweep below
  // concretizes.
  DenseMap<StringAttr, CoroutineLowering> lowerings;
  for (auto defineOp : defineOps) {
    if (failed(captureValuesAcrossSuspension(defineOp)))
      return signalPassFailure();

    // The `llhd.resample` markers protecting live event-trigger samples from
    // cross-check-block CSE have done their job: values live across
    // suspension points are now threaded as explicit resume block arguments,
    // so every derived chain is rooted in a block-local binding and CSE can
    // no longer merge samples across check blocks. Fold the markers to their
    // operand. (Process-derived coroutines arrive here already folded by
    // arc-lower-processes.)
    defineOp.getBody().walk([](llhd::ResampleOp resampleOp) {
      resampleOp.getResult().replaceAllUsesWith(resampleOp.getInput());
      resampleOp.erase();
    });

    lowerings.insert({defineOp.getSymNameAttr(), analyzeDefinition(defineOp)});
  }

  // Reject recursive coroutines. A coroutine whose persistent state
  // transitively contains its own state would require unbounded storage.
  // Detect cycles in the state containment graph with a depth-first
  // traversal. This must run before step 2, whose recursive type replacement
  // would not terminate on cycles, and before the definitions are erased, so
  // the error can point at the offending op.
  enum class Color { Unvisited, InProgress, Done };
  DenseMap<StringAttr, Color> colors;
  std::function<LogicalResult(StringAttr)> checkCycles =
      [&](StringAttr name) -> LogicalResult {
    auto it = lowerings.find(name);
    if (it == lowerings.end())
      return success();
    if (colors.lookup(name) == Color::Done)
      return success();
    if (colors.lookup(name) == Color::InProgress)
      return it->second.defineOp.emitOpError(
          "recursive coroutines are not supported");
    colors[name] = Color::InProgress;
    auto result = success();
    it->second.stateType.walk([&](CoroutineStateType type) {
      if (failed(checkCycles(type.getCoroutine().getAttr())))
        result = failure();
    });
    colors[name] = Color::Done;
    return result;
  };
  for (auto defineOp : defineOps)
    if (failed(checkCycles(defineOp.getSymNameAttr())))
      return signalPassFailure();

  // Replace each coroutine definition with a state machine function,
  // completing step 1.
  for (auto defineOp : defineOps)
    lowerDefinition(lowerings.find(defineOp.getSymNameAttr())->second);

  // Step 2: Concretize all occurrences of the opaque coroutine state and PC
  // types throughout the module. The replacer recurses into the replacement
  // types, such that a coroutine state containing the state of a nested
  // coroutine gets fully concretized; the cycle check above guarantees that
  // this recursion terminates.
  bool hasUnknownCoroutines = false;
  auto lookupLowering =
      [&](FlatSymbolRefAttr coroutine) -> CoroutineLowering * {
    auto it = lowerings.find(coroutine.getAttr());
    if (it != lowerings.end())
      return &it->second;
    hasUnknownCoroutines = true;
    mlir::emitError(module.getLoc())
        << "coroutine type references unknown coroutine " << coroutine;
    return nullptr;
  };
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](CoroutineStateType type) -> std::optional<Type> {
    if (auto *lowering = lookupLowering(type.getCoroutine()))
      return lowering->stateType;
    return std::nullopt;
  });
  replacer.addReplacement([&](CoroutinePCType type) -> std::optional<Type> {
    if (auto *lowering = lookupLowering(type.getCoroutine()))
      return lowering->pcType;
    return std::nullopt;
  });

  // Rewrite the ops that consume or produce values of the opaque types. This
  // must happen before the type sweep below, since some of these ops identify
  // their coroutine solely through the symbol carried in their types.
  SmallVector<Operation *> opsToLower;
  module->walk([&](Operation *op) {
    if (isa<CoroutineCallOp, CoroutineStartPCOp, CoroutineUndefinedStateOp,
            CoroutinePCIsReturnOp, CoroutinePCIsHaltOp>(op))
      opsToLower.push_back(op);
  });

  // Determine the PC width for a sentinel check. The PC operand either still
  // has the opaque PC type, or has already been concretized to an integer if
  // its producer was rewritten earlier in the loop below.
  auto getPCWidth = [&](Value pc) -> std::optional<unsigned> {
    if (auto intType = dyn_cast<IntegerType>(pc.getType()))
      return intType.getWidth();
    if (auto intType = dyn_cast<IntegerType>(replacer.replace(pc.getType())))
      return intType.getWidth();
    return std::nullopt;
  };

  for (auto *op : opsToLower) {
    OpBuilder builder(op);
    TypeSwitch<Operation *>(op)
        .Case<CoroutineCallOp>([&](CoroutineCallOp op) {
          // The lowered function signature matches the coroutine call ABI
          // exactly, so the call maps to a plain function call.
          auto resultTypes =
              llvm::map_to_vector(op.getResultTypes(), [&](Type type) {
                return replacer.replace(type);
              });
          auto callOp =
              func::CallOp::create(builder, op.getLoc(), op.getCalleeAttr(),
                                   resultTypes, op.getOperands());
          op->replaceAllUsesWith(callOp);
          op->erase();
        })
        .Case<CoroutineStartPCOp>([&](CoroutineStartPCOp op) {
          auto pcType = dyn_cast<IntegerType>(replacer.replace(op.getType()));
          if (!pcType)
            return;
          Value value = hw::ConstantOp::create(builder, op.getLoc(),
                                               APInt(pcType.getWidth(), 0));
          op->replaceAllUsesWith(ValueRange{value});
          op->erase();
        })
        .Case<CoroutineUndefinedStateOp>([&](CoroutineUndefinedStateOp op) {
          // The state passed on the very first entry into a coroutine is
          // never read, so any value will do.
          auto stateType = replacer.replace(op.getType());
          if (stateType == op.getType())
            return;
          Value value =
              ub::PoisonOp::create(builder, op.getLoc(), stateType,
                                   ub::PoisonAttr::get(builder.getContext()));
          op->replaceAllUsesWith(ValueRange{value});
          op->erase();
        })
        .Case<CoroutinePCIsReturnOp, CoroutinePCIsHaltOp>([&](auto op) {
          // Use the raw operand instead of the typed ODS getter; the PC may
          // already have been concretized to an integer (see `getPCWidth`).
          Value pc = op->getOperand(0);
          auto pcWidth = getPCWidth(pc);
          if (!pcWidth)
            return;
          auto sentinel = APInt::getAllOnes(*pcWidth);
          if (isa<CoroutinePCIsReturnOp>(op))
            sentinel -= 1;
          Value constValue =
              hw::ConstantOp::create(builder, op.getLoc(), sentinel);
          Value cmpValue = comb::ICmpOp::create(
              builder, op.getLoc(), comb::ICmpPredicate::eq, pc, constValue);
          op->replaceAllUsesWith(ValueRange{cmpValue});
          op->erase();
        });
  }

  // Sweep over the entire module and replace all remaining occurrences of the
  // opaque types. This covers block arguments, results of unrelated ops,
  // function signatures, and types nested within aggregates.
  replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
  if (hasUnknownCoroutines)
    return signalPassFailure();
}
