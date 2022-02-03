//===- StandardToHandshake.cpp - Convert standard MLIR into dataflow IR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This is the main Standard to Handshake Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/StandardToHandshake.h"
#include "../PassDetail.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <list>
#include <map>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace std;

typedef DenseMap<Block *, std::vector<Value>> BlockValues;
typedef DenseMap<Block *, std::vector<Operation *>> BlockOps;
typedef DenseMap<Value, Operation *> blockArgPairs;
typedef llvm::MapVector<Value, std::vector<Operation *>> MemRefToMemoryAccessOp;

// ============================================================================
// Partial lowering infrastructure
// ============================================================================

template <typename TFuncOp>
class LowerFuncOpTarget : public ConversionTarget {
public:
  explicit LowerFuncOpTarget(MLIRContext &context) : ConversionTarget(context) {
    loweredFuncs.clear();
    addLegalDialect<HandshakeDialect>();
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<arith::ArithmeticDialect>();
    addIllegalDialect<scf::SCFDialect>();
    addIllegalDialect<mlir::AffineDialect>();
    /// The root function operation to be replaced is marked dynamically legal
    /// based on the lowering status of the given function, see
    /// PartialLowerFuncOp.
    addDynamicallyLegalOp<TFuncOp>(
        [&](const auto &funcOp) { return loweredFuncs[funcOp]; });
  }
  DenseMap<Operation *, bool> loweredFuncs;
};

/// Default function for partial lowering of handshake::FuncOp. Lowering is
/// achieved by a provided partial lowering function.
///
/// A partial lowering function may only replace a subset of the operations
/// within the funcOp currently being lowered. However, the dialect conversion
/// scheme requires the matched root operation to be replaced/updated, if the
/// match was successful. To facilitate this, rewriter.updateRootInPlace
/// wraps the partial update function.
/// Next, the function operation is expected to go from illegal to legalized,
/// after matchAndRewrite returned true. To work around this,
/// LowerFuncOpTarget::loweredFuncs is used to communicate between the target
/// and the conversion, to indicate that the partial lowering was completed.
template <typename TFuncOp>
struct PartialLowerFuncOp : public ConversionPattern {
  using PartialLoweringFunc =
      std::function<LogicalResult(TFuncOp, ConversionPatternRewriter &)>;

public:
  PartialLowerFuncOp(LowerFuncOpTarget<TFuncOp> &target, MLIRContext *context,
                     LogicalResult &loweringResRef,
                     const PartialLoweringFunc &fun)
      : ConversionPattern(TFuncOp::getOperationName(), 1, context),
        target(target), loweringRes(loweringResRef), fun(fun) {}
  using ConversionPattern::ConversionPattern;
  LogicalResult
  matchAndRewrite(Operation *funcOp, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<TFuncOp>(funcOp));
    rewriter.updateRootInPlace(funcOp, [&] {
      loweringRes = fun(dyn_cast<TFuncOp>(funcOp), rewriter);
    });
    target.loweredFuncs[funcOp] = true;
    return loweringRes;
  };

private:
  LowerFuncOpTarget<TFuncOp> &target;
  LogicalResult &loweringRes;
  PartialLoweringFunc fun;
};

/// Calls to rewriter.replaceOpWithNewOp are invalid, given the commitment of
/// all side-effects of of any op replacements are deferred until after the
/// return of matchAndRewrite. This is incompatible with the design of the
/// StandardToHandshake pass, which converts a top-level FuncOp in a sequential
/// manner, assuming any side-effects from operand replacements (in particular
/// rewriting of SSA use-def's) are performed instantly.
/// The following convenience function wraps a replacement pattern @p TConv
/// inside a distinct call to applyPartialConversion, ensuring that any call to
/// rewrite replacement methods have been fully performed after this call. Each
/// replacement pattern is expected to define their own conversion target, to
/// guide pattern matching.
///
template <typename TConv, typename TFuncOp, typename... TArgs>
LogicalResult lowerToHandshake(TFuncOp op, MLIRContext *context,
                               TArgs &...args) {
  RewritePatternSet patterns(context);
  auto target = LowerFuncOpTarget<TFuncOp>(*context);
  LogicalResult partialLoweringSuccessfull = success();
  patterns.add<TConv>(target, context, partialLoweringSuccessfull, args...);
  return success(
      applyPartialConversion(op, target, std::move(patterns)).succeeded() &&
      partialLoweringSuccessfull.succeeded());
}

// Convenience function for running lowerToHandshake with a partial
// handshake::FuncOp lowering function.
template <typename TFuncOp>
LogicalResult partiallyLowerFuncOp(
    const std::function<LogicalResult(TFuncOp, ConversionPatternRewriter &)>
        &loweringFunc,
    MLIRContext *ctx, TFuncOp funcOp) {
  return lowerToHandshake<PartialLowerFuncOp<TFuncOp>>(funcOp, ctx,
                                                       loweringFunc);
}

#define returnOnError(logicalResult)                                           \
  if (failed(logicalResult))                                                   \
    return logicalResult;

// ============================================================================
// Start of lowering passes
// ============================================================================

// A class for maintaining state between all the partial handshake lowering
// passes. The std-to-handshake pass was initially written in such a way that
// all temporary state was added to the IR itself. Consequences of this are that
// the IR is in an inconsistent (non-valid) state during lowering, as well as a
// lot of assumptions being put on the IR itself for tracking things; i.e.
// control values for blocks were always assumed to be any control_merge within
// that block (which may not be the case).
// Through this class, state is made explicit, which should hopefully aid in the
// debugability and extensibility of this pass.
class FuncOpLowering {
public:
  explicit FuncOpLowering(handshake::FuncOp f) : f(f) {}
  LogicalResult addMergeOps(ConversionPatternRewriter &rewriter);
  LogicalResult addBranchOps(ConversionPatternRewriter &rewriter);
  LogicalResult replaceCallOps(ConversionPatternRewriter &rewriter);
  LogicalResult setControlOnlyPath(ConversionPatternRewriter &rewriter);
  LogicalResult connectConstantsToControl(ConversionPatternRewriter &rewriter,
                                          bool sourceConstants);
  LogicalResult loopNetworkRewriting(ConversionPatternRewriter &rewriter);

  BlockOps insertMergeOps(BlockValues blockLiveIns, blockArgPairs &mergePairs,
                          ConversionPatternRewriter &rewriter);
  Operation *insertMerge(Block *block, Value val,
                         ConversionPatternRewriter &rewriter);

  // Replaces standard memory ops with their handshake version (i.e.,
  // ops which connect to memory/LSQ). Returns a map with an ordered
  // list of new ops corresponding to each memref. Later, we instantiate
  // a memory node for each memref and connect it to its load/store ops
  LogicalResult replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemRefToMemoryAccessOp &memRefOps);

  LogicalResult connectToMemory(ConversionPatternRewriter &rewriter,
                                MemRefToMemoryAccessOp memRefOps, bool lsq);
  void setMemOpControlInputs(ConversionPatternRewriter &rewriter,
                             ArrayRef<Operation *> memOps, Operation *memOp,
                             int offset, ArrayRef<int> cntrlInd);

  LogicalResult finalize(ConversionPatternRewriter &rewriter,
                         TypeRange argTypes, TypeRange resTypes,
                         mlir::FuncOp origFunc);

  // Partial lowering driver. This function will instantiate a new
  // partial lowering call, creating a conversion rewriter context which is then
  // passed to the instance of the member function referenced.
  template <typename F, typename... TArgs>
  LogicalResult runPartialLowering(F &&memberFunc, TArgs &...args) {
    return partiallyLowerFuncOp<handshake::FuncOp>(
        [&](handshake::FuncOp,
            ConversionPatternRewriter &rewriter) -> LogicalResult {
          return ((*this).*memberFunc)(rewriter, args...);
        },
        f.getContext(), f);
  }

  // Returns the entry control value for operations contained within this block.
  Value getBlockEntryControl(Block *block) const;
  void setBlockEntryControl(Block *block, Value v);

private:
  DenseMap<Block *, Value> blockEntryControlMap;
  handshake::FuncOp f;
};

Value FuncOpLowering::getBlockEntryControl(Block *block) const {
  auto it = blockEntryControlMap.find(block);
  assert(it != blockEntryControlMap.end() &&
         "No block entry control value registerred for this block!");
  return it->second;
}

void FuncOpLowering::setBlockEntryControl(Block *block, Value v) {
  blockEntryControlMap[block] = v;
}

/// Remove basic blocks inside the given FuncOp. This allows the result to be
/// a valid graph region, since multi-basic block regions are not allowed to
/// be graph regions currently.
void removeBasicBlocks(handshake::FuncOp funcOp) {
  auto &entryBlock = funcOp.getBody().front().getOperations();

  // Erase all TerminatorOp, and move ReturnOp to the end of entry block.
  for (auto &block : funcOp) {
    Operation &termOp = block.back();
    if (isa<handshake::TerminatorOp>(termOp))
      termOp.erase();
    else if (isa<handshake::ReturnOp>(termOp))
      entryBlock.splice(entryBlock.end(), block.getOperations(), termOp);
  }

  // Move all operations to entry block and erase other blocks.
  for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(funcOp, 1))) {
    entryBlock.splice(--entryBlock.end(), block.getOperations());
  }
  for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(funcOp, 1))) {
    block.clear();
    block.dropAllDefinedValueUses();
    for (size_t i = 0; i < block.getNumArguments(); i++) {
      block.eraseArgument(i);
    }
    block.erase();
  }
}

LogicalResult
FuncOpLowering::setControlOnlyPath(ConversionPatternRewriter &rewriter) {
  // Creates start and end points of the control-only path

  // Temporary start node (removed in later steps) in entry block
  Block *entryBlock = &f.front();
  rewriter.setInsertionPointToStart(entryBlock);
  Operation *startOp = rewriter.create<StartOp>(entryBlock->front().getLoc());
  setBlockEntryControl(entryBlock, startOp->getResult(0));

  // Replace original return ops with new returns with additional control input
  for (auto retOp : llvm::make_early_inc_range(f.getOps<mlir::ReturnOp>())) {
    rewriter.setInsertionPoint(retOp);
    SmallVector<Value, 8> operands(retOp->getOperands());
    operands.push_back(startOp->getResult(0));
    rewriter.replaceOpWithNewOp<handshake::ReturnOp>(retOp, operands);
  }
  return success();
}

BlockValues getBlockUses(handshake::FuncOp f) {
  // Returns map of values used in block but defined outside of block
  // For liveness analysis
  BlockValues uses;
  for (Block &block : f) {
    // Operands of operations in b which do not originate from operations or
    // arguments of b
    for (Operation &op : block) {
      for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
        Block *operandBlock;
        if (op.getOperand(i).isa<BlockArgument>()) {
          // Operand is block argument, get its owner block
          operandBlock = op.getOperand(i).cast<BlockArgument>().getOwner();
        } else {
          // Operand comes from operation, get the block of its defining op
          auto *operand = op.getOperand(i).getDefiningOp();
          assert(operand != NULL);
          operandBlock = operand->getBlock();
        }
        // If operand defined in some other block, add to uses
        if (operandBlock != &block)
          // Add only unique uses
          if (std::find(uses[&block].begin(), uses[&block].end(),
                        op.getOperand(i)) == uses[&block].end())
            uses[&block].push_back(op.getOperand(i));
      }
    }
  }
  return uses;
}

BlockValues getBlockDefs(handshake::FuncOp f) {
  // Returns map of values defined in each block
  // For liveness analysis
  BlockValues defs;
  for (Block &block : f) {
    // Values produced by operations in b
    for (Operation &op : block) {
      if (op.getNumResults() > 0) {
        for (auto result : op.getResults())
          defs[&block].push_back(result);
      }
    }
    // Argument values of b
    for (auto &arg : block.getArguments())
      defs[&block].push_back(arg);
  }

  return defs;
}

std::vector<Value> vectorUnion(ArrayRef<Value> v1, ArrayRef<Value> v2) {
  // Returns v1 U v2
  // Assumes unique values in v1
  std::vector<Value> v1c = v1;
  for (int i = 0, e = v2.size(); i < e; ++i) {
    Value val = v2[i];
    if (std::find(v1.begin(), v1.end(), val) == v1.end())
      v1c.push_back(val);
  }
  return v1c;
}

std::vector<Value> vectorDiff(ArrayRef<Value> v1, ArrayRef<Value> v2) {
  // Returns v1 - v2
  std::vector<Value> d;
  for (int i = 0, e = v1.size(); i < e; ++i) {
    Value val = v1[i];
    if (std::find(v2.begin(), v2.end(), val) == v2.end())
      d.push_back(val);
  }
  return d;
}

BlockValues livenessAnalysis(handshake::FuncOp f) {
  // Liveness analysis algorithm adapted from:
  // https://suif.stanford.edu/~courses/cs243/lectures/l2.pdf
  // See slide 19 (Liveness: Iterative Algorithm)

  // blockUses: values used in block but not defined in block
  BlockValues blockUses = getBlockUses(f);

  // blockDefs: values defined in block
  BlockValues blockDefs = getBlockDefs(f);

  BlockValues blockLiveIns;

  bool change = true;
  // Iterate while there are any changes to any of the livein sets
  while (change) {
    change = false;

    // liveOuts(b): all liveins of all successors of b
    // liveOuts(b) = U (blockLiveIns(s)) forall successors s of b
    for (Block &block : f) {
      std::vector<Value> liveOuts;
      for (int i = 0, e = block.getNumSuccessors(); i < e; ++i) {
        Block *succ = block.getSuccessor(i);
        liveOuts = vectorUnion(liveOuts, blockLiveIns[succ]);
      }

      // diff(b):  liveouts of b which are not defined in b
      // diff(b) = liveOuts(b) - blockDefs(b)
      auto diff = vectorDiff(liveOuts, blockDefs[&block]);

      // liveIns(b) = blockUses(b) U diff(b)
      auto tmpLiveIns = vectorUnion(blockUses[&block], diff);

      // Update blockLiveIns if new liveins found
      if (tmpLiveIns.size() > blockLiveIns[&block].size()) {
        blockLiveIns[&block] = tmpLiveIns;
        change = true;
      }
    }
  }
  return blockLiveIns;
}

unsigned getBlockPredecessorCount(Block *block) {
  // Returns number of block predecessors
  auto predecessors = block->getPredecessors();
  return std::distance(predecessors.begin(), predecessors.end());
}

// Insert appropriate type of Merge CMerge for control-only path,
// Merge for single-successor blocks, Mux otherwise
Operation *FuncOpLowering::insertMerge(Block *block, Value val,
                                       ConversionPatternRewriter &rewriter) {
  unsigned numPredecessors = getBlockPredecessorCount(block);

  // Control-only path originates from StartOp
  if (!val.isa<BlockArgument>()) {
    if (isa<StartOp>(val.getDefiningOp())) {
      auto cmerge = rewriter.create<handshake::ControlMergeOp>(
          block->front().getLoc(), val, numPredecessors);
      setBlockEntryControl(block, cmerge.result());
      return cmerge;
    }
  }

  // If there are no block predecessors (i.e., entry block)
  if (numPredecessors <= 1) {
    SmallVector<Value> mergeOperands;
    mergeOperands.append(2, val); // 2 dummy values used here due to hard-coded
                                  // logic of reconnectMergeOps.
    return rewriter.create<handshake::MergeOp>(block->front().getLoc(),
                                               mergeOperands);
  }

  return rewriter.create<handshake::MuxOp>(block->front().getLoc(), val,
                                           numPredecessors);
}

// Adds Merge for every live-in and block argument
// Returns DenseMap of all inserted operations
BlockOps FuncOpLowering::insertMergeOps(BlockValues blockLiveIns,
                                        blockArgPairs &mergePairs,
                                        ConversionPatternRewriter &rewriter) {
  BlockOps blockMerges;
  for (Block &block : f) {
    // Live-ins identified by liveness analysis
    rewriter.setInsertionPointToStart(&block);
    for (auto &val : blockLiveIns[&block]) {
      Operation *newOp = insertMerge(&block, val, rewriter);
      blockMerges[&block].push_back(newOp);
      mergePairs[val] = newOp;
    }
    // Block arguments are not in livein list as they are defined inside the
    // block
    for (auto &arg : block.getArguments()) {
      // No merges on memref block arguments; these are handled separately.
      if (arg.getType().isa<mlir::MemRefType>())
        continue;

      Operation *newOp = insertMerge(&block, arg, rewriter);
      blockMerges[&block].push_back(newOp);
      mergePairs[arg] = newOp;
    }
  }
  return blockMerges;
}

// Check if block contains operation which produces val
bool blockHasSrcOp(Value val, Block *block) {
  // Arguments do not have an operation producer
  if (val.isa<BlockArgument>())
    return false;

  auto *op = val.getDefiningOp();
  assert(op != NULL);
  return (op->getBlock() == block);
}

// Get value from predBlock which will be set as operand of op (merge)
Value getMergeOperand(Operation *op, Block *predBlock, BlockOps blockMerges) {
  // Helper value (defining value of merge) to identify Merges which propagate
  // the same defining value
  Value srcVal = op->getOperand(0);
  Block *block = op->getBlock();

  // Value comes from predecessor block (i.e., not an argument of this block)
  if (std::find(block->getArguments().begin(), block->getArguments().end(),
                srcVal) == block->getArguments().end()) {
    // Value is not defined by operation in predBlock
    if (!blockHasSrcOp(srcVal, predBlock)) {
      // Find the corresponding Merge
      for (Operation *predOp : blockMerges[predBlock])
        if (predOp->getOperand(0) == srcVal)
          return predOp->getResult(0);
    } else
      return srcVal;
  }

  // Value is argument of this block
  // Operand of terminator in predecessor block should be input to Merge
  else {
    unsigned index = srcVal.cast<BlockArgument>().getArgNumber();
    Operation *termOp = predBlock->getTerminator();
    if (mlir::CondBranchOp br = dyn_cast<mlir::CondBranchOp>(termOp)) {
      if (block == br.getTrueDest())
        return br.getTrueOperand(index);
      else {
        assert(block == br.getFalseDest());
        return br.getFalseOperand(index);
      }
    } else if (isa<mlir::BranchOp>(termOp))
      return termOp->getOperand(index);
  }
  return nullptr;
}

void removeBlockOperands(handshake::FuncOp f) {
  // Remove all block arguments, they are no longer used
  // eraseArguments also removes corresponding branch operands
  for (Block &block : f) {
    if (!block.isEntryBlock()) {
      int x = block.getNumArguments() - 1;
      for (int i = x; i >= 0; --i)
        block.eraseArgument(i);
    }
  }
}

/// Returns the first occurance of an operation of type TOp, else, returns
/// null op.
template <typename TOp>
Operation *getFirstOp(Block *block) {
  auto ops = block->getOps<TOp>();
  if (ops.empty())
    return nullptr;
  return *ops.begin();
}

Operation *getControlMerge(Block *block) {
  return getFirstOp<ControlMergeOp>(block);
}

ConditionalBranchOp getControlCondBranch(Block *block) {
  for (auto cbranch : block->getOps<handshake::ConditionalBranchOp>()) {
    if (cbranch.isControl())
      return cbranch;
  }
  return nullptr;
}

void reconnectMergeOps(handshake::FuncOp f, BlockOps blockMerges,
                       blockArgPairs &mergePairs) {
  // All merge operands are initially set to original (defining) value
  // We here replace defining value with appropriate value from predecessor
  // block The predecessor can either be a merge, the original defining value,
  // or a branch Operand Operand(0) is helper defining value for identifying
  // matching merges, it does not correspond to any predecessor block

  for (Block &block : f) {
    for (Operation *op : blockMerges[&block]) {
      int count = 1;
      // Set appropriate operand from predecessor block
      for (auto *predBlock : block.getPredecessors()) {
        Value mgOperand = getMergeOperand(op, predBlock, blockMerges);
        assert(mgOperand != nullptr);
        if (!mgOperand.getDefiningOp()) {
          assert(mergePairs.count(mgOperand));
          mgOperand = mergePairs[mgOperand]->getResult(0);
        }
        op->setOperand(count, mgOperand);
        count++;
      }
      // Reconnect all operands originating from livein defining value through
      // corresponding merge of that block
      for (Operation &opp : block)
        if (!isa<MergeLikeOpInterface>(opp))
          opp.replaceUsesOfWith(op->getOperand(0), op->getResult(0));
    }
  }

  // Disconnect original value (Operand(0), used as helper) from all merges
  // If the original value must be a merge operand, it is still set as some
  // subsequent operand
  // If block has multiple predecessors, connect Muxes to ControlMerge
  for (Block &block : f) {
    unsigned numPredecessors = getBlockPredecessorCount(&block);

    if (numPredecessors <= 1) {
      for (Operation *op : blockMerges[&block])
        op->eraseOperand(0);
    } else {
      Operation *cntrlMg = getControlMerge(&block);
      assert(cntrlMg != nullptr);
      cntrlMg->eraseOperand(0);

      for (Operation *op : blockMerges[&block])
        if (op != cntrlMg)
          op->setOperand(0, cntrlMg->getResult(1));
    }
  }

  removeBlockOperands(f);
}

LogicalResult FuncOpLowering::addMergeOps(ConversionPatternRewriter &rewriter) {

  blockArgPairs mergePairs;

  // blockLiveIns: live in variables of block
  BlockValues liveIns = livenessAnalysis(f);

  // Insert merge operations
  BlockOps mergeOps = insertMergeOps(liveIns, mergePairs, rewriter);

  // Set merge operands and uses
  reconnectMergeOps(f, mergeOps, mergePairs);
  return success();
}

bool isLiveOut(Value val) {
  // Identifies liveout values after adding Merges
  for (auto &u : val.getUses())
    // Result is liveout if used by some Merge block
    if (isa<MergeLikeOpInterface>(u.getOwner()))
      return true;
  return false;
}

// A value can have multiple branches in a single successor block
// (for instance, there can be an SSA phi and a merge that we insert)
// This function determines the number of branches to insert based on the
// value uses in successor blocks
int getBranchCount(Value val, Block *block) {
  int uses = 0;
  for (int i = 0, e = block->getNumSuccessors(); i < e; ++i) {
    int curr = 0;
    Block *succ = block->getSuccessor(i);
    for (auto &u : val.getUses()) {
      if (u.getOwner()->getBlock() == succ)
        curr++;
    }
    uses = (curr > uses) ? curr : uses;
  }
  return uses;
}

static bool hasMultiplePredecessors(Block *block) {
  return std::distance(block->getPredecessors().begin(),
                       block->getPredecessors().end()) > 1;
}

// The BFS callback provides the current block as an argument and returns
// whether search should halt.
enum class BFSNextState { Halt, SkipSuccessors, Continue };
using BFSCallback = llvm::function_ref<BFSNextState(Block *)>;

static void blockBFS(Block *start, BFSCallback callback) {
  DenseSet<Block *> visited;
  SmallVector<Block *> queue = {start};
  while (!queue.empty()) {
    Block *currBlock = queue.front();
    queue.erase(queue.begin());
    visited.insert(currBlock);

    switch (callback(currBlock)) {
    case BFSNextState::Halt:
      return;
    case BFSNextState::Continue: {
      llvm::copy_if(currBlock->getSuccessors(), std::back_inserter(queue),
                    [&](Block *block) { return visited.count(block) == 0; });
      break;
    }
    case BFSNextState::SkipSuccessors:
      for (auto *succ : currBlock->getSuccessors())
        visited.insert(succ);
      continue;
    }
  }
}

// Performs a BFS to determine whether there exists a path between 'from' and
// 'to'.
static bool isReachable(Block *from, Block *to) {
  bool isReachable = false;
  blockBFS(from, [&](Block *currBlock) {
    if (currBlock == to) {
      isReachable = true;
      return BFSNextState::Halt;
    }
    return BFSNextState::Continue;
  });
  return isReachable;
}

// This function creates the loop 'continue' and 'exit' network around backedges
// in the CFG.
// We don't have a standard dialect based LoopInfo utility in MLIR
// (which could probably deduce most of the information that we need for this
// transformation), so we roll our own loop-detection analysis. This is
// simplified by the need to only detect outermost loops. Inner loops are
// not included in the loop network (since we only care about restricting
// different function invocations from activating a loop, not prevent loop
// pipelining within a single function invocation).
class LoopNetworkRewriter {
public:
  LoopNetworkRewriter(FuncOpLowering &fol) : fol(fol) {}

  LogicalResult processFunction(handshake::FuncOp f,
                                ConversionPatternRewriter &rewriter);

private:
  // An exit pair is a pair of <in loop block, outside loop block> that
  // indicates where control leaves a loop.
  using ExitPair = std::pair<Block *, Block *>;
  LogicalResult processOuterLoop(Location loc, Block *loopHeader,
                                 Block *loopLatch);

  // Builds the loop continue network in between the loop header and its loop
  // latch. The loop continuation network will replace the existing control
  // merge in the loop header with a mux + loop priming register.
  // The 'loopPrimingInput' is a backedge that will later be assigned by
  // 'buildExitNetwork'. The value is to be used as the input to the loop
  // priming buffer.
  // Returns a reference to the loop priming register.
  BufferOp buildContinueNetwork(Block *loopHeader, Block *loopLatch,
                                Backedge &loopPrimingInput);

  // Builds the loop exit network. This detects the conditional operands used in
  // each of the exit blocks, matches their parity with the convention used to
  // prime the loop register, and assigns it to the loop priming register input.
  void buildExitNetwork(Block *loopHeader, const std::set<ExitPair> &exitPairs,
                        BufferOp loopPrimingRegister,
                        Backedge &loopPrimingInput);

private:
  ConversionPatternRewriter *rewriter = nullptr;
  std::set<Block *> visited;
  std::list<Block *> queue;
  DominanceInfo domInfo;
  FuncOpLowering &fol;
};

LogicalResult
FuncOpLowering::loopNetworkRewriting(ConversionPatternRewriter &rewriter) {
  return LoopNetworkRewriter(*this).processFunction(f, rewriter);
}

LogicalResult
LoopNetworkRewriter::processFunction(handshake::FuncOp f,
                                     ConversionPatternRewriter &rewriter) {
  this->rewriter = &rewriter;

  // Initialize the BFS queue with the entry block.
  queue = {&f.getRegion().front()};
  domInfo = DominanceInfo(f);

  while (!queue.empty()) {
    Block *currBlock = queue.front();
    queue.pop_front();
    visited.insert(currBlock);

    if (!hasMultiplePredecessors(currBlock)) {
      // Not a loop header;
      llvm::copy_if(currBlock->getSuccessors(), std::back_inserter(queue),
                    [&](Block *block) { return !visited.count(block); });
    } else {
      // Figure out which predecessors are loop latches
      std::set<Block *> loopLatches;
      for (auto *backedge : currBlock->getPredecessors()) {
        bool dominates = domInfo.dominates(currBlock, backedge);
        bool isLoop = isReachable(currBlock, backedge);
        if (isLoop) {
          if (dominates)
            loopLatches.insert(backedge);
          else
            return f.emitError()
                   << "Non-canonical loop structures detected; a potential "
                      "loop header has backedges not dominated by the loop "
                      "header. This indicates that the loop has multiple entry "
                      "points. Handshake lowering does not yet support this "
                      "form of control flow, exiting.";
        }
      }

      if (!loopLatches.empty()) {
        if (loopLatches.size() > 1)
          return emitError(f.getLoc()) << "Multiple loop latches detected "
                                          "(backedges from within the loop "
                                          "to the loop header). Loop task "
                                          "pipelining is only supported for "
                                          "loops with unified loop latches.";

        // This is the start of an outer loop - go process!
        if (failed(
                processOuterLoop(f.getLoc(), currBlock, *loopLatches.begin())))
          return failure();
      }
    }
  }

  return success();
}

// Returns the operand of the 'mux' operation which originated from 'block'.
static Value getOperandFromBlock(MuxOp mux, Block *block) {
  auto inValueIt = llvm::find_if(mux.dataOperands(), [&](Value operand) {
    return block == operand.getParentBlock();
  });
  assert(
      inValueIt != mux.getOperands().end() &&
      "Expected mux to have an operand originating from the requested block.");
  return *inValueIt;
}

// Returns a list of operands from 'mux' which corresponds to the inputs of the
// 'cmerge' operation. The results are sorted such that the i'th cmerge operand
// and the i'th sorted operand originate from the same block.
static std::vector<Value> getSortedInputs(ControlMergeOp cmerge, MuxOp mux) {
  std::vector<Value> sortedOperands;
  for (auto in : cmerge.getOperands()) {
    auto *srcBlock = in.getParentBlock();
    sortedOperands.push_back(getOperandFromBlock(mux, srcBlock));
  }

  // Sanity check: ensure that operands are unique
  for (unsigned i = 0; i < sortedOperands.size(); ++i) {
    for (unsigned j = 0; j < sortedOperands.size(); ++j) {
      if (i == j)
        continue;
      assert(sortedOperands[i] != sortedOperands[j] &&
             "Cannot have an identical operand from two different blocks!");
    }
  }

  return sortedOperands;
}

BufferOp LoopNetworkRewriter::buildContinueNetwork(Block *loopHeader,
                                                   Block *loopLatch,
                                                   Backedge &loopPrimingInput) {
  // Gather the muxes to replace before modifying block internals; it's been
  // found that if this is not done, we have determinism issues wrt. generating
  // the same order of replaced muxes on repeated runs of an identical
  // conversion.
  llvm::SmallVector<MuxOp> muxesToReplace;
  llvm::copy(loopHeader->getOps<MuxOp>(), std::back_inserter(muxesToReplace));

  // Fetch the control merge of the block; it is assumed that, at this point of
  // lowering, no other form of control can be used for the loop header block
  // than a control merge.
  auto *cmerge = getControlMerge(loopHeader);
  assert(fol.getBlockEntryControl(loopHeader) == cmerge->getResult(0) &&
         "Expected control merge to be the control component of a loop header");
  auto loc = cmerge->getLoc();

  // sanity check: cmerge should have >1 input to actually be a loop
  assert(cmerge->getNumOperands() > 1 && "This cannot be a loop header");

  // Partition the control merge inputs into those originating from backedges,
  // and those originating elsewhere.
  SmallVector<Value> externalCtrls, loopCtrls;
  for (auto cval : cmerge->getOperands()) {
    if (cval.getParentBlock() == loopLatch)
      loopCtrls.push_back(cval);
    else
      externalCtrls.push_back(cval);
  }
  assert(loopCtrls.size() == 1 &&
         "Expected a single loop control value to match the single loop latch");
  Value loopCtrl = loopCtrls.front();

  // Merge all of the controls in each partition
  rewriter->setInsertionPointToStart(loopHeader);
  auto externalCtrlMerge = rewriter->create<ControlMergeOp>(loc, externalCtrls);

  // Create loop mux and the loop priming register. The loop mux will on select
  // "0" select external control, and internal control at "1". This convention
  // must be followed by the loop exit network.
  auto primingRegister =
      rewriter->create<BufferOp>(loc, rewriter->getI1Type(), /*size=*/1,
                                 loopPrimingInput, /*sequential=*/true);
  // Initialize the priming register to path 0.
  primingRegister->setAttr("initValues", rewriter->getI64ArrayAttr({0}));

  // The loop control mux will deterministically select between control entering
  // the loop from any external block or the single loop backedge.
  auto loopCtrlMux = rewriter->create<MuxOp>(
      loc, primingRegister.getResult(),
      llvm::SmallVector<Value>{externalCtrlMerge.result(), loopCtrl});

  // Replace the existing control merge 'result' output with the loop control
  // mux.
  cmerge->getResult(0).replaceAllUsesWith(loopCtrlMux.getResult());

  // Register the new block entry control value
  fol.setBlockEntryControl(loopHeader, loopCtrlMux.getResult());

  // Next, we need to consider how to replace the control merge 'index' output,
  // used to drive input selection to the block.

  // Inputs to the loop header will be sourced from muxes with inputs from both
  // the loop latch as well as external blocks. Iterate through these and sort
  // based on the input ordering to the external/internal control merge.
  // We do this by maintaining a mapping between the external and loop data
  // inputs for each data mux in the design. The key of these maps is the
  // original mux (that is to be replaced).
  DenseMap<MuxOp, std::vector<Value>> externalDataInputs;
  DenseMap<MuxOp, Value> loopDataInputs;
  for (auto muxOp : muxesToReplace) {
    if (muxOp == loopCtrlMux)
      continue;

    externalDataInputs[muxOp] = getSortedInputs(externalCtrlMerge, muxOp);
    loopDataInputs[muxOp] = getOperandFromBlock(muxOp, loopLatch);
    assert(/*loop latch input*/ 1 + externalDataInputs[muxOp].size() ==
               muxOp.dataOperands().size() &&
           "Expected all mux operands to be partitioned between loop and "
           "external data inputs");
  }

  // With this, we now replace each of the data input muxes in the loop header.
  // We instantiate a single mux driven by the external control merge.
  // This, as well as the corresponding data input coming from within the single
  // loop latch, will then be selected between by a 3rd mux, based on the
  // priming register.
  for (MuxOp mux : muxesToReplace) {
    auto externalDataMux = rewriter->create<MuxOp>(
        loc, externalCtrlMerge.index(), externalDataInputs[mux]);

    rewriter->replaceOp(
        mux, rewriter
                 ->create<MuxOp>(loc, primingRegister,
                                 llvm::SmallVector<Value>{externalDataMux,
                                                          loopDataInputs[mux]})
                 .getResult());
  }

  // Now all values defined by the original cmerge should have been replaced,
  // and it may be erased.
  rewriter->eraseOp(cmerge);

  // Return the priming register to be referenced by the exit network builder.
  return primingRegister;
}

void LoopNetworkRewriter::buildExitNetwork(Block *loopHeader,
                                           const std::set<ExitPair> &exitPairs,
                                           BufferOp loopPrimingRegister,
                                           Backedge &loopPrimingInput) {
  auto loc = loopPrimingRegister.getLoc();

  // Iterate over the exit pairs to gather up the condition signals that need to
  // be connected to the exit network. In doing so, we parity-correct these
  // condition values based on the convention used in buildContinueNetwork - The
  // loop mux will on select "0" select external control, and internal control
  // at "1". This convention which must be followed by the loop exit network.
  // External control must be selected when exiting the loop (to reprime the
  // register).
  SmallVector<Value> parityCorrectedConds;
  for (auto &[condBlock, exitBlock] : exitPairs) {
    auto condBr = getControlCondBranch(condBlock);
    assert(
        condBr &&
        "Expected a conditional control branch op in the loop condition block");
    Operation *trueUser = *condBr.trueResult().getUsers().begin();
    Operation *falseUser = *condBr.falseResult().getUsers().begin();
    bool isTrueParity = trueUser->getBlock() == exitBlock;
    bool isFalseParity = falseUser->getBlock() == exitBlock;
    assert(isTrueParity ^ isFalseParity &&
           "The user of either the true or the false result should be in the "
           "exit block");

    Value condValue = condBr.conditionOperand();
    if (isTrueParity) {
      // This goes against the convention, and we have to invert the condition
      // value before connecting it to the exit network.
      rewriter->setInsertionPoint(condBr);
      condValue = rewriter->create<arith::XOrIOp>(
          loc, condValue.getType(), condValue,
          rewriter->create<arith::ConstantOp>(
              loc, rewriter->getIntegerAttr(rewriter->getI1Type(), 1)));
    }
    parityCorrectedConds.push_back(condValue);
  }

  // Merge all of the parity-corrected exit conditions and assign them
  // to the loop priming input.
  auto exitMerge = rewriter->create<MergeOp>(loc, parityCorrectedConds);
  loopPrimingInput.setValue(exitMerge);
}

LogicalResult LoopNetworkRewriter::processOuterLoop(Location loc,
                                                    Block *loopHeader,
                                                    Block *loopLatch) {
  // To build the exit network, we need to determine the exit blocks of the
  // loop, as well as all blocks contained within the loop. Exit blocks are the
  // blocks that control is transfered to after exiting the loop. This is
  // essentially determining the strongly connected components with the loop
  // header. We perform a BFS from the loop header, and if the loop header is
  // reachable the block, it is within the loop.
  SetVector<Block *> inLoop, exitBlocks;
  blockBFS(loopHeader, [&](Block *currBlock) {
    if (isReachable(currBlock, loopHeader)) {
      inLoop.insert(currBlock);
      return BFSNextState::Continue;
    }
    exitBlocks.insert(currBlock);
    return BFSNextState::SkipSuccessors;
  });

  assert(inLoop.size() >= 2 && "A loop must have at least 2 blocks");
  assert(exitBlocks.size() != 0 && "A loop must have an exit block...?");

  // Then we determine the exit pairs of the loop; this is the in-loop nodes
  // which branch off to the exit nodes.
  std::set<ExitPair> exitPairs;
  for (auto *exitNode : exitBlocks) {
    for (auto *pred : exitNode->getPredecessors()) {
      // is the predecessor inside the loop?
      if (!inLoop.contains(pred))
        continue;

      ExitPair condPair = {pred, exitNode};
      assert(!exitPairs.count(condPair) &&
             "identical condition pairs should never be possible");
      exitPairs.insert(condPair);
    }
  }
  assert(exitPairs.size() > 0 && "No exits from loop?");

  // The first precondition to our loop transformation is that only a single
  // exit pair exists in the loop.
  if (exitPairs.size() > 1)
    return emitError(loc)
           << "Multiple exits detected within a loop. Loop task pipelining is "
              "only supported for loops with unified loop exit blocks.";

  BackedgeBuilder bebuilder(*rewriter, loopHeader->front().getLoc());

  // Build the loop continue network. Loop continuation is triggered solely by
  // backedges to the header.
  auto loopPrimingRegisterInput = bebuilder.get(rewriter->getI1Type());
  auto loopPrimingRegister =
      buildContinueNetwork(loopHeader, loopLatch, loopPrimingRegisterInput);

  // Build the loop exit network. Loop exiting is driven solely by exit pairs
  // from the loop.
  buildExitNetwork(loopHeader, exitPairs, loopPrimingRegister,
                   loopPrimingRegisterInput);

  // Finally, we must reprime the BFS in processFunction; it must skip all
  // of the inLoop nodes - since they have now been captured within the loop
  // network that we just built- and restart from the exitBlocks - since they
  // are guaranteed to be outside of the loop, but more loops may be
  // downstream of these blocks.
  llvm::copy(inLoop, std::inserter(visited, visited.begin()));
  llvm::copy_if(exitBlocks, std::back_inserter(queue),
                [&](Block *exitBlock) { return !visited.count(exitBlock); });
  return success();
}

// Return the appropriate branch result based on successor block which uses it
Value getSuccResult(Operation *termOp, Operation *newOp, Block *succBlock) {
  // For conditional block, check if result goes to true or to false successor
  if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(termOp)) {
    if (condBranchOp.getTrueDest() == succBlock)
      return dyn_cast<handshake::ConditionalBranchOp>(newOp).trueResult();
    else {
      assert(condBranchOp.getFalseDest() == succBlock);
      return dyn_cast<handshake::ConditionalBranchOp>(newOp).falseResult();
    }
  }
  // If the block is unconditional, newOp has only one result
  return newOp->getResult(0);
}

LogicalResult
FuncOpLowering::addBranchOps(ConversionPatternRewriter &rewriter) {

  BlockValues liveOuts;

  for (Block &block : f) {
    for (Operation &op : block) {
      for (auto result : op.getResults())
        if (isLiveOut(result))
          liveOuts[&block].push_back(result);
    }
  }

  for (Block &block : f) {
    Operation *termOp = block.getTerminator();
    rewriter.setInsertionPoint(termOp);

    for (Value val : liveOuts[&block]) {
      // Count the number of branches which the liveout needs
      int numBranches = getBranchCount(val, &block);

      // Instantiate branches and connect to Merges
      for (int i = 0, e = numBranches; i < e; ++i) {
        Operation *newOp = nullptr;

        if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(termOp))
          newOp = rewriter.create<handshake::ConditionalBranchOp>(
              termOp->getLoc(), condBranchOp.getCondition(), val);
        else if (isa<mlir::BranchOp>(termOp))
          newOp = rewriter.create<handshake::BranchOp>(termOp->getLoc(), val);

        if (newOp == nullptr)
          continue;

        for (int j = 0, e = block.getNumSuccessors(); j < e; ++j) {
          Block *succ = block.getSuccessor(j);
          Value res = getSuccResult(termOp, newOp, succ);

          for (auto &u : val.getUses()) {
            if (u.getOwner()->getBlock() == succ) {
              u.getOwner()->replaceUsesOfWith(val, res);
              break;
            }
          }
        }
      }
    }
  }

  // Remove StandardOp branch terminators and place new terminator
  // Should be removed in some subsequent pass (we only need it to pass checks
  // in Verifier.cpp)
  for (Block &block : f) {
    Operation *termOp = block.getTerminator();
    if (!(isa<mlir::CondBranchOp>(termOp) || isa<mlir::BranchOp>(termOp)))
      continue;

    SmallVector<mlir::Block *, 8> results(block.getSuccessors());
    rewriter.setInsertionPointToEnd(&block);
    rewriter.create<handshake::TerminatorOp>(termOp->getLoc(), results);

    // Remove the Operands to keep the single-use rule.
    for (int i = 0, e = termOp->getNumOperands(); i < e; ++i)
      termOp->eraseOperand(0);
    assert(termOp->getNumOperands() == 0);
    rewriter.eraseOp(termOp);
  }
  return success();
}

LogicalResult
FuncOpLowering::connectConstantsToControl(ConversionPatternRewriter &rewriter,
                                          bool sourceConstants) {
  // Create new constants which have a control-only input to trigger them. These
  // are conneted to the control network or optionally to a Source operation
  // (always triggering). Control-network connected constants may help
  // debugability, but result in a slightly larger circuit.

  if (sourceConstants) {
    for (auto constantOp :
         llvm::make_early_inc_range(f.getOps<arith::ConstantOp>())) {
      rewriter.setInsertionPointAfter(constantOp);
      rewriter.replaceOpWithNewOp<handshake::ConstantOp>(
          constantOp, constantOp.getValue(),
          rewriter.create<handshake::SourceOp>(constantOp.getLoc()));
    }
  } else {
    for (Block &block : f) {
      Value blockEntryCtrl = getBlockEntryControl(&block);
      for (auto constantOp :
           llvm::make_early_inc_range(block.getOps<arith::ConstantOp>())) {
        rewriter.setInsertionPointAfter(constantOp);
        rewriter.replaceOpWithNewOp<handshake::ConstantOp>(
            constantOp, constantOp.getValue(), blockEntryCtrl);
      }
    }
  }
  return success();
}

LogicalResult checkUseCount(Operation *op, Value res) {
  // Checks if every result has single use
  if (!res.hasOneUse()) {
    int i = 0;
    for (auto *user : res.getUsers()) {
      user->emitWarning("user here");
      i++;
    }
    return op->emitOpError("every result must have exactly one user, but had ")
           << i;
  }
  return success();
}

// Checks if op successors are in appropriate blocks
LogicalResult checkSuccessorBlocks(Operation *op, Value res) {
  for (auto &u : res.getUses()) {
    Operation *succOp = u.getOwner();
    // Non-branch ops: succesors must be in same block
    if (!(isa<handshake::ConditionalBranchOp>(op) ||
          isa<handshake::BranchOp>(op))) {
      if (op->getBlock() != succOp->getBlock())
        return op->emitOpError("cannot be block live-out");
    } else {
      // Branch ops: must have successor per successor block
      if (op->getBlock()->getNumSuccessors() != op->getNumResults())
        return op->emitOpError("incorrect successor count");
      bool found = false;
      for (int i = 0, e = op->getBlock()->getNumSuccessors(); i < e; ++i) {
        Block *succ = op->getBlock()->getSuccessor(i);
        if (succOp->getBlock() == succ || isa<SinkOp>(succOp))
          found = true;
      }
      if (!found)
        return op->emitOpError("branch successor in incorrect block");
    }
  }
  return success();
}

// Checks if merge predecessors are in appropriate block
LogicalResult checkMergeOps(MergeLikeOpInterface mergeOp,
                            bool disableTaskPipelining) {
  Block *block = mergeOp->getBlock();
  unsigned operand_count = mergeOp.dataOperands().size();

  // Merges in entry block have single predecessor (argument)
  if (block->isEntryBlock()) {
    if (operand_count != 1)
      return mergeOp->emitOpError(
                 "merge operations in entry block must have a ")
             << "single predecessor";
  } else {
    if (operand_count > getBlockPredecessorCount(block))
      return mergeOp->emitOpError("merge operation has ")
             << operand_count << " data inputs, but only "
             << getBlockPredecessorCount(block) << " predecessor blocks";
  }

  if (disableTaskPipelining) {
    // There must be a predecessor from each predecessor block
    for (auto *predBlock : block->getPredecessors()) {
      bool found = false;
      for (auto operand : mergeOp.dataOperands()) {
        auto *operandOp = operand.getDefiningOp();
        if (operandOp->getBlock() == predBlock) {
          found = true;
          break;
        }
      }
      if (!found)
        return mergeOp->emitOpError(
            "missing predecessor from predecessor block");
    }
  }

  // Select operand must come from same block
  if (auto muxOp = dyn_cast<MuxOp>(mergeOp.getOperation())) {
    auto *operand = muxOp.selectOperand().getDefiningOp();
    if (operand->getBlock() != block)
      return mergeOp->emitOpError("mux select operand must be from same block");
  }
  return success();
}

LogicalResult checkDataflowConversion(handshake::FuncOp f,
                                      bool disableTaskPipelining) {
  for (Operation &op : f.getOps()) {
    if (isa<mlir::CondBranchOp, mlir::BranchOp, memref::LoadOp,
            arith::ConstantOp, mlir::AffineReadOpInterface, mlir::AffineForOp>(
            op))
      continue;

    if (op.getNumResults() > 0) {
      for (auto result : op.getResults()) {
        if (checkUseCount(&op, result).failed() ||
            checkSuccessorBlocks(&op, result).failed())
          return failure();
      }
    }
    if (auto mergeOp = dyn_cast<MergeLikeOpInterface>(op); mergeOp)
      if (checkMergeOps(mergeOp, disableTaskPipelining).failed())
        return failure();
  }
  return success();
}

Value getBlockControlValue(Block *block) {
  // Get control-only value sent to the block terminator
  for (Operation &op : *block) {
    if (auto BranchOp = dyn_cast<handshake::BranchOp>(op))
      if (BranchOp.isControl())
        return BranchOp.dataOperand();
    if (auto BranchOp = dyn_cast<handshake::ConditionalBranchOp>(op))
      if (BranchOp.isControl())
        return BranchOp.dataOperand();
    if (auto endOp = dyn_cast<handshake::ReturnOp>(op))
      return endOp.control();
  }
  return nullptr;
}

LogicalResult getOpMemRef(Operation *op, Value &out) {
  out = Value();
  if (auto memOp = dyn_cast<memref::LoadOp>(op))
    out = memOp.getMemRef();
  else if (auto memOp = dyn_cast<memref::StoreOp>(op))
    out = memOp.getMemRef();
  else if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
    MemRefAccess access(op);
    out = access.memref;
  }
  if (out != Value())
    return success();
  return op->emitOpError("Unknown Op type");
}

bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, mlir::AffineReadOpInterface,
             mlir::AffineWriteOpInterface>(op);
}

LogicalResult
FuncOpLowering::replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemRefToMemoryAccessOp &memRefOps) {

  std::vector<Operation *> opsToErase;

  // Replace load and store ops with the corresponding handshake ops
  // Need to traverse ops in blocks to store them in memRefOps in program order
  for (Operation &op : f.getOps()) {
    if (!isMemoryOp(&op))
      continue;

    rewriter.setInsertionPoint(&op);
    Value memref;
    if (getOpMemRef(&op, memref).failed())
      return failure();
    Operation *newOp = nullptr;

    llvm::TypeSwitch<Operation *>(&op)
        .Case<memref::LoadOp>([&](auto loadOp) {
          // Get operands which correspond to address indices
          // This will add all operands except alloc
          SmallVector<Value, 8> operands(loadOp.getIndices());

          newOp =
              rewriter.create<handshake::LoadOp>(op.getLoc(), memref, operands);
          op.getResult(0).replaceAllUsesWith(newOp->getResult(0));
        })
        .Case<memref::StoreOp>([&](auto storeOp) {
          // Get operands which correspond to address indices
          // This will add all operands except alloc and data
          SmallVector<Value, 8> operands(storeOp.getIndices());

          // Create new op where operands are store data and address indices
          newOp = rewriter.create<handshake::StoreOp>(
              op.getLoc(), storeOp.getValueToStore(), operands);
        })
        .Case<mlir::AffineReadOpInterface,
              mlir::AffineWriteOpInterface>([&](auto) {
          // Get essential memref access inforamtion.
          MemRefAccess access(&op);
          // The address of an affine load/store operation can be a result of
          // an affine map, which is a linear combination of constants and
          // parameters. Therefore, we should extract the affine map of each
          // address and expand it into proper expressions that calculate the
          // result.
          AffineMap map;
          if (auto loadOp = dyn_cast<mlir::AffineReadOpInterface>(op))
            map = loadOp.getAffineMap();
          else
            map = dyn_cast<AffineWriteOpInterface>(op).getAffineMap();

          // The returned object from expandAffineMap is an optional list of
          // the expansion results from the given affine map, which are the
          // actual address indices that can be used as operands for handshake
          // LoadOp/StoreOp. The following processing requires it to be a
          // valid result.
          auto operands =
              expandAffineMap(rewriter, op.getLoc(), map, access.indices);
          assert(operands &&
                 "Address operands of affine memref access cannot be reduced.");

          if (isa<mlir::AffineReadOpInterface>(op)) {
            auto loadOp = rewriter.create<handshake::LoadOp>(
                op.getLoc(), access.memref, *operands);
            newOp = loadOp;
            op.getResult(0).replaceAllUsesWith(loadOp.dataResult());
          } else {
            newOp = rewriter.create<handshake::StoreOp>(
                op.getLoc(), op.getOperand(0), *operands);
          }
        })
        .Default([&](auto) {
          op.emitOpError("Load/store operation cannot be handled.");
        });

    memRefOps[memref].push_back(newOp);
    opsToErase.push_back(&op);
  }

  // Erase old memory ops
  for (unsigned i = 0, e = opsToErase.size(); i != e; ++i) {
    auto *op = opsToErase[i];
    for (int j = 0, e = op->getNumOperands(); j < e; ++j)
      op->eraseOperand(0);
    assert(op->getNumOperands() == 0);

    rewriter.eraseOp(op);
  }

  return success();
}

std::vector<Block *> getOperationBlocks(ArrayRef<Operation *> ops) {
  // Get list of (unique) blocks which ops belong to
  // Used to connect control network to memory
  std::vector<Block *> blocks;

  for (Operation *op : ops) {
    Block *b = op->getBlock();
    if (std::find(blocks.begin(), blocks.end(), b) == blocks.end())
      blocks.push_back(b);
  }
  return blocks;
}

SmallVector<Value, 8> getResultsToMemory(Operation *op) {
  // Get load/store results which are given as inputs to MemoryOp

  if (handshake::LoadOp loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For load, get all address outputs/indices
    // (load also has one data output which goes to successor operation)
    SmallVector<Value, 8> results(loadOp.addressResults());
    return results;

  } else {
    // For store, all outputs (data and address indices) go to memory
    assert(dyn_cast<handshake::StoreOp>(op));
    handshake::StoreOp storeOp = dyn_cast<handshake::StoreOp>(op);
    SmallVector<Value, 8> results(storeOp.getResults());
    return results;
  }
}

void addLazyForks(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {

  for (Block &block : f) {
    Value res = getBlockControlValue(&block);
    if (!res.hasOneUse())
      insertFork(res, true, rewriter);
  }
}

void addMemOpForks(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {

  for (Block &block : f) {
    for (Operation &op : block) {
      if (isa<MemoryOp, ExternalMemoryOp, StartOp, ControlMergeOp>(op)) {
        for (auto result : op.getResults()) {
          // If there is a result and it is used more than once
          if (!result.use_empty() && !result.hasOneUse())
            insertFork(result, false, rewriter);
        }
      }
    }
  }
}

void removeAllocOps(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {
  std::vector<Operation *> opsToDelete;

  /// TODO(mortbopet): use explicit template parameter when moving to C++20
  auto remover = [&](auto allocType) {
    for (auto allocOp : f.getOps<decltype(allocType)>()) {
      assert(allocOp.getResult().hasOneUse());
      for (auto &u : allocOp.getResult().getUses()) {
        Operation *useOp = u.getOwner();
        if (isa<SinkOp>(useOp))
          opsToDelete.push_back(allocOp);
      }
    }
  };

  remover(memref::AllocOp());
  remover(memref::AllocaOp());
  llvm::for_each(opsToDelete, [&](auto allocOp) { rewriter.eraseOp(allocOp); });
}

void removeRedundantSinks(handshake::FuncOp f,
                          ConversionPatternRewriter &rewriter) {
  std::vector<Operation *> redundantSinks;

  for (Block &block : f)
    for (Operation &op : block) {
      if (!isa<SinkOp>(op))
        continue;

      if (!op.getOperand(0).hasOneUse() ||
          isa<memref::AllocOp, memref::AllocaOp>(
              op.getOperand(0).getDefiningOp()))
        redundantSinks.push_back(&op);
    }
  for (unsigned i = 0, e = redundantSinks.size(); i != e; ++i) {
    auto *op = redundantSinks[i];
    rewriter.eraseOp(op);
    // op->erase();
  }
}

void addJoinOps(ConversionPatternRewriter &rewriter,
                ArrayRef<Value> controlVals) {
  for (auto val : controlVals) {
    assert(val.hasOneUse());
    auto srcOp = val.getDefiningOp();

    // Insert only single join per block
    if (!isa<JoinOp>(srcOp)) {
      rewriter.setInsertionPointAfter(srcOp);
      Operation *newOp = rewriter.create<JoinOp>(srcOp->getLoc(), val);
      for (auto &u : val.getUses())
        if (u.getOwner() != newOp)
          u.getOwner()->replaceUsesOfWith(val, newOp->getResult(0));
    }
  }
}

std::vector<Value> getControlValues(ArrayRef<Operation *> memOps) {
  std::vector<Value> vals;

  for (Operation *op : memOps) {
    // Get block from which the mem op originates
    Block *block = op->getBlock();
    // Add control signal from each block
    // Use result which goes to the branch
    Value res = getBlockControlValue(block);
    assert(res != nullptr);
    if (std::find(vals.begin(), vals.end(), res) == vals.end())
      vals.push_back(res);
  }
  return vals;
}

void addValueToOperands(Operation *op, Value val) {

  SmallVector<Value, 8> results(op->getOperands());
  results.push_back(val);
  op->setOperands(results);
}

void setLoadDataInputs(ArrayRef<Operation *> memOps, Operation *memOp) {
  // Set memory outputs as load input data
  int ld_count = 0;
  for (auto *op : memOps) {
    if (isa<handshake::LoadOp>(op))
      addValueToOperands(op, memOp->getResult(ld_count++));
  }
}

LogicalResult setJoinControlInputs(ArrayRef<Operation *> memOps,
                                   Operation *memOp, int offset,
                                   ArrayRef<int> cntrlInd) {
  // Connect all memory ops to the join of that block (ensures that all mem
  // ops terminate before a new block starts)
  for (int i = 0, e = memOps.size(); i < e; ++i) {
    auto *op = memOps[i];
    Value val = getBlockControlValue(op->getBlock());
    auto srcOp = val.getDefiningOp();
    if (!isa<JoinOp, StartOp>(srcOp)) {
      return srcOp->emitOpError("Op expected to be a JoinOp or StartOp");
    }
    addValueToOperands(srcOp, memOp->getResult(offset + cntrlInd[i]));
  }
  return success();
}

void FuncOpLowering::setMemOpControlInputs(ConversionPatternRewriter &rewriter,
                                           ArrayRef<Operation *> memOps,
                                           Operation *memOp, int offset,
                                           ArrayRef<int> cntrlInd) {
  for (int i = 0, e = memOps.size(); i < e; ++i) {
    std::vector<Value> controlOperands;
    Operation *currOp = memOps[i];
    Block *currBlock = currOp->getBlock();

    // Set load/store control inputs from the block input control value
    Value blockEntryCtrl = getBlockEntryControl(currBlock);
    controlOperands.push_back(blockEntryCtrl);

    // Set load/store control inputs from predecessors in block
    for (int j = 0, f = i; j < f; ++j) {
      Operation *predOp = memOps[j];
      Block *predBlock = predOp->getBlock();
      if (currBlock == predBlock)
        // Any dependency but RARs
        if (!(isa<handshake::LoadOp>(currOp) && isa<handshake::LoadOp>(predOp)))
          // cntrlInd maps memOps index to correct control output index
          controlOperands.push_back(memOp->getResult(offset + cntrlInd[j]));
    }

    // If there is only one control input, add directly to memory op
    if (controlOperands.size() == 1)
      addValueToOperands(currOp, controlOperands[0]);

    // If multiple, join them and connect join output to memory op
    else {
      rewriter.setInsertionPoint(currOp);
      Operation *joinOp =
          rewriter.create<JoinOp>(currOp->getLoc(), controlOperands);
      addValueToOperands(currOp, joinOp->getResult(0));
    }
  }
}

LogicalResult
FuncOpLowering::connectToMemory(ConversionPatternRewriter &rewriter,
                                MemRefToMemoryAccessOp memRefOps, bool lsq) {
  // Add MemoryOps which represent the memory interface
  // Connect memory operations and control appropriately
  int mem_count = 0;
  for (auto memory : memRefOps) {
    // First operand corresponds to memref (alloca or function argument)
    Value memrefOperand = memory.first;

    // A memory is external if the memref that defines it is provided as a
    // function (block) argument.
    bool isExternalMemory = memrefOperand.isa<BlockArgument>();

    mlir::MemRefType memrefType =
        memrefOperand.getType().cast<mlir::MemRefType>();
    if (memrefType.getNumDynamicDims() != 0 ||
        memrefType.getShape().size() != 1)
      return emitError(memrefOperand.getLoc())
             << "memref's must be both statically sized and unidimensional.";

    std::vector<Value> operands;

    // Get control values which need to connect to memory
    std::vector<Value> controlVals = getControlValues(memory.second);

    // In case of LSQ interface, set control values as inputs (used to trigger
    // allocation to LSQ)
    if (lsq)
      operands.insert(operands.end(), controlVals.begin(), controlVals.end());

    // Add load indices and store data+indices to memory operands
    // Count number of loads so that we can generate appropriate number of
    // memory outputs (data to load ops)

    // memory.second is in program order
    // Enforce MemoryOp port ordering as follows:
    // Operands: all stores then all loads (stdata1, staddr1, stdata2,...,
    // ldaddr1, ldaddr2,....) Outputs: all load outputs, ordered the same as
    // load addresses (lddata1, lddata2, ...), followed by all none outputs,
    // ordered as operands (stnone1, stnone2,...ldnone1, ldnone2,...)
    std::vector<int> newInd(memory.second.size(), 0);
    int ind = 0;
    for (int i = 0, e = memory.second.size(); i < e; ++i) {
      auto *op = memory.second[i];
      if (isa<handshake::StoreOp>(op)) {
        SmallVector<Value, 8> results = getResultsToMemory(op);
        operands.insert(operands.end(), results.begin(), results.end());
        newInd[i] = ind++;
      }
    }

    int ld_count = 0;

    for (int i = 0, e = memory.second.size(); i < e; ++i) {
      auto *op = memory.second[i];
      if (isa<handshake::LoadOp>(op)) {
        SmallVector<Value, 8> results = getResultsToMemory(op);
        operands.insert(operands.end(), results.begin(), results.end());

        ld_count++;
        newInd[i] = ind++;
      }
    }

    // control-only outputs for each access (indicate access completion)
    int cntrl_count = lsq ? 0 : memory.second.size();

    Block *entryBlock = &f.front();
    rewriter.setInsertionPointToStart(entryBlock);

    // Place memory op next to the alloc op
    Operation *newOp = nullptr;
    if (isExternalMemory)
      newOp = rewriter.create<ExternalMemoryOp>(
          entryBlock->front().getLoc(), memrefOperand, operands, ld_count,
          cntrl_count - ld_count, mem_count++);
    else
      newOp = rewriter.create<MemoryOp>(entryBlock->front().getLoc(), operands,
                                        ld_count, cntrl_count, lsq, mem_count++,
                                        memrefOperand);

    setLoadDataInputs(memory.second, newOp);

    if (!lsq) {
      // Create Joins which join done signals from memory with the
      // control-only network
      addJoinOps(rewriter, controlVals);

      // Connect all load/store done signals to the join of their block
      // Ensure that the block terminates only after all its accesses have
      // completed
      // True is default. When no sync needed, set to false (for now,
      // user-determined)
      bool control = true;

      if (control) {
        if (setJoinControlInputs(memory.second, newOp, ld_count, newInd)
                .failed())
          return failure();

      } else {
        for (int i = 0, e = cntrl_count; i < e; ++i) {
          rewriter.setInsertionPointAfter(newOp);
          rewriter.create<SinkOp>(newOp->getLoc(),
                                  newOp->getResult(ld_count + i));
        }
      }

      // Set control-only inputs to each memory op
      // Ensure that op starts only after prior blocks have completed
      // Ensure that op starts only after predecessor ops (with RAW, WAR, or
      // WAW) have completed
      setMemOpControlInputs(rewriter, memory.second, newOp, ld_count, newInd);
    }
  }

  if (lsq)
    addLazyForks(f, rewriter);
  else
    addMemOpForks(f, rewriter);

  removeAllocOps(f, rewriter);

  // Loads and stores have some sinks which are no longer needed now that they
  // connect to MemoryOp
  removeRedundantSinks(f, rewriter);
  return success();
}

// A handshake::StartOp should have been created in the first block of the
// enclosing function region
static Operation *findStartOp(Region *region) {
  for (Operation &op : region->front())
    if (isa<handshake::StartOp>(op))
      return &op;

  llvm::report_fatal_error("No handshake::StartOp in first block");
}

struct HandshakeCanonicalizePattern : public ConversionPattern {
  using ConversionPattern::ConversionPattern;
  LogicalResult match(Operation *op) const override {
    // Ignore forks, ops without results, and branches (ops with succ blocks)
    op->emitWarning("checking...");
    return success();
    if (op->getNumSuccessors() == 0 && op->getNumResults() > 0 &&
        !isa<ForkOp>(op))
      return success();
    else
      return failure();
  }

  void rewrite(Operation *op, ArrayRef<Value> /*operands*/,
               ConversionPatternRewriter &rewriter) const override {
    op->emitWarning("skipping...");
    if (op->getNumSuccessors() == 0 && op->getNumResults() > 0 &&
        !isa<ForkOp>(op)) {
      op->emitWarning("skipping...");
    }
    for (auto result : op->getResults()) {
      // If there is a result and it is used more than once
      if (!result.use_empty() && !result.hasOneUse())
        insertFork(result, false, rewriter);
    }
  }
};

LogicalResult
FuncOpLowering::replaceCallOps(ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    /// An instance is activated whenever control arrives at the basic block of
    /// the source callOp.
    Value blockEntryControl = getBlockEntryControl(&block);
    for (Operation &op : block) {
      if (auto callOp = dyn_cast<CallOp>(op)) {
        llvm::SmallVector<Value> operands;
        llvm::copy(callOp.getOperands(), std::back_inserter(operands));
        operands.push_back(blockEntryControl);
        rewriter.setInsertionPoint(callOp);
        auto instanceOp = rewriter.create<handshake::InstanceOp>(
            callOp.getLoc(), callOp.getCallee(), callOp.getResultTypes(),
            operands);
        // Replace all results of the source callOp.
        for (auto it : llvm::zip(callOp.getResults(), instanceOp.getResults()))
          std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
        rewriter.eraseOp(callOp);
      }
    }
  }
  return success();
}

LogicalResult FuncOpLowering::finalize(ConversionPatternRewriter &rewriter,
                                       TypeRange argTypes, TypeRange resTypes,
                                       mlir::FuncOp origFunc) {
  SmallVector<Type> newArgTypes(argTypes);
  newArgTypes.push_back(rewriter.getNoneType());

  auto funcType = rewriter.getFunctionType(newArgTypes, resTypes);
  f.setType(funcType);
  auto ctrlArg = f.front().addArgument(rewriter.getNoneType());

  // We've now added all types to the handshake.funcOp; resolve arg- and
  // res names to ensure they are up to date with the final type
  // signature.
  f.resolveArgAndResNames();

  Operation *startOp = findStartOp(&f.getRegion());
  startOp->getResult(0).replaceAllUsesWith(ctrlArg);
  rewriter.eraseOp(startOp);
  rewriter.eraseOp(origFunc);
  return success();
}

LogicalResult lowerFuncOp(mlir::FuncOp funcOp, MLIRContext *ctx,
                          bool sourceConstants, bool disableTaskPipelining) {
  // Only retain those attributes that are not constructed by build.
  SmallVector<NamedAttribute, 4> attributes;
  for (const auto &attr : funcOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == function_like_impl::getTypeAttrName())
      continue;
    attributes.push_back(attr);
  }

  // Get function arguments
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto &arg : funcOp.getArguments()) {
    mlir::Type type = arg.getType();
    argTypes.push_back(type);
  }

  // Get function results
  llvm::SmallVector<mlir::Type, 8> resTypes;
  for (auto arg : funcOp.getType().getResults())
    resTypes.push_back(arg);

  handshake::FuncOp newFuncOp;

  // Add control input/output to function arguments/results and create a
  // handshake::FuncOp of appropriate type
  returnOnError(partiallyLowerFuncOp<mlir::FuncOp>(
      [&](mlir::FuncOp funcOp, PatternRewriter &rewriter) {
        auto noneType = rewriter.getNoneType();
        resTypes.push_back(noneType);
        auto func_type = rewriter.getFunctionType(argTypes, resTypes);
        newFuncOp = rewriter.create<handshake::FuncOp>(
            funcOp.getLoc(), funcOp.getName(), func_type, attributes);
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                    newFuncOp.end());
        return success();
      },
      ctx, funcOp));

  FuncOpLowering fol(newFuncOp);

  // Perform initial dataflow conversion. This process allows for the use of
  // non-deterministic merge-like operations.
  MemRefToMemoryAccessOp memOps;
  returnOnError(
      fol.runPartialLowering(&FuncOpLowering::replaceMemoryOps, memOps));
  returnOnError(fol.runPartialLowering(&FuncOpLowering::setControlOnlyPath));
  returnOnError(fol.runPartialLowering(&FuncOpLowering::addMergeOps));
  returnOnError(fol.runPartialLowering(&FuncOpLowering::replaceCallOps));
  returnOnError(fol.runPartialLowering(&FuncOpLowering::replaceCallOps));
  returnOnError(fol.runPartialLowering(&FuncOpLowering::addBranchOps));

  // The following passes modifies the dataflow graph to being safe for task
  // pipelining. In doing so, non-deterministic merge structures are replaced
  // for deterministic structures.
  if (!disableTaskPipelining) {
    returnOnError(
        fol.runPartialLowering(&FuncOpLowering::loopNetworkRewriting));
  }

  // Fork/sink materialization. @todo: this should be removed and
  // materialization should be run as a separate pass afterward initial dataflow
  // conversion! However, connectToMemory has some hard-coded assumptions on the
  // existence of fork/sink operations...
  returnOnError(
      partiallyLowerFuncOp<handshake::FuncOp>(addSinkOps, ctx, newFuncOp));
  returnOnError(fol.runPartialLowering(
      &FuncOpLowering::connectConstantsToControl, sourceConstants));
  returnOnError(
      partiallyLowerFuncOp<handshake::FuncOp>(addForkOps, ctx, newFuncOp));
  returnOnError(checkDataflowConversion(newFuncOp, disableTaskPipelining));

  bool lsq = false;
  returnOnError(
      fol.runPartialLowering(&FuncOpLowering::connectToMemory, memOps, lsq));

  // Add  control argument to entry block, replace references to the
  // temporary handshake::StartOp operation, and finally remove the start
  // op.
  returnOnError(fol.runPartialLowering(&FuncOpLowering::finalize, argTypes,
                                       resTypes, funcOp));
  return success();
}

namespace {

struct ConvertSelectOps : public OpConversionPattern<mlir::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<handshake::SelectOp>(op, adaptor.getCondition(),
                                                     adaptor.getFalseValue(),
                                                     adaptor.getTrueValue());
    return success();
  };
};

struct HandshakeRemoveBlockPass
    : HandshakeRemoveBlockBase<HandshakeRemoveBlockPass> {
  void runOnOperation() override { removeBasicBlocks(getOperation()); }
};

struct StandardToHandshakePass
    : public StandardToHandshakeBase<StandardToHandshakePass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : llvm::make_early_inc_range(m.getOps<mlir::FuncOp>())) {
      if (failed(lowerFuncOp(funcOp, &getContext(), sourceConstants,
                             disableTaskPipelining))) {
        signalPassFailure();
        return;
      }
    }

    // Legalize the resulting regions, removing basic blocks and performing
    // any simple conversions.
    for (auto func : m.getOps<handshake::FuncOp>()) {
      removeBasicBlocks(func);
      if (failed(postDataflowConvert(func)))
        return signalPassFailure();
    }
  }

  LogicalResult postDataflowConvert(handshake::FuncOp op) {
    ConversionTarget target(getContext());
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalOp<mlir::SelectOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertSelectOps>(&getContext());
    return applyPartialConversion(op, target, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::createStandardToHandshakePass() {
  return std::make_unique<StandardToHandshakePass>();
}

std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
circt::createHandshakeRemoveBlockPass() {
  return std::make_unique<HandshakeRemoveBlockPass>();
}
