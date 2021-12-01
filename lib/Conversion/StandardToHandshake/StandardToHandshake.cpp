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
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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

#include <map>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace std;

typedef DenseMap<Block *, std::vector<Value>> BlockValues;
typedef DenseMap<Block *, std::vector<Operation *>> BlockOps;
typedef DenseMap<Value, Operation *> blockArgPairs;

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

LogicalResult setControlOnlyPath(handshake::FuncOp f,
                                 ConversionPatternRewriter &rewriter) {
  // Creates start and end points of the control-only path

  // Temporary start node (removed in later steps) in entry block
  Block *entryBlock = &f.front();
  rewriter.setInsertionPointToStart(entryBlock);
  Operation *startOp = rewriter.create<StartOp>(entryBlock->front().getLoc());

  // Replace original return ops with new returns with additional control input
  for (Block &block : f) {
    Operation *termOp = block.getTerminator();
    if (!isa<mlir::ReturnOp>(termOp))
      continue;

    rewriter.setInsertionPoint(termOp);

    // Remove operands from old return op and add them to new op
    SmallVector<Value, 8> operands(termOp->getOperands());
    for (int i = 0, e = termOp->getNumOperands(); i < e; ++i)
      termOp->eraseOperand(0);
    assert(termOp->getNumOperands() == 0);
    operands.push_back(startOp->getResult(0));
    rewriter.replaceOpWithNewOp<handshake::ReturnOp>(termOp, operands);
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
Operation *insertMerge(Block *block, Value val,
                       ConversionPatternRewriter &rewriter) {
  unsigned numPredecessors = getBlockPredecessorCount(block);

  // Control-only path originates from StartOp
  if (!val.isa<BlockArgument>()) {
    if (isa<StartOp>(val.getDefiningOp())) {
      return rewriter.create<handshake::ControlMergeOp>(block->front().getLoc(),
                                                        val, numPredecessors);
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
BlockOps insertMergeOps(handshake::FuncOp f, BlockValues blockLiveIns,
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

Operation *getStartOp(Block *block) { return getFirstOp<StartOp>(block); }

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

LogicalResult addMergeOps(handshake::FuncOp f,
                          ConversionPatternRewriter &rewriter) {

  blockArgPairs mergePairs;

  // blockLiveIns: live in variables of block
  BlockValues liveIns = livenessAnalysis(f);

  // Insert merge operations
  BlockOps mergeOps = insertMergeOps(f, liveIns, mergePairs, rewriter);

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

LogicalResult addBranchOps(handshake::FuncOp f,
                           ConversionPatternRewriter &rewriter) {

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

LogicalResult connectConstantsToControl(handshake::FuncOp f,
                                        ConversionPatternRewriter &rewriter) {
  // Create new constants which have a control-only input to trigger them
  // Connect input to ControlMerge (trigger const when its block is entered)

  for (Block &block : f) {
    Operation *cntrlMg =
        block.isEntryBlock() ? getStartOp(&block) : getControlMerge(&block);
    assert(cntrlMg != nullptr);
    std::vector<Operation *> cstOps;
    for (Operation &op : block) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        rewriter.setInsertionPointAfter(&op);
        Operation *newOp = rewriter.create<handshake::ConstantOp>(
            op.getLoc(), constantOp.value(), cntrlMg->getResult(0));

        op.getResult(0).replaceAllUsesWith(newOp->getResult(0));
        cstOps.push_back(&op);
      }
    }

    // Erase StandardOp constants
    for (unsigned i = 0, e = cstOps.size(); i != e; ++i) {
      auto *op = cstOps[i];
      for (int j = 0, e = op->getNumOperands(); j < e; ++j)
        op->eraseOperand(0);
      assert(op->getNumOperands() == 0);
      rewriter.eraseOp(op);
    }
  }
  return success();
}

void checkUseCount(Operation *op, Value res) {
  // Checks if every result has single use
  if (!res.hasOneUse()) {
    int i = 0;
    for (auto *user : res.getUsers()) {
      user->emitWarning("user here");
      i++;
    }
    op->emitError("every result must have exactly one user, but had ") << i;
  }
  return;
}

// Checks if op successors are in appropriate blocks
void checkSuccessorBlocks(Operation *op, Value res) {
  for (auto &u : res.getUses()) {
    Operation *succOp = u.getOwner();
    // Non-branch ops: succesors must be in same block
    if (!(isa<handshake::ConditionalBranchOp>(op) ||
          isa<handshake::BranchOp>(op))) {
      if (op->getBlock() != succOp->getBlock())
        op->emitError("cannot be block live-out");
    } else {
      // Branch ops: must have successor per successor block
      if (op->getBlock()->getNumSuccessors() != op->getNumResults())
        op->emitError("incorrect successor count");
      bool found = false;
      for (int i = 0, e = op->getBlock()->getNumSuccessors(); i < e; ++i) {
        Block *succ = op->getBlock()->getSuccessor(i);
        if (succOp->getBlock() == succ || isa<SinkOp>(succOp))
          found = true;
      }
      if (!found)
        op->emitError("branch successor in incorrect block");
    }
  }
  return;
}

// Checks if merge predecessors are in appropriate block
void checkMergePredecessors(MergeLikeOpInterface mergeOp) {
  Block *block = mergeOp->getBlock();
  unsigned operand_count = mergeOp.dataOperands().size();

  // Merges in entry block have single predecessor (argument)
  if (block->isEntryBlock()) {
    if (operand_count != 1)
      mergeOp->emitError("merge operations in entry block must have a ")
          << "single predecessor";
  } else {
    if (operand_count > getBlockPredecessorCount(block))
      mergeOp->emitError("merge operation has ")
          << operand_count << " data inputs, but only "
          << getBlockPredecessorCount(block) << " predecessor blocks";
  }

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
      mergeOp->emitError("missing predecessor from predecessor block");
  }

  // Select operand must come from same block
  if (auto muxOp = dyn_cast<MuxOp>(mergeOp.getOperation())) {
    auto *operand = muxOp.selectOperand().getDefiningOp();
    if (operand->getBlock() != block)
      mergeOp->emitError("mux select operand must be from same block");
  }
  return;
}

void checkDataflowConversion(handshake::FuncOp f) {
  for (Operation &op : f.getOps()) {
    if (isa<mlir::CondBranchOp, mlir::BranchOp, memref::LoadOp,
            arith::ConstantOp, mlir::AffineReadOpInterface, mlir::AffineForOp>(
            op))
      continue;

    if (op.getNumResults() > 0) {
      for (auto result : op.getResults()) {
        checkUseCount(&op, result);
        checkSuccessorBlocks(&op, result);
      }
    }
    if (auto mergeOp = dyn_cast<MergeLikeOpInterface>(op); mergeOp)
      checkMergePredecessors(mergeOp);
  }
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

Value getOpMemRef(Operation *op) {
  if (auto memOp = dyn_cast<memref::LoadOp>(op))
    return memOp.getMemRef();
  if (auto memOp = dyn_cast<memref::StoreOp>(op))
    return memOp.getMemRef();
  if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
    MemRefAccess access(op);
    return access.memref;
  }
  op->emitError("Unknown Op type");
  return Value();
}

bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, mlir::AffineReadOpInterface,
             mlir::AffineWriteOpInterface>(op);
}

typedef llvm::MapVector<Value, std::vector<Operation *>> MemRefToMemoryAccessOp;

// Replaces standard memory ops with their handshake version (i.e.,
// ops which connect to memory/LSQ). Returns a map with an ordered
// list of new ops corresponding to each memref. Later, we instantiate
// a memory node for each memref and connect it to its load/store ops
MemRefToMemoryAccessOp replaceMemoryOps(handshake::FuncOp f,
                                        ConversionPatternRewriter &rewriter) {
  // Map from original memref to new load/store operations.
  MemRefToMemoryAccessOp MemRefOps;

  std::vector<Operation *> opsToErase;

  // Replace load and store ops with the corresponding handshake ops
  // Need to traverse ops in blocks to store them in MemRefOps in program order
  for (Operation &op : f.getOps()) {
    if (!isMemoryOp(&op))
      continue;

    rewriter.setInsertionPoint(&op);
    Value memref = getOpMemRef(&op);
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
          op.emitError("Load/store operation cannot be handled.");
        });

    MemRefOps[memref].push_back(newOp);
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

  return MemRefOps;
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

void setJoinControlInputs(ArrayRef<Operation *> memOps, Operation *memOp,
                          int offset, ArrayRef<int> cntrlInd) {
  // Connect all memory ops to the join of that block (ensures that all mem
  // ops terminate before a new block starts)
  for (int i = 0, e = memOps.size(); i < e; ++i) {
    auto *op = memOps[i];
    Value val = getBlockControlValue(op->getBlock());
    auto srcOp = val.getDefiningOp();
    if (!isa<JoinOp, StartOp>(srcOp)) {
      srcOp->emitError("Op expected to be a JoinOp or StartOp");
    }
    addValueToOperands(srcOp, memOp->getResult(offset + cntrlInd[i]));
  }
}

void setMemOpControlInputs(ConversionPatternRewriter &rewriter,
                           ArrayRef<Operation *> memOps, Operation *memOp,
                           int offset, ArrayRef<int> cntrlInd) {
  for (int i = 0, e = memOps.size(); i < e; ++i) {
    std::vector<Value> controlOperands;
    Operation *currOp = memOps[i];
    Block *currBlock = currOp->getBlock();

    // Set load/store control inputs from control merge
    Operation *cntrlMg = currBlock->isEntryBlock() ? getStartOp(currBlock)
                                                   : getControlMerge(currBlock);
    controlOperands.push_back(cntrlMg->getResult(0));

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

LogicalResult connectToMemory(handshake::FuncOp f,
                              MemRefToMemoryAccessOp memRefOps, bool lsq,
                              ConversionPatternRewriter &rewriter) {
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

      if (control)
        setJoinControlInputs(memory.second, newOp, ld_count, newInd);
      else {
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

template <typename TFuncOp>
class LowerFuncOpTarget : public ConversionTarget {
public:
  explicit LowerFuncOpTarget(MLIRContext &context) : ConversionTarget(context) {
    loweredFuncs.clear();
    addLegalDialect<HandshakeDialect>();
    addLegalDialect<StandardOpsDialect>();
    addLegalDialect<arith::ArithmeticDialect>();
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
        m_target(target), loweringRes(loweringResRef), m_fun(fun) {}
  using ConversionPattern::ConversionPattern;
  LogicalResult
  matchAndRewrite(Operation *funcOp, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<TFuncOp>(funcOp));
    rewriter.updateRootInPlace(funcOp, [&] {
      loweringRes = m_fun(dyn_cast<TFuncOp>(funcOp), rewriter);
    });
    m_target.loweredFuncs[funcOp] = true;
    return loweringRes;
  };

private:
  LowerFuncOpTarget<TFuncOp> &m_target;
  LogicalResult &loweringRes;
  PartialLoweringFunc m_fun;
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

/// Rewrite affine.for operations in a handshake.func into its representations
/// as a CFG in the standard dialect. Affine expressions in loop bounds will be
/// expanded to code in the standard dialect that actually computes them. We
/// combine the lowering of affine loops in the following two conversions:
/// [AffineToStandard](https://mlir.llvm.org/doxygen/AffineToStandard_8cpp.html),
/// [SCFToStandard](https://mlir.llvm.org/doxygen/SCFToStandard_8cpp_source.html)
/// into this function.
/// The affine memory operations will be preserved until other rewrite
/// functions, e.g.,`replaceMemoryOps`, are called. Any affine analysis, e.g.,
/// getting dependence information, should be carried out before calling this
/// function; otherwise, the affine for loops will be destructed and key
/// information will be missing.
LogicalResult rewriteAffineFor(handshake::FuncOp f,
                               ConversionPatternRewriter &rewriter) {
  // Get all affine.for operations in the function body.
  SmallVector<mlir::AffineForOp, 8> forOps;
  f.walk([&](mlir::AffineForOp op) { forOps.push_back(op); });

  // TODO: how to deal with nested loops?
  for (unsigned i = 0, e = forOps.size(); i < e; i++) {
    auto forOp = forOps[i];

    // Insert lower and upper bounds right at the position of the original
    // affine.for operation.
    rewriter.setInsertionPoint(forOp);
    auto loc = forOp.getLoc();
    auto lowerBound = expandAffineMap(rewriter, loc, forOp.getLowerBoundMap(),
                                      forOp.getLowerBoundOperands());
    auto upperBound = expandAffineMap(rewriter, loc, forOp.getUpperBoundMap(),
                                      forOp.getUpperBoundOperands());
    if (!lowerBound || !upperBound)
      return failure();
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, forOp.getStep());

    // Build blocks for a common for loop. initBlock and initPosition are the
    // block that contains the current forOp, and the position of the forOp.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();

    // Split the current block into several parts. `endBlock` contains the code
    // starting from forOp. `conditionBlock` will have the condition branch.
    // `firstBodyBlock` is the loop body, and `lastBodyBlock` is about the loop
    // iterator stepping. Here we move the body region of the AffineForOp here
    // and split it into `conditionBlock`, `firstBodyBlock`, and
    // `lastBodyBlock`.
    // TODO: is there a simpler API for doing so?
    auto *endBlock = rewriter.splitBlock(initBlock, initPosition);
    // Split and get the references to different parts (blocks) of the original
    // loop body.
    auto *conditionBlock = &forOp.region().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock =
        rewriter.splitBlock(firstBodyBlock, firstBodyBlock->end());
    rewriter.inlineRegionBefore(forOp.region(), endBlock);

    // The loop IV is the first argument of the conditionBlock.
    auto iv = conditionBlock->getArgument(0);

    // Get the loop terminator, which should be the last operation of the
    // original loop body. And `firstBodyBlock` points to that loop body.
    auto terminator = dyn_cast<mlir::AffineYieldOp>(firstBodyBlock->back());
    if (!terminator)
      return failure();

    // First, we fill the content of the lastBodyBlock with how the loop
    // iterator steps.
    rewriter.setInsertionPointToEnd(lastBodyBlock);
    auto stepped = rewriter.create<arith::AddIOp>(loc, iv, step).getResult();

    // Next, we get the loop carried values, which are terminator operands.
    SmallVector<Value, 8> loopCarried;
    loopCarried.push_back(stepped);
    loopCarried.append(terminator.operand_begin(), terminator.operand_end());
    rewriter.create<mlir::BranchOp>(loc, conditionBlock, loopCarried);

    // Then we fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comparison = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, iv, upperBound.getValue()[0]);

    rewriter.create<mlir::CondBranchOp>(loc, comparison, firstBodyBlock,
                                        ArrayRef<Value>(), endBlock,
                                        ArrayRef<Value>());

    // We also insert the branch operation at the end of the initBlock.
    rewriter.setInsertionPointToEnd(initBlock);
    // TODO: should we add more operands here?
    rewriter.create<mlir::BranchOp>(loc, conditionBlock,
                                    lowerBound.getValue()[0]);

    // Finally, setup the firstBodyBlock.
    rewriter.setInsertionPointToEnd(firstBodyBlock);
    // TODO: is it necessary to add this explicit branch operation?
    rewriter.create<mlir::BranchOp>(loc, lastBodyBlock);

    // Remove the original forOp and the terminator in the loop body.
    rewriter.eraseOp(terminator);
    rewriter.eraseOp(forOp);
  }

  return success();
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

LogicalResult replaceCallOps(handshake::FuncOp f,
                             ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    /// An instance is activated whenever control arrives at the basic block of
    /// the source callOp.
    Operation *cntrlMg =
        block.isEntryBlock() ? getStartOp(&block) : getControlMerge(&block);
    assert(cntrlMg);
    for (Operation &op : block) {
      if (auto callOp = dyn_cast<CallOp>(op)) {
        llvm::SmallVector<Value> operands;
        llvm::copy(callOp.getOperands(), std::back_inserter(operands));
        operands.push_back(cntrlMg->getResult(0));
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

#define returnOnError(logicalResult)                                           \
  if (failed(logicalResult))                                                   \
    return logicalResult;

LogicalResult lowerFuncOp(mlir::FuncOp funcOp, MLIRContext *ctx) {
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

  // Rewrite affine.for operations.
  if (failed(partiallyLowerFuncOp<handshake::FuncOp>(rewriteAffineFor, ctx,
                                                     newFuncOp)))
    return newFuncOp.emitOpError("failed to rewrite Affine loops");

  // Perform dataflow conversion
  MemRefToMemoryAccessOp MemOps;
  returnOnError(partiallyLowerFuncOp<handshake::FuncOp>(
      [&](handshake::FuncOp nfo, ConversionPatternRewriter &rewriter) {
        MemOps = replaceMemoryOps(nfo, rewriter);
        return success();
      },
      ctx, newFuncOp));

  returnOnError(partiallyLowerFuncOp<handshake::FuncOp>(setControlOnlyPath, ctx,
                                                        newFuncOp));
  returnOnError(
      partiallyLowerFuncOp<handshake::FuncOp>(addMergeOps, ctx, newFuncOp));
  returnOnError(
      partiallyLowerFuncOp<handshake::FuncOp>(replaceCallOps, ctx, newFuncOp));
  returnOnError(
      partiallyLowerFuncOp<handshake::FuncOp>(addBranchOps, ctx, newFuncOp));
  returnOnError(
      partiallyLowerFuncOp<handshake::FuncOp>(addSinkOps, ctx, newFuncOp));
  returnOnError(partiallyLowerFuncOp<handshake::FuncOp>(
      connectConstantsToControl, ctx, newFuncOp));
  returnOnError(
      partiallyLowerFuncOp<handshake::FuncOp>(addForkOps, ctx, newFuncOp));
  checkDataflowConversion(newFuncOp);

  bool lsq = false;
  returnOnError(partiallyLowerFuncOp<handshake::FuncOp>(
      [&](handshake::FuncOp nfo, ConversionPatternRewriter &rewriter) {
        return connectToMemory(nfo, MemOps, lsq, rewriter);
      },
      ctx, newFuncOp));

  // Add  control argument to entry block, replace references to the
  // temporary handshake::StartOp operation, and finally remove the start
  // op.
  returnOnError(partiallyLowerFuncOp<handshake::FuncOp>(
      [&](handshake::FuncOp nfo, PatternRewriter &rewriter) {
        argTypes.push_back(rewriter.getNoneType());
        auto funcType = rewriter.getFunctionType(argTypes, resTypes);
        nfo.setType(funcType);
        auto ctrlArg = nfo.front().addArgument(rewriter.getNoneType());

        // We've now added all types to the handshake.funcOp; resolve arg- and
        // res names to ensure they are up to date with the final type
        // signature.
        nfo.resolveArgAndResNames();

        Operation *startOp = findStartOp(&nfo.getRegion());
        startOp->getResult(0).replaceAllUsesWith(ctrlArg);
        rewriter.eraseOp(startOp);
        rewriter.eraseOp(funcOp);
        return success();
      },
      ctx, newFuncOp));

  return success();
}

namespace {
struct HandshakeInsertBufferPass
    : public HandshakeInsertBufferBase<HandshakeInsertBufferPass> {

  // Returns true if a block argument should have buffers added to its uses.
  static bool shouldBufferArgument(BlockArgument arg) {
    // At the moment, buffers only make sense on arguments which we know
    // will lower down to a handshake bundle.
    return arg.getType().isIntOrFloat() || arg.getType().isa<NoneType>();
  }
  // Perform a depth first search and insert buffers when cycles are detected.
  void bufferCyclesStrategy() {
    auto f = getOperation();
    DenseSet<Operation *> opVisited;
    DenseSet<Operation *> opInFlight;

    // Traverse each use of each argument of the entry block.
    auto builder = OpBuilder(f.getContext());
    for (auto &arg : f.getBody().front().getArguments()) {
      if (!shouldBufferArgument(arg))
        continue;
      for (auto &operand : arg.getUses()) {
        if (opVisited.count(operand.getOwner()) == 0)
          insertBufferDFS(operand.getOwner(), builder, bufferSize, opVisited,
                          opInFlight);
      }
    }
  }

  // Perform a depth first search and add a buffer to any un-buffered channel.
  void bufferAllStrategy() {
    auto f = getOperation();
    auto builder = OpBuilder(f.getContext());
    for (auto &arg : f.getArguments()) {
      if (!shouldBufferArgument(arg))
        continue;
      for (auto &use : arg.getUses())
        insertBufferRecursive(use, builder, bufferSize,
                              [](Operation *definingOp, Operation *usingOp) {
                                return !isa_and_nonnull<BufferOp>(definingOp) &&
                                       !isa<BufferOp>(usingOp);
                              });
    }
  }

  /// DFS-based graph cycle detection and naive buffer insertion. Exactly one
  /// 2-slot non-transparent buffer will be inserted into each graph cycle.
  void insertBufferDFS(Operation *op, OpBuilder &builder, unsigned numSlots,
                       DenseSet<Operation *> &opVisited,
                       DenseSet<Operation *> &opInFlight) {
    // Mark operation as visited and push into the stack.
    opVisited.insert(op);
    opInFlight.insert(op);

    // Traverse all uses of the current operation.
    for (auto &operand : op->getUses()) {
      auto *user = operand.getOwner();

      // If graph cycle detected, insert a BufferOp into the edge.
      if (opInFlight.count(user) != 0 && !isa<handshake::BufferOp>(op) &&
          !isa<handshake::BufferOp>(user)) {
        auto value = operand.get();

        builder.setInsertionPointAfter(op);
        auto bufferOp =
            builder.create<handshake::BufferOp>(op->getLoc(), value.getType(),
                                                /*slots=*/numSlots, value,
                                                /*sequential=*/true);
        value.replaceUsesWithIf(
            bufferOp,
            function_ref<bool(OpOperand &)>([](OpOperand &operand) -> bool {
              return !isa<handshake::BufferOp>(operand.getOwner());
            }));
      }
      // For unvisited operations, recursively call insertBufferDFS() method.
      else if (opVisited.count(user) == 0)
        insertBufferDFS(user, builder, bufferSize, opVisited, opInFlight);
    }
    // Pop operation out of the stack.
    opInFlight.erase(op);
  }

  void
  insertBufferRecursive(OpOperand &use, OpBuilder builder, size_t numSlots,
                        function_ref<bool(Operation *, Operation *)> callback) {
    auto oldValue = use.get();
    auto *definingOp = oldValue.getDefiningOp();
    auto *usingOp = use.getOwner();
    if (callback(definingOp, usingOp)) {
      builder.setInsertionPoint(usingOp);
      auto buffer = builder.create<handshake::BufferOp>(
          oldValue.getLoc(), oldValue.getType(),
          /*slots=*/numSlots, oldValue,
          /*sequential=*/true);
      use.getOwner()->setOperand(use.getOperandNumber(), buffer);
    }

    for (auto &childUse : usingOp->getUses())
      if (!isa<handshake::BufferOp>(childUse.getOwner()))
        insertBufferRecursive(childUse, builder, numSlots, callback);
  }

  void runOnOperation() override {
    if (strategies.empty())
      strategies = {"all"};

    for (auto strategy : strategies) {
      if (strategy == "cycles")
        bufferCyclesStrategy();
      else if (strategy == "all")
        bufferAllStrategy();
      else {
        emitError(getOperation().getLoc())
            << "Unknown buffer strategy: " << strategy;
        signalPassFailure();
        return;
      }
    }
  }
};

struct HandshakeRemoveBlockPass
    : HandshakeRemoveBlockBase<HandshakeRemoveBlockPass> {
  void runOnOperation() override { removeBasicBlocks(getOperation()); }
};

struct HandshakeDataflowPass
    : public HandshakeDataflowBase<HandshakeDataflowPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : llvm::make_early_inc_range(m.getOps<mlir::FuncOp>())) {
      if (failed(lowerFuncOp(funcOp, &getContext())))
        signalPassFailure();
    }

    // Legalize the resulting regions, which can have no basic blocks.
    for (auto func : m.getOps<handshake::FuncOp>())
      removeBasicBlocks(func);
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::createHandshakeDataflowPass() {
  return std::make_unique<HandshakeDataflowPass>();
}

std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
circt::createHandshakeRemoveBlockPass() {
  return std::make_unique<HandshakeRemoveBlockPass>();
}

std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
circt::createHandshakeInsertBufferPass() {
  return std::make_unique<HandshakeInsertBufferPass>();
}
