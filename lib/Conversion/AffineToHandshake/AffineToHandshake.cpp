//===- AffineToHandshake.cpp - Convert MLIR affine to handshake -*- C++ -*-===//
//
// This file implements the affine-to-handshake conversion.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToHandshake/AffineToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace std;
using namespace circt;
using namespace circt::handshake;

#define DEBUG_TYPE "affine-to-handshake"

namespace {

/// Search for a handshake::ControlMergeOp in the given block.
Operation *getControlMerge(Block *block) {
  // Returns CMerge of block
  for (Operation &op : *block) {
    if (isa<handshake::ControlMergeOp>(op)) {
      return &op;
    }
  }
  return nullptr;
}

/// Search for a handshake::StartOp in the given block.
Operation *getStartOp(Block *block) {
  // Returns CMerge of block
  for (Operation &op : *block) {
    if (isa<handshake::StartOp>(op)) {
      return &op;
    }
  }
  return nullptr;
}

/// Get the control operation from the given block. Will raise error if there
/// isn't one.
Operation *getControlOp(Block *block) {
  bool isEntryBlock = block->isEntryBlock();
  Operation *result;
  for (Operation &op : *block) {
    if ((isEntryBlock && isa<handshake::StartOp>(op)) ||
        (!isEntryBlock && isa<handshake::ControlMergeOp>(op))) {
      result = &op;
      break;
    }
  }

  assert(result != nullptr);
  return result;
}

} // namespace

namespace {
/// Functions and data structures that help with liveness analysis.

/// Maps from basic blocks to sets of values.
typedef llvm::DenseMap<mlir::Block *, std::vector<mlir::Value>> BlockValues;
/// Maps from basic blocks to sets of operations.
typedef llvm::DenseMap<mlir::Block *, std::vector<mlir::Operation *>> BlockOps;

/// Get the USE values for all basic blocks in the given handshake::FuncOp. For
/// each block, this function iterates through each of its operations to find
/// operands that are not owned by the current block.
LogicalResult getBlockUses(handshake::FuncOp f, BlockValues &uses) {
  for (Block &block : f) {
    for (Operation &op : block) {
      if (isa<AffineForOp, scf::ForOp, AffineYieldOp, scf::YieldOp,
              mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op))
        continue;

      for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
        mlir::Value operand = op.getOperand(i);
        Block *owner;

        if (operand.getKind() == mlir::Value::Kind::BlockArgument) {
          // Operand is block argument, get its owner block
          owner = operand.cast<BlockArgument>().getOwner();
        } else {
          // Operand comes from operation, get the block of its defining op
          Operation *defOp = operand.getDefiningOp();
          // TODO: Needs better error handling with more verbose information.
          assert(defOp != NULL);
          owner = defOp->getBlock();
        }

        // If this operand is defined in some other block, and it has not yet
        // been added, we add it to the USE set of the current block.
        if (owner != &block &&
            std::find(uses[&block].begin(), uses[&block].end(), operand) ==
                uses[&block].end())
          uses[&block].push_back(operand);
      }
    }
  }

  return success();
}

/// Get the DEF set of values for each basic block in the given
/// handshake::FuncOp. A value is defined in a block if it is the result of an
/// operation in that block, or it is the block arguments of that block.
LogicalResult getBlockDefs(handshake::FuncOp f, BlockValues &defs) {
  for (Block &block : f) {
    // Add operation results to the DEF set.
    for (Operation &op : block) {
      // HACK: operations of these types should have been removed.
      if (isa<AffineForOp, scf::ForOp, AffineYieldOp, scf::YieldOp,
              mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op))
        continue;
      if (op.getNumResults() > 0) {
        for (auto result : op.getResults())
          defs[&block].push_back(result);
      }
    }

    // Add block arguments to the DEF set.
    for (auto &arg : block.getArguments())
      defs[&block].push_back(arg);
  }

  return success();
}

vector<Value> vectorUnion(vector<Value> v1, vector<Value> v2) {
  // Returns v1 U v2
  // Assumes unique values in v1

  for (int i = 0, e = v2.size(); i < e; ++i) {
    Value val = v2[i];
    if (std::find(v1.begin(), v1.end(), val) == v1.end())
      v1.push_back(val);
  }
  return v1;
}

vector<Value> vectorDiff(vector<Value> v1, vector<Value> v2) {
  // Returns v1 - v2
  vector<Value> d;

  for (int i = 0, e = v1.size(); i < e; ++i) {
    Value val = v1[i];
    if (std::find(v2.begin(), v2.end(), val) == v2.end())
      d.push_back(val);
  }
  return d;
}

/// Perform liveness analysis on all the basic blocks in the given
/// handshake::FuncOp. The algorithm is adapted from:
/// https://suif.stanford.edu/~courses/cs243/lectures/l2.pdf (slide 19)
/// Results will be updated to the input map mapping from blocks to the sets of
/// entry variables (mlir::Values) that are live. Liveness analysis in this
/// conversion is mainly used to create merge operations.
LogicalResult livenessAnalysis(handshake::FuncOp f, BlockValues &blockLiveIns) {
  // blockUses: values used in block but not defined in block
  // blockDefs: values defined in block
  BlockValues blockUses, blockDefs;
  if (failed(getBlockUses(f, blockUses)) || failed(getBlockDefs(f, blockDefs)))
    return failure();

  // Iterate while there are any changes to any of the livein sets
  bool change = true;
  while (change) {
    change = false;
    // liveOuts(b): all liveins of all successors of b
    // liveOuts(b) = U (blockLiveIns(s)) forall successors s of b
    for (Block &block : f) {
      vector<Value> liveOuts;
      for (int i = 0, e = block.getNumSuccessors(); i < e; ++i) {
        Block *succ = block.getSuccessor(i);
        liveOuts = vectorUnion(liveOuts, blockLiveIns[succ]);
      }
      // diff(b):  liveouts of b which are not defined in b
      // diff(b) = liveOuts(b) - blockDefs(b)
      vector<Value> diff = vectorDiff(liveOuts, blockDefs[&block]);
      // liveIns(b) = blockUses(b) U diff(b)
      vector<Value> tmpLiveIns = vectorUnion(blockUses[&block], diff);
      // Update blockLiveIns if new liveins found
      if (tmpLiveIns.size() > blockLiveIns[&block].size()) {
        blockLiveIns[&block] = tmpLiveIns;
        change = true;
      }
    }
  }

  return success();
}

unsigned getNumBlockPredecessors(Block *block) {
  // Returns number of block predecessors
  auto predecessors = block->getPredecessors();
  return std::distance(predecessors.begin(), predecessors.end());
}

/// Insert merge operation for the given Value in the given basic block.
/// Depending on the type of the Value and the number of predecessor blocks, we
/// create different types of merge:
/// - A ControlMergeOp (CMerge) is created for (non-BlockArgument) values on the
/// control-only path, which is originated from the StartOp.
/// - A MergeOp is created if the value is a BlockArgument and the block has
/// less than one predecessor.
/// - A MuxOp is created if the value is a BlockArgument and the block has
/// multiple predecessors.
Operation *insertMerge(Block *block, mlir::Value val,
                       ConversionPatternRewriter &rewriter) {
  unsigned numPredecessors = getNumBlockPredecessors(block);

  // Control-only path originates from StartOp.
  if (val.getKind() != mlir::Value::Kind::BlockArgument) {
    Operation *defOp = val.getDefiningOp();
    // A live-in value should have exactly a single def-op.
    assert(defOp != nullptr);
    if (isa<handshake::StartOp>(defOp))
      return rewriter.create<handshake::ControlMergeOp>(block->front().getLoc(),
                                                        val, numPredecessors);
  }

  // Now we handle Values that are either BlockArguments or values that are not
  // originated from StartOp.
  // If there are no block predecessors (i.e., entry block), function argument
  // is set as its single operand.
  if (numPredecessors <= 1)
    return rewriter.create<handshake::MergeOp>(block->front().getLoc(), val, 1);

  // Otherwise, we create a MuxOp that has an explicit selector to select one
  // of its inputs as its output.
  return rewriter.create<handshake::MuxOp>(block->front().getLoc(), val,
                                           numPredecessors);
}

/// Insert merge operations after live-in values of each basic block in the
/// given handshake::FuncOp. The newly inserted merge operations are stored into
/// the blockMerges argument.
LogicalResult insertMergeOps(handshake::FuncOp f, BlockValues blockLiveIns,
                             BlockOps &blockMerges,
                             ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    rewriter.setInsertionPointToStart(&block);
    // Insert merge operations for all live-in values.
    for (auto &val : blockLiveIns[&block]) {
      Operation *newOp = insertMerge(&block, val, rewriter);
      if (newOp)
        blockMerges[&block].push_back(newOp);
    }

    // Insert merge operations for all BlockArguments of this current block,
    // which are not in the live-in set. We need to add merge operations for
    // them as well, because, for example:
    //
    // ```mlir
    // %1 = ...
    // br bb1(%1)
    // ^bb1(%arg1):
    //   %2 = op(%arg1)
    // ```
    //
    // In this case, `%arg1` is a BlockArgument of the block `^bb1`, and based
    // on the control-flow, `%1` will be passed into `^bb1` as `%arg1`, and
    // therefore, `%arg1` is pratically a live-in variable and we need to add a
    // merge operation for it.
    for (auto &arg : block.getArguments()) {
      Operation *newOp = insertMerge(&block, arg, rewriter);
      if (newOp)
        blockMerges[&block].push_back(newOp);
    }
  }

  return success();
}

/// Check if block contains operation which produces the given Value `val`.
/// `val` cannot be BlockArgument in this case.
bool blockHasSrcOp(mlir::Value val, Block *block) {
  // Arguments do not have an operation producer
  if (val.getKind() == Value::Kind::BlockArgument)
    return false;
  // The defining operation of `val` should exist.
  Operation *op = val.getDefiningOp();
  assert(op != nullptr);
  // If the defining op is in the given block, we can say that `val` is defined
  // in that block.
  return (op->getBlock() == block);
}

/// The given MergeOp `op` has a list of operands, the first of which is the
/// value being merged, and the rest are the values that are merged into the
/// first value. However, when creating `op`, these other operands are not
/// explicitly specified (they stay the same as the first operand), and this
/// function aims to replace them with the correct values. Each function call
/// will handle one predecessor block (`predBlock`), which corresponds to one
/// merge operand.
///
/// Depending on the value being merged, there are two cases for replacement:
///
/// 1) If the value being merged is a BlockArgument of its owning block, then we
/// know that the actual values to be merged into the first arguments are those
/// passed into the block as that BlockArgument. We can simply search for them
/// in the terminator of the given `predBlock`.
/// 2) If not, given the pressumption that, besides BlockArguments, values
/// beging merged can only be live-in values. Each live-in value has a single
/// source of definition. If it is defined in the given `predBlock`, then we are
/// merging the right thing; if not, it should come from another MergeOp in that
/// `predBlock`. We should search for that MergeOp and return its result, which
/// will further become the merge operand of `op`.
mlir::Value getMergeOperand(Operation *op, Block *predBlock,
                            BlockOps blockMerges) {
  // The value that is merged by the given MergeOp `op`, which can be retrieved
  // from the first operand of the MergeOp.
  mlir::Value srcVal = op->getOperand(0);
  // The parent block of the MergeOp.
  Block *block = op->getBlock();

  // If the value being merged is NOT a BlockArgument of the current block, then
  // we return itself if it is defined in the given predecessor `predBlock`;
  // otherwise, that value should be a merged result from the `predBlock`, and
  // we should find the exact MergeOp that merges that value and return the
  // merge result.
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
  // If the Value being merged is a BlockArgument of this current block, then we
  // should replace the merge operands by the actual values passed in as the
  // BlockArgument, and these values should come from the predecessors of this
  // current block. We do this by extracting the operands that passed into the
  // terminator of the predecessor block, which should either be a CondBranchOp
  // or a BranchOp. From these terminator operands we can get what the exact
  // values that are passed into the current block and will be merged.
  else {
    unsigned index = srcVal.cast<BlockArgument>().getArgNumber();
    Operation *termOp = predBlock->getTerminator();
    if (auto br = dyn_cast<mlir::CondBranchOp>(termOp)) {
      if (block == br.getTrueDest())
        return br.getTrueOperand(index);
      else {
        assert(block == br.getFalseDest());
        return br.getFalseOperand(index);
      }
    } else if (isa<mlir::BranchOp>(termOp)) {
      return termOp->getOperand(index);
    } else {
      // TODO: what this case could be?
      return nullptr;
    }
  }
}

/// Remove all block arguments, which should be guaranteed to have no usage.
/// When removing BlockArguments, the branch operations that pass them in will
/// be removed as well by `eraseArguments`.
/// TODO: is it so?
void removeBlockOperands(handshake::FuncOp f) {
  for (Block &block : f) {
    if (!block.isEntryBlock()) {
      int x = block.getNumArguments() - 1;
      for (int i = x; i >= 0; --i)
        block.eraseArgument(i);
    }
  }
}

/// All merge operands are initially set to original (defining) value. This
/// function iterates through every basic block in the given handshake::FuncOp,
/// and for each block, it goes through every merge op recorded in the map
/// `blockMerges`. For each operand starting from index 1, we relate them to the
/// corresponding block predecessor (ordering determined by
/// `getPredecessors()`). This relationship is built by `getMergeOperand`.
/// Before calling `reconnectMergeOps`, those operations that use the
/// value to be merged are still using the original value. We will use the 0-th
/// operand of each merge, i.e., the value being merged, to identify operations
/// that use it and replace such usage with the merge result.
void reconnectMergeOps(handshake::FuncOp f, BlockOps &blockMerges) {
  for (Block &block : f) {
    for (Operation *op : blockMerges[&block]) {
      unsigned count = 1;
      // Set appropriate operand from predecessor block.
      for (auto *predBlock : block.getPredecessors()) {
        mlir::Value mgOperand = getMergeOperand(op, predBlock, blockMerges);
        // TODO: is it possible that mgOperand can actually be NULL?
        assert(mgOperand != nullptr);
        op->setOperand(count, mgOperand);
        count++;
      }
      // Let other operations to use the merge result instead of the original
      // object representing the value to be merged (the first operand).
      for (Operation &otherOp : block)
        if (!isa<MergeLikeOpInterface>(otherOp))
          otherOp.replaceUsesOfWith(op->getOperand(0), op->getResult(0));
    }
  }

  // Disconnect original value (Operand(0), used as helper) from all merges. If
  // the original value must be a merge operand, it is still set as some
  // subsequent operand. If block has multiple predecessors, connect Muxes to
  // ControlMerge
  for (Block &block : f) {
    unsigned numPredecessors = getNumBlockPredecessors(&block);

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
}

/// For all blocks, we add MergeOp to their live-in values and their
/// BlockArguments. A MergeOp is analogous to the Phi-functions in SSA, its
/// behaviour is to dynamically select values that may come from any
/// control-flow path. More details can be found in this paper:
/// https://dl.acm.org/doi/abs/10.1145/3174243.3174264
LogicalResult addMergeOps(handshake::FuncOp f,
                          ConversionPatternRewriter &rewriter) {
  // blockLiveIns: live in variables of block
  BlockValues liveIns;
  if (failed(livenessAnalysis(f, liveIns)))
    return failure();
  // Insert merge operations if necessary for all live-in values of all basic
  // blocks. The mapping between the new merge ops and their owner blocks are
  // created as well.
  BlockOps mergeOps;
  if (failed(insertMergeOps(f, liveIns, mergeOps, rewriter)))
    return failure();

  // Set merge operands and uses
  reconnectMergeOps(f, mergeOps);

  // After the previous steps, all BlockArguments should have no usage (replaced
  // by MergeOp results) and can be safely removed.
  removeBlockOperands(f);

  return success();
}
} // namespace

namespace {

/// This function determines whether a value is a live-out value of a block by
/// checking whether it is being used by any MergeOp. This works because
/// live-out values are the union of all live-in values from all successors, and
/// live-in values are always being merged. This function should only be called
/// when all the necessary MergeOp operations are added.
bool isLiveOut(Value val) {
  for (auto &u : val.getUses())
    if (isa<MergeLikeOpInterface>(u.getOwner()))
      return true;
  return false;
}

/// A value can have multiple branches in a single successor block (for
/// instance, there can be an SSA phi and a merge that we insert). This function
/// determines the number of branches to insert based on the value uses in
/// successor blocks. For each successor of the given block, if its own number
/// of uses of `val` is the largest among all successors, then we return this
/// number.
unsigned getBranchCount(mlir::Value val, Block *block) {
  unsigned uses = 0;
  for (unsigned i = 0, e = block->getNumSuccessors(); i < e; ++i) {
    unsigned curr = 0;
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
mlir::Value getSuccResult(Operation *termOp, Operation *newOp,
                          Block *succBlock) {
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

/// Get the live-out set for each block in the given handshake::FuncOp. Whether
/// a value is live-out is determined by the `isLiveOut` function.
LogicalResult getBlockLiveOuts(handshake::FuncOp f, BlockValues &liveOuts) {
  for (Block &block : f) {
    for (Operation &op : block) {
      for (auto result : op.getResults())
        if (isLiveOut(result))
          liveOuts[&block].push_back(result);
    }
  }
  return success();
}

/// Insert a BranchOp for each live-out value of every block in the given
/// handshake::FuncOp. The insertion point will immediately be after each
/// block's terminator. And we create BranchOp from the original terminators,
/// which should be either mlir::CondBranchOp or mlir::BranchOp.
LogicalResult insertBranchOps(handshake::FuncOp f, BlockValues &liveOuts,
                              ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    Operation *termOp = block.getTerminator();
    rewriter.setInsertionPoint(termOp);

    for (Value val : liveOuts[&block]) {
      // Count the number of branches which the liveout needs (being used).
      unsigned numBranches = getBranchCount(val, &block);

      // Instantiate branches and connect them to MergeOps.
      for (unsigned i = 0, e = numBranches; i < e; ++i) {
        Operation *newOp = nullptr;

        if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(termOp))
          newOp = rewriter.create<handshake::ConditionalBranchOp>(
              termOp->getLoc(), condBranchOp.getCondition(), val);
        else if (isa<mlir::BranchOp>(termOp))
          newOp = rewriter.create<handshake::BranchOp>(termOp->getLoc(), val);

        // When the BranchOp in the handshake realm is created, we replace the
        // uses of the revious BranchOps' results to the current handshake
        // branches.
        if (newOp != nullptr) {
          for (int i = 0, e = block.getNumSuccessors(); i < e; ++i) {
            Block *succ = block.getSuccessor(i);
            mlir::Value res = getSuccResult(termOp, newOp, succ);

            for (auto &u : val.getUses()) {
              if (u.getOwner()->getBlock() == succ) {
                u.getOwner()->replaceUsesOfWith(val, res);
                break;
              }
            }
          }
        }
        // TODO: what if newOp is nullptr?
      }
    }
  }
  return success();
}

/// This function creates new handshake terminators after the original mlir
/// branches that will be removed.
LogicalResult rewriteBranchTerminators(handshake::FuncOp f,
                                       ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    Operation *termOp = block.getTerminator();
    if (isa<mlir::CondBranchOp>(termOp) || isa<mlir::BranchOp>(termOp)) {
      SmallVector<mlir::Block *, 8> results(block.getSuccessors());
      // The new terminator will take all the original successors list as its
      // arguments.
      rewriter.setInsertionPointToEnd(&block);
      rewriter.create<handshake::TerminatorOp>(termOp->getLoc(), results);

      // Remove the Operands to keep the single-use rule.
      for (int i = 0, e = termOp->getNumOperands(); i < e; ++i)
        termOp->eraseOperand(0);
      assert(termOp->getNumOperands() == 0);
      rewriter.eraseOp(termOp);
    }
  }
  return success();
}

LogicalResult addBranchOps(handshake::FuncOp f,
                           ConversionPatternRewriter &rewriter) {
  // Get the mapping between blocks and their live-out values.
  BlockValues liveOuts;
  if (failed(getBlockLiveOuts(f, liveOuts)))
    return failure();

  // Insert a BranchOp for every live-out value in every block.
  if (failed(insertBranchOps(f, liveOuts, rewriter)))
    return failure();

  // Remove StandardOp branch terminators and place new terminator.
  // TODO: should be removed in some subsequent pass (we only need it to pass
  // checks in Verifier.cpp)
  if (failed(rewriteBranchTerminators(f, rewriter)))
    return failure();
  return success();
}

} // namespace

namespace {
/// Functions related to rewriting AffineForOp.

/// AffineMap related.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  AffineApplyExpander(OpBuilder &builder, ValueRange dimValues,
                      ValueRange symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy>
  Value buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op.getResult();
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) { return nullptr; }
  Value visitModExpr(AffineBinaryOpExpr expr) { return nullptr; }
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) { return nullptr; }
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) { return nullptr; }

  Value visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op = builder.create<mlir::ConstantOp>(loc, builder.getIndexType(),
                                               valueAttr);
    return op.getResult();
  }

  Value visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  ValueRange dimValues;
  ValueRange symbolValues;

  Location loc;
};

mlir::Value expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                             ValueRange dimValues, ValueRange symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

/// Lower the results of an affine map into operations that do the actual
/// computation.
Optional<llvm::SmallVector<mlir::Value, 8>>
expandAffineMap(OpBuilder &builder, Location loc, AffineMap affineMap,
                ValueRange operands) {
  auto numDims = affineMap.getNumDims();

  // Expand each affine map result.
  auto expanded = llvm::to_vector<8>(
      llvm::map_range(affineMap.getResults(),
                      [numDims, &builder, loc, operands](AffineExpr expr) {
                        return expandAffineExpr(builder, loc, expr,
                                                operands.take_front(numDims),
                                                operands.drop_front(numDims));
                      }));

  // Return the expanded affine map results if all of them are not nullptr.
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;

  return llvm::None;
}

/// Build a sequence of min/max operations to reduce the output from the values.
/// We use this function to resolve the lower/upper bound of AffineForOp.
mlir::Value buildMinMaxReductionSeq(Location loc, CmpIPredicate predicate,
                                    ValueRange values, OpBuilder &builder) {
  assert(!llvm::empty(values) && "empty min/max chain");

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<SelectOp>(loc, cmpOp.getResult(), value, *valueIt);
  }

  return value;
}

/// Take the maximum of the results from the expanded affine map.
mlir::Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                              ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, CmpIPredicate::sgt, *values, builder);
  return nullptr;
}

/// Take the minimum of the results from the expanded affine map.
mlir::Value lowerAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                              ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, CmpIPredicate::slt, *values, builder);
  return nullptr;
}

/// Extract the upper bound from the results of the upper-bound affine map in
/// the AffineForOp. We take the min of them. Operands are dim and symbol
/// values.
mlir::Value lowerAffineUpperBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMin(builder, op.getLoc(), op.getUpperBoundMap(),
                           op.getUpperBoundOperands());
}

/// Extract the lower bound from the results of the lower-bound affine map in
/// the AffineForOp. We take the max of them. Operands are dim and symbol
/// values.
mlir::Value lowerAffineLowerBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMax(builder, op.getLoc(), op.getLowerBoundMap(),
                           op.getLowerBoundOperands());
}

/// Rewrite the given AffineForOp instance to a scf::ForOp, which will be
/// further lowered in later rewrite passes.
LogicalResult rewriteAffineForOp(AffineForOp op,
                                 ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();

  mlir::Value lowerBound = lowerAffineLowerBound(op, rewriter);
  mlir::Value upperBound = lowerAffineUpperBound(op, rewriter);
  mlir::Value step = rewriter.create<mlir::ConstantIndexOp>(loc, op.getStep());

  auto f = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
  rewriter.eraseBlock(f.getBody());
  // Put the body region of the AffineForOp into the scf::ForOp.
  rewriter.inlineRegionBefore(op.region(), f.region(), f.region().end());
  rewriter.eraseOp(op);

  SmallVector<Operation *, 4> yieldOpsToRemove;
  for (Block &block : f.region()) {
    for (Operation &op : block) {
      if (auto yieldOp = dyn_cast<AffineYieldOp>(op)) {
        rewriter.setInsertionPoint(&op);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(&op);
      }
    }
  }

  return success();
}

void rewriteAffineForOps(handshake::FuncOp f,
                         ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    for (Operation &op : block) {
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        rewriter.setInsertionPoint(&op);
        rewriteAffineForOp(forOp, rewriter);
      }
    }
  }
}

LogicalResult rewriteSCFForOp(scf::ForOp forOp,
                              ConversionPatternRewriter &rewriter) {
  Location loc = forOp.getLoc();

  auto *initBlock = rewriter.getInsertionBlock();
  auto initPosition = rewriter.getInsertionPoint();
  auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

  auto *conditionBlock = &forOp.region().front();
  auto *firstBodyBlock =
      rewriter.splitBlock(conditionBlock, conditionBlock->begin());
  auto *lastBodyBlock = &forOp.region().back();
  rewriter.inlineRegionBefore(forOp.region(), endBlock);
  auto iv = conditionBlock->getArgument(0);

  // lastBodyBlock->print(llvm::outs());
  // Operation *terminator = lastBodyBlock->getTerminator();
  // HACK
  Operation *terminator = nullptr;
  for (Operation &op : *lastBodyBlock) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      terminator = &op;
      break;
    }
  }

  rewriter.setInsertionPointToEnd(lastBodyBlock);
  auto step = forOp.step();
  auto stepped = rewriter.create<AddIOp>(loc, iv, step).getResult();
  if (!stepped)
    return failure();

  SmallVector<Value, 8> loopCarried;
  loopCarried.push_back(stepped);
  loopCarried.append(terminator->operand_begin(), terminator->operand_end());
  rewriter.create<mlir::BranchOp>(loc, conditionBlock, loopCarried);
  rewriter.eraseOp(terminator);

  // Compute loop bounds before branching to the condition.
  rewriter.setInsertionPointToEnd(initBlock);
  Value lowerBound = forOp.lowerBound();
  Value upperBound = forOp.upperBound();
  if (!lowerBound || !upperBound)
    return failure();

  // The initial values of loop-carried values is obtained from the operands
  // of the loop operation.
  SmallVector<Value, 8> destOperands;
  destOperands.push_back(lowerBound);
  auto iterOperands = forOp.getIterOperands();
  destOperands.append(iterOperands.begin(), iterOperands.end());
  rewriter.create<mlir::BranchOp>(loc, conditionBlock, destOperands);

  // With the body block done, we can fill in the condition block.
  rewriter.setInsertionPointToEnd(conditionBlock);
  auto comparison =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, iv, upperBound);

  rewriter.create<mlir::CondBranchOp>(loc, comparison, firstBodyBlock,
                                      ArrayRef<Value>(), endBlock,
                                      ArrayRef<Value>());

  // The result of the loop operation is the values of the condition block
  // arguments except the induction variable on the last iteration.
  rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());

  return success();
}

void rewriteSCFForOps(handshake::FuncOp f,
                      ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *, 4> forOps;
  f.walk([&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      forOps.push_back(op);
    }
  });

  for (Operation *op : forOps) {
    rewriter.setInsertionPoint(op);
    rewriteSCFForOp(dyn_cast<scf::ForOp>(op), rewriter);
  }
}

} // namespace

namespace {

/// Get control-only value sent to the block terminator. We iterate every
/// operations in the given block, and if it is any of the terminator
/// operations, we get its control values and pass it to controlValues.
void getControlValues(Block *block,
                      SmallVectorImpl<mlir::Value> &controlValues) {
  for (Operation &op : *block) {
    if (auto BranchOp = dyn_cast<handshake::BranchOp>(op)) {
      if (BranchOp.isControl())
        controlValues.push_back(BranchOp.dataOperand());
    }
    if (auto BranchOp = dyn_cast<handshake::ConditionalBranchOp>(op)) {
      if (BranchOp.isControl())
        controlValues.push_back(BranchOp.dataOperand());
    }
    if (auto endOp = dyn_cast<handshake::ReturnOp>(op))
      controlValues.push_back(endOp.control());
  }
}

/// Get control values from memory access operations, which are results from
/// unioning control values collected from the parent blocks of all accessOps.
void getControlValues(std::vector<Operation *> &accessOps,
                      SmallVectorImpl<mlir::Value> &controlValues) {
  for (Operation *op : accessOps) {
    SmallVector<mlir::Value, 8> blockControlValues;
    Block *block = op->getBlock();
    getControlValues(block, blockControlValues);

    // Put the collected block control values into the result vector without
    // duplication.
    // TODO: maybe change the data structure to SmallSet?
    for (auto val : blockControlValues)
      if (std::find(controlValues.begin(), controlValues.end(), val) ==
          controlValues.end())
        controlValues.push_back(val);
  }
}

/// Collect the results from handshake load/store operations. The results of
/// handshake::LoadOp are a list of address indices, while the results of a
/// handshake::StoreOp are the data to be stored and the list of the accessed
/// address indices. These results will be used to initialize a MemoryOp.
void getHandshakeMemoryOpResults(Operation *op,
                                 SmallVectorImpl<mlir::Value> &results) {
  assert((isa<handshake::LoadOp, handshake::StoreOp>(op)));

  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    auto opResults = loadOp.addressResults();
    results.insert(results.end(), opResults.begin(), opResults.end());
  } else if (auto storeOp = dyn_cast<handshake::StoreOp>(op)) {
    auto opResults = storeOp.getResults();
    results.insert(results.end(), opResults.begin(), opResults.end());
  }
}

} // namespace

namespace {

/// For each operand of op, if it is oldValue, we update it to be the newValue
/// and return.
void rewriteFirstUse(Operation *op, mlir::Value oldValue,
                     mlir::Value newValue) {
  for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (op->getOperand(i) == oldValue) {
      op->setOperand(i, newValue);
      break;
    }
  }
}

/// Fork the result of the Operation op and updates its future usage.
void insertFork(Operation *op, mlir::Value result, bool isLazy,
                OpBuilder &rewriter) {
  // Get list of operations that use the given result. There could be
  // duplication since result might be used for multiple times.
  SmallVector<Operation *, 8> opsToProcess;
  for (auto &u : result.getUses()) {
    Operation *op = u.getOwner();
    if (isa<AffineForOp, scf::ForOp>(op))
      continue;
    opsToProcess.push_back(op);
  }

  unsigned numOpsToProcess = opsToProcess.size();
  if (numOpsToProcess <= 1)
    return;

  // Insert handshake::ForkOp directly after op.
  rewriter.setInsertionPointAfter(op);
  Operation *newOp = nullptr;
  if (isLazy)
    newOp = rewriter.create<handshake::LazyForkOp>(op->getLoc(), result,
                                                   numOpsToProcess);
  else
    newOp = rewriter.create<handshake::ForkOp>(op->getLoc(), result,
                                               numOpsToProcess);

  // Then we modify the uses of result to change them to a specific output from
  // the new ForkOp. Same operation that has multiple uses of result will have
  // different ForkOp outputs.
  for (int i = 0; i < numOpsToProcess; ++i)
    rewriteFirstUse(opsToProcess[i], result, newOp->getResult(i));
}

/// Insert Fork Operation for every operation with more than one successor
void addForkOps(handshake::FuncOp f, OpBuilder &rewriter) {
  for (Block &block : f) {
    for (Operation &op : block) {
      if (isa<AffineForOp, scf::ForOp>(op))
        continue;
      // Ignore terminators, and don't add Forks to Forks.
      if (op.getNumSuccessors() == 0 && !isa<handshake::ForkOp>(op)) {
        for (auto result : op.getResults()) {
          // If there is a result and it is used more than once, we insert a new
          // ForkOp.
          if (!result.use_empty() && !result.hasOneUse())
            insertFork(&op, result, /*isLazy=*/false, rewriter);
        }
      }
    }
  }
}

/// Add a handshake::SinkOp to any result that has no use. A SinkOp is designed
/// to discard any data that arrives at it.
void addSinkOps(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    for (Operation &op : block) {
      // Don't add SinkOp to those operations that will be later removed.
      // TODO: is this list completed?
      if (!isa<mlir::CondBranchOp, mlir::BranchOp, mlir::LoadOp,
               mlir::ConstantOp, mlir::AffineReadOpInterface,
               mlir::AffineWriteOpInterface, mlir::AffineForOp, scf::ForOp>(
              op)) {
        // Create SinkOps to any operation that has positive number of results,
        // if such results are not being used.
        if (op.getNumResults() > 0) {
          for (auto result : op.getResults()) {
            if (result.use_empty()) {
              rewriter.setInsertionPointAfter(&op);
              rewriter.create<handshake::SinkOp>(op.getLoc(), result);
            }
          }
        }
      }
    }
  }
}

/// Add control values to the constants in the function.
void rewriteConstantOps(handshake::FuncOp f,
                        ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    // Get the control operation in the current block.
    Operation *controlOp = getControlOp(&block);
    // Get all the constant operations for future deletion.
    SmallVector<Operation *, 8> constOps;

    for (Operation &op : block) {
      if (auto constOp = dyn_cast<mlir::ConstantOp>(op)) {
        rewriter.setInsertionPointAfter(&op);

        // Pass the constant value from the original mlir::ConstantOp and the
        // control value from the controlOp.
        Operation *newOp = rewriter.create<handshake::ConstantOp>(
            op.getLoc(), constOp.getValue(), controlOp->getResult(0));

        // Other operations that use the result from this old constant operation
        // will be replaced to use the newly created one.
        op.getResult(0).replaceAllUsesWith(newOp->getResult(0));

        // To be removed later.
        constOps.push_back(&op);
      }
    }

    // Erase old constant operations.
    unsigned numConstOps = constOps.size();
    for (unsigned i = 0; i < numConstOps; ++i) {
      Operation *op = constOps[i];
      for (unsigned j = 0, e = op->getNumOperands(); j < e; j++)
        op->eraseOperand(0);

      rewriter.eraseOp(op);
    }
  }
}

/// Remove alloc operations in the handshake::FuncOp.
void removeAllocOps(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {
  std::vector<Operation *> allocOps;

  for (Block &block : f) {
    for (Operation &op : block) {
      if (isa<mlir::AllocOp>(op)) {
        // The standard AllocOp should have exactly one use, and that use should
        // have been sinked.
        assert(op.getResult(0).hasOneUse());
        for (auto &u : op.getResult(0).getUses()) {
          Operation *useOp = u.getOwner();
          if (auto mg = dyn_cast<handshake::SinkOp>(useOp))
            allocOps.push_back(&op);
        }
      }
    }
  }

  for (unsigned i = 0, e = allocOps.size(); i != e; ++i) {
    auto *op = allocOps[i];
    rewriter.eraseOp(op);
  }
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
    for (int i = 0; i < block.getNumArguments(); i++) {
      block.eraseArgument(i);
    }
    block.erase();
  }
}

/// This function creates a pair of handshake::StartOp and handshake::ReturnOp
/// (hence a "control-only path") in the given handshake::FuncOp. The result of
/// the handshake::StartOp will be the input to the handshake::ReturnOp. Blocks
/// in the given handshake::FuncOp must have terminators of type mlir::ReturnOp,
/// and we will remove them and put their arguments into the newly created
/// handshake::ReturnOp.
void setControlOnlyPath(handshake::FuncOp f,
                        ConversionPatternRewriter &rewriter) {
  // Temporary start node (removed in later steps) in entry block
  Block *entryBlock = &f.front();
  rewriter.setInsertionPointToStart(entryBlock);
  Operation *startOp = rewriter.create<StartOp>(entryBlock->front().getLoc());

  // Replace original return ops with new returns with additional control input
  for (Block &block : f) {
    Operation *termOp = block.getTerminator();
    if (dyn_cast<mlir::ReturnOp>(termOp)) {
      rewriter.setInsertionPoint(termOp);

      // Remove operands from old return op and add them to new op
      SmallVector<Value, 8> operands(termOp->getOperands());
      for (int i = 0, e = termOp->getNumOperands(); i < e; ++i)
        termOp->eraseOperand(0);
      assert(termOp->getNumOperands() == 0);
      operands.push_back(startOp->getResult(0));
      rewriter.replaceOpWithNewOp<handshake::ReturnOp>(termOp, operands);
    }
  }
}

/// This function rewrites all the CallOp operations in the body by
/// handshake::InstanceOp, which represents data-flow design instances, which
/// has the same set of operands and results. The name of the instance will be
/// the same as the original callee.
void rewriteCallOps(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    for (Operation &op : block) {
      if (auto callOp = dyn_cast<CallOp>(op)) {
        rewriter.setInsertionPointAfter(&op);
        rewriter.replaceOpWithNewOp<handshake::InstanceOp>(
            callOp, callOp.getCallee(), callOp.getResultTypes(),
            callOp.getOperands());
      }
    }
  }
}

/// Mapping from memref objects to the operations that use them.
typedef llvm::MapVector<mlir::Value, std::vector<Operation *>> MemToAccess;

/// Stores all memory operands related information.
struct MemoryOperandInfo {
  unsigned numLoadOps; // Number of load operations.
};

/// Get the operands of a MemoryOp from all the access operations that access
/// it, which are ordered in a store-before-load sequence. The index mapping for
/// ordering (old->new) will be stored in indexMap. Additional information will
/// be stored in MemoryOperandInfo.
void getMemoryOperandsFromAccessOps(ArrayRef<Operation *> accessOps,
                                    SmallVectorImpl<mlir::Value> &operands,
                                    SmallVectorImpl<unsigned> &indexMap,
                                    MemoryOperandInfo &memOpInfo) {
  unsigned numAccessOps = accessOps.size();
  unsigned cnt = 0;        // The new index.
  unsigned numLoadOps = 0; // The number of load operations.

  SmallVector<mlir::Value, 8> opResults;
  // First we collect all store operation results.
  for (unsigned i = 0; i < numAccessOps; i++) {
    if (isa<handshake::StoreOp>(accessOps[i])) {
      opResults.clear();

      // Collect results and insert them into the operands.
      getHandshakeMemoryOpResults(accessOps[i], opResults);
      operands.insert(operands.end(), opResults.begin(), opResults.end());

      indexMap[i] = cnt++;
    }
  }

  // Then we go over the array again to collect results from load operations.
  for (unsigned i = 0; i < numAccessOps; i++) {
    if (isa<handshake::LoadOp>(accessOps[i])) {
      opResults.clear();

      // Collect results and insert them into the operands.
      getHandshakeMemoryOpResults(accessOps[i], opResults);
      operands.insert(operands.end(), opResults.begin(), opResults.end());

      indexMap[i] = cnt++;
      // Increase the number of load operations.
      numLoadOps++;
    }
  }

  // Summarizes the operand information collected.
  memOpInfo.numLoadOps = numLoadOps;
};

/// There are SinkOps that are no longer useful. They can be removed.
void removeRedundantSinkOps(handshake::FuncOp f,
                            ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *, 8> redundantSinkOps;

  for (Block &block : f) {
    for (Operation &op : block) {
      if (isa<handshake::SinkOp>(op)) {
        // If the SinkOp's single operand doesn't have exactly one use, or it is
        // defined by an AllocOp, i.e., it is a memref object, we treat this
        // SinkOp as redundant, because its operands should be taken care of by
        // some other operations.
        // TODO: verify this claim.
        auto operand = op.getOperand(0);
        if (!operand.hasOneUse() || isa<mlir::AllocOp>(operand.getDefiningOp()))
          redundantSinkOps.push_back(&op);
      }
    }
  }

  // Erase these redundant operations.
  for (unsigned i = 0, e = redundantSinkOps.size(); i < e; i++)
    rewriter.eraseOp(redundantSinkOps[i]);
}

/// Rewrite the affine memory operations (load/store) in handshake::FuncOp
/// instances to handshake LoadOp/StoreOp. We will keep a record of the
/// load/store relationship among memref objects and operations.
void rewriteMemoryOps(handshake::FuncOp f, MemToAccess &memToAccess,
                      ConversionPatternRewriter &rewriter) {
  // Cache all the load and store operations.
  SmallVector<Operation *, 8> loadAndStoreOps;
  // Traverse every operation in all blocks in the given handshake::FuncOp. If
  // it is a AffineLoadOp or AffineStoreOp, we will create a corresponding
  // new operation in the handshake scope.
  f.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
      loadAndStoreOps.push_back(op);
    }
  });

  unsigned numMemoryOps = loadAndStoreOps.size();
  for (unsigned i = 0; i < numMemoryOps; i++) {
    Operation *op = loadAndStoreOps[i];
    rewriter.setInsertionPoint(op);

    // Get essential memref access inforamtion.
    MemRefAccess access(op);
    // Use access indices as the operands for handshake load/store operands.
    SmallVector<mlir::Value, 4> operands(access.indices);
    // Pointer to the new operation.
    Operation *newOp;

    // TODO: There might be other cases not handled belong to
    // AffineReadOpInterface or AffineWriteOpInterface.
    if (isa<AffineLoadOp>(op)) {
      newOp = rewriter.create<handshake::LoadOp>(op->getLoc(), access.memref,
                                                 operands);
      op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
    } else if (isa<AffineStoreOp>(op)) {
      newOp = rewriter.create<handshake::StoreOp>(op->getLoc(),
                                                  op->getOperand(0), operands);
    }

    // Update the memref to access mapping.
    memToAccess[access.memref].push_back(newOp);
  }

  // Erase old memory operations.
  // TODO: Is there a more efficient way to write this?
  for (unsigned i = 0; i < numMemoryOps; i++) {
    Operation *op = loadAndStoreOps[i];
    unsigned numOperands = op->getNumOperands();
    for (unsigned j = 0; j < numOperands; ++j)
      op->eraseOperand(0);
    assert(op->getNumOperands() == 0);
    rewriter.eraseOp(op);
    // TODO: switch back to the rewriter based erase.
    // op->erase();
  }
}

/// Set all the load operations in accessOps to use the newly created memOp.
void setMemoryForLoadOps(ArrayRef<Operation *> accessOps, Operation *memOp) {
  unsigned loadOpIndex = 0;
  for (auto *op : accessOps) {
    if (isa<handshake::LoadOp>(op)) {
      op->insertOperands(op->getNumOperands(), memOp->getResult(loadOpIndex++));
    }
  }
}

/// Setup the control network for the MemoryOp.
void setMemoryControlNetwork(Operation *memOp, ArrayRef<Operation *> accessOps,
                             SmallVectorImpl<mlir::Value> &controlValues,
                             SmallVectorImpl<unsigned> &indexMap,
                             MemoryOperandInfo &memOpInfo,
                             ConversionPatternRewriter &rewriter) {
  unsigned numAccessOps = accessOps.size();

  // Create join operations for each control value.
  for (auto val : controlValues) {
    assert(val.hasOneUse());
    auto srcOp = val.getDefiningOp();

    // Insert only single join per block. For each newly created join operation,
    // we replace the uses of the corresponding control value to this join
    // operation result.
    if (!isa<handshake::JoinOp>(srcOp)) {
      rewriter.setInsertionPointAfter(srcOp);
      Operation *joinOp =
          rewriter.create<handshake::JoinOp>(srcOp->getLoc(), val);
      for (auto &u : val.getUses())
        if (u.getOwner() != joinOp)
          u.getOwner()->replaceUsesOfWith(val, joinOp->getResult(0));
    }
  }

  // Connect the memory access ops to the newly joined control values.
  for (unsigned i = 0; i < numAccessOps; i++) {
    Operation *op = accessOps[i];
    SmallVector<mlir::Value, 4> blockControlValues;
    getControlValues(op->getBlock(), blockControlValues);

    // Insert the control value for the current accessOp to the operands of the
    // defining operation of the blockControValues[0], which should be either a
    // JoinOp or a StartOp.
    if (blockControlValues.size() == 1) {
      auto srcOp = blockControlValues[0].getDefiningOp();
      if (!isa<handshake::JoinOp, handshake::StartOp>(srcOp))
        srcOp->emitError("Op expected to be a JoinOp or StartOp");

      srcOp->insertOperands(
          srcOp->getNumOperands(),
          memOp->getResult(memOpInfo.numLoadOps + indexMap[i]));
    }
  }

  // Finally, setup the control value inputs for the memory op. This is useful
  // for dependences analysis and scheduling.
  for (unsigned i = 0; i < numAccessOps; i++) {
    Operation *currOp = accessOps[i];
    Block *currBlock = currOp->getBlock();

    // Control operands for the memory.
    SmallVector<mlir::Value, 4> controlOperands;

    // The control value for the current block will be a control operand.
    Operation *controlOp = getControlOp(currBlock);
    controlOperands.push_back(controlOp->getResult(0));

    // For any access op before the current one, within this block, we add their
    // control values to the control operand list as well. This represents the
    // dependences at some level.
    for (unsigned j = 0; j < i; ++j) {
      Operation *predOp = accessOps[j];
      Block *predBlock = predOp->getBlock();

      if (currBlock == predBlock) {
        // Exclude RAR dependences. The control operand will be collected from
        // the memory operation's output.
        if (!(isa<handshake::LoadOp>(currOp) && isa<handshake::LoadOp>(predOp)))
          controlOperands.push_back(
              memOp->getResult(memOpInfo.numLoadOps + indexMap[j]));
      }
    }

    if (controlOperands.size() == 1) {
      // Single operand, no need to be joined.
      currOp->insertOperands(currOp->getNumOperands(), controlOperands[0]);
    } else {
      // Multiple operands, need to be joined.
      rewriter.setInsertionPoint(currOp);
      Operation *joinOp =
          rewriter.create<handshake::JoinOp>(currOp->getLoc(), controlOperands);
      currOp->insertOperands(currOp->getNumOperands(), joinOp->getResult(0));
    }
  }
}

/// Forks the output from MemoryOp.
void forkMemoryOpResults(handshake::FuncOp f, bool isLazy,
                         ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    for (Operation &op : block) {
      if (isa<handshake::MemoryOp, handshake::StartOp,
              handshake::ControlMergeOp>(op)) {
        for (auto result : op.getResults()) {
          // If there is a result and it is used more than once
          if (!result.use_empty() && !result.hasOneUse()) {
            insertFork(&op, result, isLazy, rewriter);
          }
        }
      }
    }
  }
}

/// Connect handshake access operations to memref objects. A new
/// handshake::MemoryOp will be created for each of the original memrefs.
/// `useLSQ` indicates whether we should create load-store queues for MemoryOps.
void connectAccessToMemory(handshake::FuncOp f, MemToAccess &memToAccess,
                           bool useLSQ, ConversionPatternRewriter &rewriter) {
  // Counter for the number of MemoryOp created.
  unsigned numMemoryOps = 0;

  // First we create new memory objects that are results of
  // handshake::MemoryOp.
  for (auto mem : memToAccess) {
    mlir::Value memref = mem.first;
    std::vector<Operation *> accessOps = mem.second;

    unsigned numAccessOps = accessOps.size();
    // Number of control-only outputs for each access, which indicate access
    // completion.
    unsigned numControls = useLSQ ? 0 : numAccessOps;

    // The new MemoryOp will be placed at the front of the entry block.
    Block *entryBlock = &f.front();
    rewriter.setInsertionPointToStart(entryBlock);

    // MemoryOp operands.
    SmallVector<mlir::Value, 8> operands;
    // MemoryOp control variables.
    SmallVector<mlir::Value, 8> controlValues;
    // Index mapping for reordered access operations.
    SmallVector<unsigned, 8> indexMap(numAccessOps, 0);
    // Additional information for memory operands.
    MemoryOperandInfo memOpInfo;

    // Get operands and the reordering index map.
    getMemoryOperandsFromAccessOps(accessOps, operands, indexMap, memOpInfo);
    // Collect control values from the blocks of all memory access operations.
    getControlValues(accessOps, controlValues);

    // In case of LSQ interface, set control values as inputs (used to trigger
    // allocation to LSQ)
    if (useLSQ)
      operands.insert(operands.end(), controlValues.begin(),
                      controlValues.end());

    // Create the MemoryOp object.
    // TODO: handle the case that memref is BlockArgument
    Operation *memoryOp = rewriter.create<handshake::MemoryOp>(
        entryBlock->front().getLoc(), operands, memOpInfo.numLoadOps,
        numControls, useLSQ, numMemoryOps, memref);
    numMemoryOps++;

    // Connect the result of this newly created MemoryOp instance, which is a
    // memory in handshake, to the access operations.
    setMemoryForLoadOps(accessOps, memoryOp);

    if (!useLSQ)
      setMemoryControlNetwork(memoryOp, accessOps, controlValues, indexMap,
                              memOpInfo, rewriter);
  }

  // Add ForkOp to memory results that have been used more than once.
  forkMemoryOpResults(f, useLSQ, rewriter);

  removeAllocOps(f, rewriter);
  // This is called because some previously dangling results are now connected
  // to the memory operation, and therefore, the SinkOps connected to them
  // should be removed.
  removeRedundantSinkOps(f, rewriter);
}

/// Remove the redundant StartOp from the entry block of f. We need to remove
/// this StartOp because the control value that it produces becomes the last
/// block argument of the entry block. Therefore, the StartOp becomes redundant.
void rewriteStartOp(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {
  Block *entryBlock = &f.front();

  // Get the last block argument and replace every use of the StartOp's result
  // with it.
  mlir::Value val = entryBlock->getArguments().back();
  Operation *startOp = getStartOp(entryBlock);
  assert(startOp != nullptr);
  startOp->getResult(0).replaceAllUsesWith(val);

  // Remove the StartOp.
  rewriter.eraseOp(startOp);
}

/// Lowering from standard FuncOp to handshake FuncOp. The rewrite function
/// does the following things: Creating a new handshake::FuncOp from the
/// standard mlir::FuncOp, rewriting a bunch of operations within the
/// mlir::FuncOp, and does some analysis.
class FuncOpLowering : public OpConversionPattern<mlir::FuncOp> {
public:
  using OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

  /// Get the attributes for the handshake::FuncOp from the original
  /// mlir::FuncOp. Only symbol and type attributes are kept.
  void getFuncOpAttributes(mlir::FuncOp funcOp,
                           SmallVectorImpl<NamedAttribute> &attrs) const {
    for (const auto &attr : funcOp.getAttrs()) {
      if (attr.first == SymbolTable::getSymbolAttrName() ||
          attr.first == impl::getTypeAttrName())
        continue;
      attrs.push_back(attr);
    }
  }

  /// Get the types of the arguments and results of mlir::FuncOp. A None type
  /// will be appended to both arguments and results lists to represent
  /// control input/output. The rewriter argument will be used to generate
  /// that None type object.
  void getArgAndResultTypes(mlir::FuncOp funcOp,
                            SmallVectorImpl<mlir::Type> &argTypes,
                            SmallVectorImpl<mlir::Type> &resTypes,
                            ConversionPatternRewriter &rewriter) const {
    unsigned numArguments = funcOp.getNumArguments();
    unsigned numResults = funcOp.getNumResults();
    auto args = funcOp.getArguments();
    auto resultTypes = funcOp.getType().getResults();

    argTypes.resize(numArguments + 1);
    resTypes.resize(numResults + 1);

    // Get function argument types.
    for (unsigned i = 0; i < numArguments; i++)
      argTypes[i] = args[i].getType();
    // Get function result types.
    std::copy(resultTypes.begin(), resultTypes.end(), resTypes.begin());

    // Add control input/output to function arguments/results
    auto noneType = rewriter.getNoneType();
    argTypes[numArguments] = noneType;
    resTypes[numResults] = noneType;
  }

  LogicalResult
  matchAndRewrite(mlir::FuncOp funcOp, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the new handshake::FuncOp from the given mlir::FuncOp.
    llvm::SmallVector<NamedAttribute, 4> attributes;
    llvm::SmallVector<mlir::Type, 8> argTypes;
    llvm::SmallVector<mlir::Type, 8> resTypes;

    getFuncOpAttributes(funcOp, attributes);
    getArgAndResultTypes(funcOp, argTypes, resTypes, rewriter);

    auto funcType = rewriter.getFunctionType(argTypes, resTypes);
    auto newFuncOp = rewriter.create<handshake::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), funcType, attributes);

    // Move the body region from the mlir::FuncOp to the new
    // handshake::FuncOp.
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Explicitly convert the signature of the handshake::FuncOp body region.
    unsigned numArguments = argTypes.size();
    TypeConverter::SignatureConversion result(numArguments);
    for (unsigned i = 0; i < numArguments; ++i)
      result.addInputs(i, argTypes[i]);
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // Now we start to rewrite the body of this new handshake::FuncOp.

    // Rewrite all the CallOp in the function body to handshake::InstanceOp.
    rewriteCallOps(newFuncOp, rewriter);

    // Create a pair of handshake::StartOp andhandshake::ReturnOp in the
    // newFuncOp.
    setControlOnlyPath(newFuncOp, rewriter);

    // Rewrite affine for ops
    rewriteAffineForOps(newFuncOp, rewriter);
    // Rewrite all the scf::ForOps created by rewriteAffineForOps.
    rewriteSCFForOps(newFuncOp, rewriter);

    addMergeOps(newFuncOp, rewriter);

    addBranchOps(newFuncOp, rewriter);

    // Rewrite memory operations
    MemToAccess memToAccess;
    rewriteMemoryOps(newFuncOp, memToAccess, rewriter);
    // Create SinkOp for those dangling values.
    addSinkOps(newFuncOp, rewriter);
    // Rewrite ConstantOp with handshake::ConstantOp that has control connected.
    rewriteConstantOps(newFuncOp, rewriter);
    // There might be constant operations that are not being used.
    addSinkOps(newFuncOp, rewriter);
    // Make sure every result in the program has exactly one use.
    addForkOps(newFuncOp, rewriter);

    // Connect memory access operations to newly created handshake::MemoryOp.
    // TODO: is there a way to trigger flag from CLI?
    bool useLSQ = false;
    connectAccessToMemory(newFuncOp, memToAccess, useLSQ, rewriter);

    // Rewrite the StartOp and remove its duplicantions. A bit more on why there
    // will be duplications: The single result of the StartOp will be used by
    // many operations, e.g., handshake::ConstantOp, handshake::ReturnOp, etc.
    // Changing their use to the block argument will avoid such problems.
    rewriteStartOp(newFuncOp, rewriter);

    rewriter.eraseOp(funcOp);

    return success();
  }
};

struct HandshakePass
    : public mlir::PassWrapper<HandshakePass, OperationPass<mlir::ModuleOp>> {

  HandshakePass() = default;
  HandshakePass(const HandshakePass &pass) {}

  // TODO: how to pass this to the ConversionPattern?
  Option<bool> useLSQ{
      *this, "use-lsq",
      llvm::cl::desc("Whether to use load-store queue when building the "
                     "memory interface.")};

  void runOnOperation() override {
    ModuleOp m = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<HandshakeOpsDialect, StandardOpsDialect,
                           scf::SCFDialect, AffineDialect>();

    OwningRewritePatternList patterns;
    patterns.insert<FuncOpLowering>(m.getContext());

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();

    // Legalize the resulting regions, which can have no basic blocks.
    for (auto func : m.getOps<handshake::FuncOp>())
      removeBasicBlocks(func);
  }
};
} // namespace

void handshake::registerAffineToHandshakePasses() {
  PassRegistration<HandshakePass>("affine-to-handshake",
                                  "Convert MLIR affine into CIRCT handshake");
}
