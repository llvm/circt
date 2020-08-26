//===- AffineToHandshake.cpp - Convert MLIR affine to handshake -*- C++ -*-===//
//
// Copyright 2019 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// =============================================================================

#include "circt/Conversion/AffineToHandshake/AffineToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
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

/// Perform liveness analysis on blocks in the given handshake::FuncOp. Results
/// will be updated to the input map mapping from blocks to the sets of entry
/// variables (mlir::Values) that are live. Liveness analysis in this conversion
/// is mainly used to create merge operations.
void livenessAnalysis(handshake::FuncOp f, BlockValues &blockLiveIns) {}

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
void insertFork(Operation *op, mlir::Value result, OpBuilder &rewriter) {
  // Get list of operations that use the given result. There could be
  // duplication since result might be used for multiple times.
  SmallVector<Operation *, 8> opsToProcess;
  for (auto &u : result.getUses())
    opsToProcess.push_back(u.getOwner());
  unsigned numOpsToProcess = opsToProcess.size();

  // Insert handshake::ForkOp directly after op.
  rewriter.setInsertionPointAfter(op);
  Operation *newOp =
      rewriter.create<handshake::ForkOp>(op->getLoc(), result, numOpsToProcess);

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
      // Ignore terminators, and don't add Forks to Forks.
      if (op.getNumSuccessors() == 0 && !isa<handshake::ForkOp>(op)) {
        for (auto result : op.getResults()) {
          // If there is a result and it is used more than once, we insert a new
          // ForkOp.
          if (!result.use_empty() && !result.hasOneUse())
            insertFork(&op, result, rewriter);
        }
      }
    }
  }
}

/// Add merge operations to the handshake::FuncOp.
void addMergeOps(handshake::FuncOp f, ConversionPatternRewriter &rewriter) {
  BlockValues liveIns;
  livenessAnalysis(f, liveIns);
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
               mlir::AffineWriteOpInterface>(op)) {
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
        // The standard AllocOp should have exactly one use.
        // assert(op.getResult(0).hasOneUse());
        for (auto &u : op.getResult(0).getUses()) {
          Operation *useOp = u.getOwner();
          if (auto mg = dyn_cast<SinkOp>(useOp))
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
    if (isa<AffineLoadOp>(op))
      newOp = rewriter.create<handshake::LoadOp>(op->getLoc(), access.memref,
                                                 operands);
    else if (isa<AffineStoreOp>(op))
      newOp = rewriter.create<handshake::StoreOp>(op->getLoc(),
                                                  op->getOperand(0), operands);

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
    // rewriter.eraseOp(op);
    // TODO: switch back to the rewriter based erase.
    op->erase();
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
void forkMemoryOpResults(handshake::FuncOp f,
                         ConversionPatternRewriter &rewriter) {
  for (Block &block : f) {
    for (Operation &op : block) {
      if (isa<MemoryOp, StartOp, ControlMergeOp>(op)) {
        for (auto result : op.getResults()) {
          // If there is a result and it is used more than once
          if (!result.use_empty() && !result.hasOneUse()) {
          }
        }
      }
    }
  }
}

/// Connect handshake access operations to memref objects.
void connectAccessToMemory(handshake::FuncOp f, MemToAccess &memToAccess,
                           ConversionPatternRewriter &rewriter) {
  // Counter for the number of MemoryOp created.
  unsigned numMemoryOps = 0;
  // A placeholder for the flag of load-store queue.
  bool isLoadStoreQueue = false;

  // First we create new memory objects that are results of
  // handshake::MemoryOp.
  for (auto mem : memToAccess) {
    mlir::Value memref = mem.first;
    std::vector<Operation *> accessOps = mem.second;

    unsigned numAccessOps = accessOps.size();

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

    // Create the MemoryOp object.
    // TODO: handle the case that memref is BlockArgument
    Operation *memoryOp = rewriter.create<MemoryOp>(
        entryBlock->front().getLoc(), operands, memOpInfo.numLoadOps,
        numAccessOps, isLoadStoreQueue, numMemoryOps, memref);

    // Connect the result of this newly created MemoryOp instance, which is a
    // memory in handshake, to the access operations.
    setMemoryForLoadOps(accessOps, memoryOp);

    // TODO: enable the control network later.
    setMemoryControlNetwork(memoryOp, accessOps, controlValues, indexMap,
                            memOpInfo, rewriter);
  }

  forkMemoryOpResults(f, rewriter);

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
    // First of all, we create a pair of handshake::StartOp and
    // handshake::ReturnOp in the newFuncOp.
    setControlOnlyPath(newFuncOp, rewriter);
    // Rewrite all the CallOp in the function body to handshake::InstanceOp.
    rewriteCallOps(newFuncOp, rewriter);
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
    connectAccessToMemory(newFuncOp, memToAccess, rewriter);

    // Rewrite the StartOp and remove its duplicantions. A bit more on why there
    // will be duplications: The single result of the StartOp will be used by
    // many operations, e.g., handshake::ConstantOp, handshake::ReturnOp, etc.
    // Changing their use to the block argument will avoid such problems.
    rewriteStartOp(newFuncOp, rewriter);

    LLVM_DEBUG(newFuncOp.dump());

    rewriter.eraseOp(funcOp);

    return success();
  }
};

struct HandshakePass
    : public mlir::PassWrapper<HandshakePass, OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<HandshakeOpsDialect, StandardOpsDialect,
                           AffineDialect>();

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