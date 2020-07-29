//===- Ops.h - StaticLogic MLIR Operations ----------------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "circt/Conversion/StandardToStaticLogic/StandardToStaticLogic.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace circt;
using namespace staticlogic;
using namespace std;

typedef DenseMap<Block *, vector<Value>> BlockValues;
typedef DenseMap<Block *, vector<Operation *>> BlockOps;

static BlockValues getBlockUses(mlir::FuncOp f) {
  // Returns map of values used in block but defined outside of block
  // For liveness analysis
  BlockValues uses;

  for (Block &block : f) {

    // Operands of operations in b which do not originate from operations or
    // arguments of b
    for (Operation &op : block) {
      for (int i = 0, e = op.getNumOperands(); i < e; ++i) {

        Block *operandBlock;

        if (op.getOperand(i).getKind() == Value::Kind::BlockArgument) {
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

static BlockValues getBlockDefs(mlir::FuncOp f) {
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

static vector<Value> vectorUnion(vector<Value> v1, vector<Value> v2) {
  // Returns v1 U v2
  // Assumes unique values in v1

  for (int i = 0, e = v2.size(); i < e; ++i) {
    Value val = v2[i];
    if (std::find(v1.begin(), v1.end(), val) == v1.end())
      v1.push_back(val);
  }
  return v1;
}

static vector<Value> vectorDiff(vector<Value> v1, vector<Value> v2) {
  // Returns v1 - v2
  vector<Value> d;

  for (int i = 0, e = v1.size(); i < e; ++i) {
    Value val = v1[i];
    if (std::find(v2.begin(), v2.end(), val) == v2.end())
      d.push_back(val);
  }
  return d;
}

static BlockValues getBlockLiveIns(mlir::FuncOp f) {

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
  return blockLiveIns;
}

/// This function converts all live-ins of a block into its arguments, and
/// converts all live-outs of a block into its return values. In this way, the
/// sequencial code inside of each block can be isolated and converted to a
/// function call.
static void canonicalizeFuncOp(mlir::FuncOp f, OpBuilder &builder,
                               BlockValues blockLiveIns) {
  for (Block &block : f) {
    // Add terminator operands.
    auto *termOp = block.getTerminator();
    if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(termOp)) {

      // Add true destination operands.
      SmallVector<Value, 4> trueOperands;
      for (auto trueOperand : condBranchOp.getTrueOperands())
        trueOperands.push_back(trueOperand);
      for (auto trueLiveOut : blockLiveIns[condBranchOp.getTrueDest()])
        trueOperands.push_back(trueLiveOut);

      // Add false destination operands.
      SmallVector<Value, 4> falseOperands;
      for (auto falseOperand : condBranchOp.getFalseOperands())
        falseOperands.push_back(falseOperand);
      for (auto falseLiveOut : blockLiveIns[condBranchOp.getFalseDest()])
        falseOperands.push_back(falseLiveOut);

      builder.setInsertionPointAfter(condBranchOp);
      builder.create<mlir::CondBranchOp>(
          f.getLoc(), condBranchOp.getCondition(), condBranchOp.getTrueDest(),
          trueOperands, condBranchOp.getFalseDest(), falseOperands);
      condBranchOp.erase();

    } else if (auto branchOp = dyn_cast<mlir::BranchOp>(termOp)) {
      SmallVector<Value, 4> operands;
      for (auto operand : branchOp.getOperands())
        operands.push_back(operand);
      for (auto liveOut : blockLiveIns[branchOp.getDest()])
        operands.push_back(liveOut);

      builder.setInsertionPointAfter(branchOp);
      builder.create<mlir::BranchOp>(f.getLoc(), branchOp.getDest(), operands);
      branchOp.erase();
    }

    // Add block arguments.
    for (auto val : blockLiveIns[&block]) {
      auto newArg = block.addArgument(val.getType());
      val.replaceUsesWithIf(
          newArg,
          function_ref<bool(OpOperand &)>([&block](OpOperand &operand) -> bool {
            return operand.getOwner()->getBlock() == &block;
          }));
    }
  }
}

static void createPipeline(mlir::FuncOp f, OpBuilder &builder) {
  BlockValues blockLiveIns = getBlockLiveIns(f);
  canonicalizeFuncOp(f, builder, blockLiveIns);

  unsigned blockIdx = 0;
  for (Block &block : f) {
    if (block.front().isKnownNonTerminator() &&
        block.back().isKnownTerminator()) {
      // Traverse all inputs and outputs to get their types.
      SmallVector<mlir::Type, 8> argTypes;
      for (auto &val : block.getArguments()) {
        argTypes.push_back(val.getType());
      }
      SmallVector<mlir::Type, 8> resTypes;
      for (auto val : block.back().getOperands()) {
        resTypes.push_back(val.getType());
      }
      auto opType = builder.getFunctionType(argTypes, resTypes);

      // Get attributes of the pipeline.
      SmallVector<NamedAttribute, 4> attributes;

      // Create pipeline operation.
      builder.setInsertionPoint(f);
      auto pipeline = builder.create<staticlogic::PipelineOp>(
          f.getLoc(), "pipeline" + std::to_string(blockIdx), opType,
          attributes);
      auto &pipelineOps = pipeline.addEntryBlock()->getOperations();

      // Create return operation in the block and move all operations except the
      // terminator operation into the pipeline.
      builder.setInsertionPoint(&block.back());
      builder.create<staticlogic::ReturnOp>(f.getLoc(),
                                            block.back().getOperands());
      pipelineOps.splice(pipelineOps.end(), block.getOperations(),
                         block.begin(), --block.end());

      unsigned argIdx = 0;
      for (auto &arg : pipeline.getArguments()) {
        block.getArgument(argIdx).replaceAllUsesWith(arg);
        argIdx += 1;
      }

      // Create call operation.
      auto pipelineCallOp = builder.create<staticlogic::InstanceOp>(
          f.getLoc(), pipeline.getType().getResults(), pipeline.getName(),
          block.getArguments());
      block.back().setOperands(pipelineCallOp.getResults());

      blockIdx += 1;
    }
  }
}

namespace {

struct CreatePipelinePass
    : public PassWrapper<CreatePipelinePass, OperationPass<mlir::FuncOp>> {
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    auto builder = OpBuilder(f.getContext());
    createPipeline(f, builder);
  }
};

} // namespace

void staticlogic::registerStandardToStaticLogicPasses() {
  PassRegistration<CreatePipelinePass>(
      "create-pipeline", "Create StaticLogic pipeline operations.");
}