//===- Ops.h - StaticLogic MLIR Operations ----------------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/StandardToStaticLogic/StandardToStaticLogic.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
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

static BlockValues getBlockLiveOuts(mlir::FuncOp f, BlockValues blockLiveIns) {
  BlockValues blockLiveOuts;

  for (Block &block : f) {
    vector<Value> liveOuts;
    for (int i = 0, e = block.getNumSuccessors(); i < e; ++i) {
      Block *succ = block.getSuccessor(i);
      liveOuts = vectorUnion(liveOuts, blockLiveIns[succ]);
    }
    blockLiveOuts[&block] = liveOuts;
  }
  return blockLiveOuts;
}

static BlockValues getPipelineIns(mlir::FuncOp f, BlockValues blockLiveIns) {
  // Push back block arguments
  for (Block &block : f) {
    for (auto &val : block.getArguments()) {
      blockLiveIns[&block].push_back(val);
    }
  }
  return blockLiveIns;
}

static BlockValues getPipelineOuts(mlir::FuncOp f, BlockValues blockLiveOuts) {
  for (Block &block : f) {
    Operation *termOp = block.getTerminator();
    if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(termOp)) {
      for (auto val : condBranchOp.getTrueOperands())
        blockLiveOuts[&block].push_back(val);
      for (auto val : condBranchOp.getFalseOperands())
        blockLiveOuts[&block].push_back(val);
    } else {
      for (auto val : termOp->getOperands())
        blockLiveOuts[&block].push_back(val);
    }
  }
  return blockLiveOuts;
}

static void createPipeline(mlir::FuncOp f, OpBuilder &builder) {
  BlockValues blockLiveIns = getBlockLiveIns(f);
  BlockValues blockLiveOuts = getBlockLiveOuts(f, blockLiveIns);

  BlockValues pipelineIns = getPipelineIns(f, blockLiveIns);
  BlockValues pipelineOuts = getPipelineOuts(f, blockLiveOuts);

  unsigned opIdx = 0;
  for (Block &block : f) {
    // Add block arguments.
    for (auto val : blockLiveIns[&block]) {
      auto newArg = block.addArgument(val.getType());
      val.replaceUsesWithIf(
          newArg,
          function_ref<bool(OpOperand &)>([&block](OpOperand &operand) -> bool {
            return operand.getOwner()->getBlock() == block;
          }));
    }

    // Add terminator operands.
    auto *termOp = block.getTerminator();
    if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(termOp)) {

      // Add true destination operands.
      SmallVector<Value, 8> trueOperands;
      for (auto trueOperand : condBranchOp.getTrueOperands())
        trueOperands.push_back(trueOperand);
      for (auto trueLiveOut : blockLiveIns[condBranchOp.getTrueDest()])
        trueOperands.push_back(trueLiveOut);

      // Add false destination operands.
      SmallVector<Value, 8> falseOperands;
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
      SmallVector<Value, 8> operands;
      for (auto operand : branchOp.getOperands())
        operands.push_back(operand);
      for (auto liveOut : blockLiveIns[branchOp.getDest()])
        operands.push_back(liveOut);

      builder.setInsertionPointAfter(branchOp);
      builder.create<mlir::BranchOp>(f.getLoc(), branchOp.getDest(), operands);
      branchOp.erase();
    }

    if (!block.front().isKnownTerminator()) {

      // Traverse all inputs to get their types.
      SmallVector<mlir::Type, 8> argTypes;
      for (auto &val : pipelineIns[&block]) {
        argTypes.push_back(val.getType());
      }

      // Traverse all outputs to get their types.
      SmallVector<mlir::Type, 8> resTypes;
      for (auto &val : pipelineOuts[&block]) {
        resTypes.push_back(val.getType());
      }

      auto opType = builder.getFunctionType(argTypes, resTypes);

      // Get attributes of the pipeline.
      SmallVector<NamedAttribute, 4> attributes;

      // Build pipeline operation.
      builder.setInsertionPoint(block.getTerminator());
      auto pipelineOp = builder.create<staticlogic::PipelineOp>(
          f.getLoc(), "pipeline" + std::to_string(opIdx), opType,
          ValueRange(pipelineIns[&block]), attributes);

      auto &pipelineOps = pipelineOp.getBlocks().front().getOperations();
      pipelineOps.splice(--pipelineOps.end(), block.getOperations(),
                         block.begin(), (--block.end()));

      // Create return operation and reconnect results of the pipeline.
      builder.setInsertionPoint(&pipelineOps.back());
      auto returnOp = builder.create<mlir::ReturnOp>(
          f.getLoc(), ValueRange(pipelineOuts[&block]));
      pipelineOps.back().erase();

      // Reconnect arguments of the pipeline.
      unsigned argIdx = 0;
      for (auto &arg : pipelineOp.getArguments()) {
        Value value = pipelineIns[&block][argIdx];
        value.replaceAllUsesWith(arg);
        argIdx += 1;
      }

      llvm::outs() << "new block\n";

      opIdx += 1;
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