//===- StandardToStaticLogic.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Standard to StaticLogic Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/StandardToStaticLogic.h"
#include "../PassDetail.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace circt;
using namespace staticlogic;
using namespace std;

using valueVector = SmallVector<Value, 4>;

valueVector getPipelineArgs(Block &block) {
  valueVector arguments;
  for (auto &op : block) {
    if (!op.mightHaveTrait<OpTrait::IsTerminator>()) {
      for (auto operand : op.getOperands()) {
        if (operand.isa<BlockArgument>()) {
          // Add only unique uses
          if (std::find(arguments.begin(), arguments.end(), operand) ==
              arguments.end())
            arguments.push_back(operand);
        } else if (operand.getDefiningOp()->getBlock() != &block) {
          // Add only unique uses
          if (std::find(arguments.begin(), arguments.end(), operand) ==
              arguments.end())
            arguments.push_back(operand);
        }
      }
    }
  }
  return arguments;
}

valueVector getPipelineResults(Block &block) {
  SmallVector<Value, 4> results;
  for (auto &op : block) {
    for (auto result : op.getResults()) {
      bool isResult = false;
      for (auto user : result.getUsers()) {
        if (user->getBlock() != &block ||
            user->hasTrait<OpTrait::IsTerminator>()) {
          isResult = true;
          break;
        }
      }
      if (isResult)
        results.push_back(result);
    }
  }
  return results;
}

static void createPipeline(mlir::FuncOp f, OpBuilder &builder) {
  for (Block &block : f) {
    if (!block.front().mightHaveTrait<OpTrait::IsTerminator>()) {

      auto arguments = getPipelineArgs(block);
      auto results = getPipelineResults(block);
      builder.setInsertionPoint(&block.back());
      builder.create<staticlogic::ReturnOp>(f.getLoc(), ValueRange(results));

      // Create pipeline operation, and move all operations except terminator
      // into the pipeline.
      builder.setInsertionPoint(&block.front());
      auto pipeline = builder.create<staticlogic::PipelineOp>(
          f.getLoc(), ValueRange(arguments), ValueRange(results));

      auto &body = pipeline.getRegion().front();
      body.getOperations().splice(body.getOperations().begin(),
                                  block.getOperations(), ++block.begin(),
                                  --block.end());

      // Reconnect arguments of the pipeline operation.
      unsigned argIdx = 0;
      for (auto value : arguments) {
        value.replaceUsesWithIf(
            body.getArgument(argIdx),
            function_ref<bool(OpOperand &)>([&body](OpOperand &use) -> bool {
              return use.getOwner()->getBlock() == &body;
            }));
        argIdx += 1;
      }

      // Reconnect results of the pipeline operation.
      unsigned resultIdx = 0;
      for (auto value : results) {
        value.replaceUsesWithIf(
            pipeline.getResult(resultIdx),
            function_ref<bool(OpOperand &)>([&body](OpOperand &use) -> bool {
              return use.getOwner()->getBlock() != &body;
            }));
        resultIdx += 1;
      }
    }
  }
}

namespace {

struct CreatePipelinePass : public CreatePipelineBase<CreatePipelinePass> {
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    auto builder = OpBuilder(f.getContext());
    createPipeline(f, builder);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::createCreatePipelinePass() {
  return std::make_unique<CreatePipelinePass>();
}
