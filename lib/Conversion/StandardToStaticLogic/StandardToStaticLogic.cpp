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

static void createPipeline(mlir::FuncOp f, OpBuilder &builder) {
  for (Block &block : f) {
    if (block.front().isKnownNonTerminator()) {

      // Get values that are required to be returned by the pipeline, and then
      // create the return operation.
      SmallVector<Value, 4> values;
      SmallVector<Type, 4> types;
      for (auto &op : block) {
        for (auto result : op.getResults()) {
          bool isLiveOut = false;
          for (auto user : result.getUsers()) {
            if (user->getBlock() != &block || user->isKnownTerminator()) {
              isLiveOut = true;
              break;
            }
          }
          if (isLiveOut) {
            values.push_back(result);
            types.push_back(result.getType());
          }
        }
      }
      builder.setInsertionPoint(&block.back());
      builder.create<staticlogic::ReturnOp>(f.getLoc(), ValueRange(values));

      // Create pipeline operation, and move all operations except terminator
      // into the pipeline.
      builder.setInsertionPoint(&block.front());
      auto pipeline =
          builder.create<staticlogic::PipelineOp>(f.getLoc(), types);

      auto &body = pipeline.getRegion().front();
      body.getOperations().splice(body.getOperations().begin(),
                                  block.getOperations(), ++block.begin(),
                                  --block.end());

      // Reconnect uses of the pipeline operation.
      unsigned returnIdx = 0;
      for (auto value : values) {
        value.replaceUsesWithIf(
            pipeline.getResult(returnIdx),
            function_ref<bool(OpOperand &)>([&body](OpOperand &use) -> bool {
              return use.getOwner()->getBlock() != &body;
            }));
        returnIdx += 1;
      }
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
