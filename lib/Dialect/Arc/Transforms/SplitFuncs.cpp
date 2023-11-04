//===- SplitFuncs.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-split-funcs"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_SPLITFUNCS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;
using namespace func;
using mlir::OpTrait::ConstantLike;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct SplitFuncsPass : public arc::impl::SplitFuncsBase<SplitFuncsPass> {
  SplitFuncsPass() = default;
  SplitFuncsPass(const SplitFuncsPass &pass) : SplitFuncsPass() {}

  void runOnOperation() override;
  LogicalResult lowerModel(ModelOp modelOp);
  LogicalResult lowerFunc(FuncOp funcOp);

  SymbolTable *symbolTable;
  int splitBound;

  Statistic numFuncsCreated{this, "funcs-created",
                            "Number of new functions created"};

  // using SplitFuncsBase::funcSizeThreshold;
};
} // namespace

void SplitFuncsPass::runOnOperation() {
  // symbolTable = &getAnalysis<SymbolTable>();
  lowerFunc(getOperation());
  // for (auto op : getOperation().getOps<ModelOp>()) {
  //   if (failed(lowerModel(op)))
  //     return signalPassFailure();
  // }
}

LogicalResult SplitFuncsPass::lowerModel(ModelOp modelOp) {
  for (auto op : modelOp.getOps<FuncOp>()) {
    if (failed(lowerFunc(op)))
      return failure();
  }
}

LogicalResult SplitFuncsPass::lowerFunc(FuncOp funcOp) {
  int funcSizeThreshold = 2;
  int numOps =
      funcOp->getRegion(0).front().getOperations().size(); // TODO neaten this!
  int numBlocks = ceil(numOps / funcSizeThreshold);
  OpBuilder opBuilder(funcOp->getContext());
  std::vector<Block *> blocks;
  assert(funcOp->getNumRegions() == 1);
  blocks.push_back(&(funcOp->getRegion(0).front()));
  for (int i = 0; i < numBlocks - 1; i++) {
    auto *block = opBuilder.createBlock(&funcOp->getRegion(0));
    blocks.push_back(block);
  }
  int numOpsInBlock = 0;
  std::vector<Block *>::iterator blockIter = blocks.begin();
  for (auto &op : llvm::make_early_inc_range(funcOp.getOps())) {
    if (numOpsInBlock >= funcSizeThreshold) {
      blockIter++;
      numOpsInBlock = 0;
      opBuilder.setInsertionPointToEnd(*blockIter);
    }
    numOpsInBlock++;
    // Don't bother moving ops to the original block
    if (*blockIter == &(funcOp->getRegion(0).front())) {
      continue;
    }
    // Remove op from original block and insert in new block
    op.remove();
    // opBuilder.insert(&op);
    (*blockIter)->push_back(&op);
  }

  // Create funcs to contain blocks
  Liveness liveness(funcOp);
  Block *currentBlock = blocks[0];
  Block *previousBlock;
  auto liveOut = liveness.getLiveOut(currentBlock);
  auto liveIn = liveOut;
  for (int i = 1; i < blocks.size(); i++) {
    previousBlock = currentBlock;
    liveIn = liveOut;
    currentBlock = blocks[i];
    liveOut = liveness.getLiveOut(currentBlock);
    std::vector<Type> inTypes;
    llvm::for_each(liveIn,
                   [&inTypes](auto el) { inTypes.push_back(el.getType()); });
    std::vector<Type> outTypes;
    llvm::for_each(liveOut,
                   [&outTypes](auto el) { outTypes.push_back(el.getType()); });
    auto newFunc = opBuilder.create<FuncOp>(
        funcOp->getLoc(), funcOp.getName(),
        opBuilder.getFunctionType({inTypes}, {outTypes}));
    // TODO: can do without temp?
    auto *tempBlock = opBuilder.createBlock(&funcOp.getRegion());
    currentBlock->moveBefore(tempBlock);
    tempBlock->erase();
    std::vector<Value> valVec;
    int j = 0;
    for (auto el : liveOut) {
      valVec.push_back(el);
      replaceAllUsesInRegionWith(el, funcOp.getArgument(j++),
                                 newFunc.getRegion());
    }
    ValueRange vals(valVec);
    opBuilder.setInsertionPointToEnd(currentBlock);
    Operation *returnOp = opBuilder.create<ReturnOp>(funcOp->getLoc(), vals);
    opBuilder.setInsertionPointToStart(currentBlock);
    auto prevReturn = *previousBlock->getOps<ReturnOp>().begin();
    Operation *callOp = opBuilder.create<func::CallOp>(
        funcOp->getLoc(), inTypes, funcOp.getName(), vals);
  }
  return success();
}

std::unique_ptr<Pass> arc::createSplitFuncsPass() {
  return std::make_unique<SplitFuncsPass>();
}
