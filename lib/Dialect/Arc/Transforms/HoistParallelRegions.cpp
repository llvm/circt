//===- HoistParallelRegions.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

#define DEBUG_TYPE "arc-hoist-parallel-regions"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_HOISTPARALLELREGIONS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;

using llvm::MapVector;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

/// Check whether an `op` is defined outside of the given `block`.
static bool isOutsideOfBlock(Operation *op, Block *block) {
  Block *opBlock = op->getBlock();
  while (opBlock) {
    if (opBlock == block)
      return false;
    opBlock = opBlock->getParentOp()->getBlock();
  }
  return true;
}

namespace {
struct HoistParallelRegionsPass
    : public arc::impl::HoistParallelRegionsBase<HoistParallelRegionsPass> {
  using HoistParallelRegionsBase::HoistParallelRegionsBase;
  void runOnOperation() override;
  LogicalResult runOnModel(ModelOp modelOp);
};
} // namespace

void HoistParallelRegionsPass::runOnOperation() {
  for (auto op : getOperation().getOps<ModelOp>())
    if (failed(runOnModel(op)))
      return signalPassFailure();
}

LogicalResult HoistParallelRegionsPass::runOnModel(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "Hoisting parallel regions in "
                          << modelOp.getNameAttr() << "\n");

  // auto outlineIntoOmpSingle = [&](Block *source, Block::iterator firstIt,
  //                                 Block::iterator lastIt, Block *intoBlock,
  //                                 Block::iterator before) {
  //   intoBlock->getOperations().splice(before, source->getOperations(),
  //   firstIt, lastIt);
  // };

  OpBuilder stateBuilder(modelOp);
  stateBuilder.setInsertionPointToStart(&modelOp.getBodyBlock());

  // Operations that are aware of parallelism and have corresponding internal
  // `omp.single` regions.
  std::function<bool(Block *)> handleBlock = [&](Block *block) {
    bool anyParallel = false;
    Block::iterator firstIt = block->begin();
    Block::iterator currentIt = block->begin();
    while (currentIt != block->end()) {
      auto *op = &*currentIt;
      auto lastIt = currentIt;
      bool opIsParallel = false;
      if (auto parallelOp = dyn_cast<omp::ParallelOp>(op)) {
        // TODO: Inline the parallel op and mark everything inside as aware of
        // parallelism.
        LLVM_DEBUG(llvm::dbgs() << "- Inlining omp.parallel op\n");
        opIsParallel = true;
        ++currentIt;
      } else {
        for (auto &region : op->getRegions())
          for (auto &block : region)
            opIsParallel |= handleBlock(&block);
        ++currentIt;
        if (opIsParallel)
          LLVM_DEBUG(llvm::dbgs() << "- " << op->getName() << " is parallel\n");
      }
      anyParallel |= opIsParallel;
      if (opIsParallel && firstIt != lastIt) {
        LLVM_DEBUG(llvm::dbgs() << "- Moving " << std::distance(firstIt, lastIt)
                                << " ops into omp.single\n");
        OpBuilder builder(block, firstIt);
        auto singleOp = builder.create<omp::SingleOp>(op->getLoc(), TypeRange{},
                                                      ValueRange{});
        auto &singleBlock = singleOp.getRegion().emplaceBlock();
        singleBlock.getOperations().splice(
            singleBlock.end(), block->getOperations(), firstIt, lastIt);
        builder.setInsertionPointToEnd(&singleBlock);

        for (auto &op : llvm::make_early_inc_range(singleBlock)) {
          if (isa<LLVM::AllocaOp, AllocStateOp>(&op) ||
              op.hasTrait<OpTrait::ConstantLike>()) {
            op.remove();
            stateBuilder.insert(&op);
            continue;
          }
          for (auto result : op.getResults()) {
            Value storage;
            for (auto &use : llvm::make_early_inc_range(result.getUses())) {
              if (!isOutsideOfBlock(use.getOwner(), &singleBlock))
                continue;
              OpBuilder loadBuilder(use.getOwner());
              if (op.getNumResults() == 1 &&
                  isa<StorageType>(result.getType())) {
                auto *clonedOp = loadBuilder.clone(op);
                use.set(clonedOp->getResult(0));
                continue;
              }
              if (!storage) {
                storage = stateBuilder.create<LLVM::AllocaOp>(
                    result.getLoc(),
                    LLVM::LLVMPointerType::get(builder.getContext()),
                    result.getType(),
                    stateBuilder.create<hw::ConstantOp>(
                        result.getLoc(), builder.getI32Type(), 1));
                builder.create<LLVM::StoreOp>(result.getLoc(), result, storage);
              }
              auto loadOp = loadBuilder.create<LLVM::LoadOp>(
                  result.getLoc(), result.getType(), storage);
              use.set(loadOp);
            }
          }
        }

        builder.create<omp::TerminatorOp>(singleOp.getLoc(), ValueRange{});
        firstIt = currentIt;
      }
    }
    return anyParallel;
  };
  handleBlock(&modelOp.getBodyBlock());

  return success();
}
