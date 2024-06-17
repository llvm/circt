//===- LayerMerge.cpp - Merge layer blocks together -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass merges layer blocks in a module which reference the same layer
// definition.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-layer-merge"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LAYERMERGE
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
/// A pass that merges layer blocks referencing the same layer definition.
struct LayerMerge : public circt::firrtl::impl::LayerMergeBase<LayerMerge> {
  void runOnOperation() override;
};
} // namespace

void LayerMerge::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LayerMerge "
                      "--------------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);

  // Track the last layer block that we saw which referenced a specific layer
  // definition.  Because this pass operates as a single walk of the IR, it is
  // only ever possible that there is one prior layer block that references a
  // given layer definition.
  llvm::DenseMap<SymbolRefAttr, LayerBlockOp> priorLayerBlocks;

  // Recursively walk LayerBlockOps in the module.  Whenever we see a layer
  // block, check to see if there is a prior layer block that references the
  // same declaration.  If not, this layer block becomes the prior layer block
  // and we continue.  If there is a prior layer block, then splice the prior
  // layer block's body into the beginning of this layer block and erase the
  // prior layer block.  This layer block then becomes the new prior layer
  // block.
  //
  // The recursive walk will cause nested layer blocks to also be merged.
  auto moduleOp = getOperation();
  mlir::IRRewriter rewriter(moduleOp.getContext());
  moduleOp.walk([&](LayerBlockOp layerBlock) {
    auto layerName = layerBlock.getLayerName();
    // If we haven't seen this layer before, then just insert it into
    // priorLayerBlocks.
    auto priorLayerBlockIt = priorLayerBlocks.find(layerName);
    if (priorLayerBlockIt == priorLayerBlocks.end()) {
      priorLayerBlocks[layerName] = layerBlock;
      return WalkResult::advance();
    }

    auto &priorLayerBlock = priorLayerBlockIt->getSecond();
    rewriter.inlineBlockBefore(priorLayerBlock.getBody(), layerBlock.getBody(),
                               layerBlock.getBody()->begin());
    priorLayerBlock->erase();
    priorLayerBlocks[layerName] = layerBlock;
    numMerged++;
    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLayerMergePass() {
  return std::make_unique<LayerMerge>();
}
