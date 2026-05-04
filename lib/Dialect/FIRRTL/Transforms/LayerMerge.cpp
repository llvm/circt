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
#include "circt/Dialect/FIRRTL/LayerSet.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
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

  // Track the last layer block in a module associated with a specific layer.
  llvm::DenseMap<SymbolRefAttr, LayerBlockOp> lastBlockMap;

  // Recursively visit LayerBlockOps in reverse order.  Merge all earlier layer
  // blocks into the last layer blocks of the same layer definition.
  //
  // The recursive walk will cause nested layer blocks to also be merged.
  auto moduleOp = getOperation();
  LayerSet enabledLayers;
  enabledLayers.insert_range(
      moduleOp.getLayersAttr().getAsRange<SymbolRefAttr>());

  mlir::IRRewriter rewriter(moduleOp.getContext());
  moduleOp.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](LayerBlockOp thisBlock) {
        auto layer = thisBlock.getLayerName();
        // If this layer is enabled, simply inline it.
        if (isLayerCompatibleWith(layer, enabledLayers)) {
          rewriter.inlineBlockBefore(thisBlock.getBody(), thisBlock->getBlock(),
                                     Block::iterator(thisBlock));
          thisBlock->erase();
          numMerged++;
          return WalkResult::advance();
        }

        // If we haven't seen this block before, then it is the last block.
        auto [item, inserted] = lastBlockMap.try_emplace(layer, thisBlock);
        if (inserted)
          return WalkResult::advance();

        auto &lastBlock = item->getSecond();
        rewriter.inlineBlockBefore(thisBlock.getBody(), lastBlock.getBody(),
                                   lastBlock.getBody()->begin());
        thisBlock->erase();
        numMerged++;
        return WalkResult::advance();
      });
}
