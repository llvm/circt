//===- CheckLayers.cpp - check layer legality -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKLAYERS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
class CheckLayers {
  CheckLayers(InstanceGraph &instanceGraph) : instanceGraph(instanceGraph) {}

  /// Walk the LayerBlock and report any illegal instantiation found within.
  void run(LayerBlockOp layerBlock) {
    layerBlock.getBody()->walk([&](FInstanceLike instance) {
      auto moduleName = instance.getReferencedModuleNameAttr();
      auto *targetModule =
          instanceGraph.lookup(moduleName)->getModule<Operation *>();
      auto childLayerBlock = layerBlocks.lookup(targetModule);
      if (childLayerBlock) {
        auto diag = emitError(instance.getLoc())
                    << "cannot instantiate " << moduleName
                    << " under a layerblock, because " << moduleName
                    << " contains a layerblock";
        diag.attachNote(childLayerBlock.getLoc()) << "layerblock here";
        error = true;
      }
    });
  }

  /// Walk a module, reporting any illegal instantation of layers under layers,
  /// and record if this module contains any layerblocks.
  void run(FModuleOp module) {
    // If this module directly contains a layerblock, or instantiates another
    // module with a layerblock, then this will point to the layerblock. We use
    // this for error reporting. We keep track of only the first layerblock
    // found.
    LayerBlockOp firstLayerblock = nullptr;
    module.getBodyBlock()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      // If we are instantiating a module, check if it contains a layerblock. If
      // it does, mark this target layerblock as our layerblock.
      if (auto instance = dyn_cast<FInstanceLike>(op)) {
        // If this is the first layer block in this module, record it.
        if (!firstLayerblock) {
          auto moduleName = instance.getReferencedModuleNameAttr();
          auto *targetModule =
              instanceGraph.lookup(moduleName)->getModule().getOperation();
          firstLayerblock = layerBlocks.lookup(targetModule);
        }
        return WalkResult::advance();
      }
      if (auto layerBlock = dyn_cast<LayerBlockOp>(op)) {
        // If this is the first layer block in this module, record it.
        if (!firstLayerblock)
          firstLayerblock = layerBlock;
        // Process the layerblock.
        run(layerBlock);
        // Don't recurse on elements of the layerblock.  If an instance within
        // did contain a layerblock, then an error would have been reported for
        // it already.
        return WalkResult::skip();
      }
      // Do nothing for all other operations.
      return WalkResult::advance();
    });
    // If this module contained a layerblock, then record it.
    if (firstLayerblock)
      layerBlocks.try_emplace(module, firstLayerblock);
  }

public:
  static LogicalResult run(InstanceGraph &instanceGraph) {
    CheckLayers checkLayers(instanceGraph);
    DenseSet<InstanceGraphNode *> visited;
    for (auto *root : instanceGraph) {
      for (auto *node : llvm::post_order_ext(root, visited)) {
        if (auto module = dyn_cast<FModuleOp>(node->getModule<Operation *>()))
          checkLayers.run(module);
      }
    }
    return failure(checkLayers.error);
  }

private:
  InstanceGraph &instanceGraph;
  /// A mapping from a module to the first layerblock that it contains,
  /// transitively through instances.
  DenseMap<Operation *, LayerBlockOp> layerBlocks;
  bool error = false;
};
} // end anonymous namespace

class CheckLayersPass
    : public circt::firrtl::impl::CheckLayersBase<CheckLayersPass> {
public:
  void runOnOperation() override {
    if (failed(CheckLayers::run(getAnalysis<InstanceGraph>())))
      return signalPassFailure();
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckLayers() {
  return std::make_unique<CheckLayersPass>();
}
