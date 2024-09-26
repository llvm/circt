//===- CheckLayers.cpp - check layer legality -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/InstanceGraphInterface.h"
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
using namespace mlir;

namespace {
class CheckLayers {
  CheckLayers(InstanceGraph &instanceGraph, InstanceInfo &instanceInfo)
      : instanceGraph(instanceGraph), instanceInfo(instanceInfo) {}

  /// Walk a module, reporting any illegal instantation of layers under layers,
  /// and record if this module contains any layerblocks.
  void run(FModuleOp moduleOp) {
    // No instance is under a layer block.  No further examination is necessary.
    if (!instanceInfo.anyInstanceUnderLayer(moduleOp))
      return;

    // The module is under a layer block.  Verify that it has no layer blocks.
    LayerBlockOp layerBlockOp;
    moduleOp.getBodyBlock()->walk([&](LayerBlockOp op) {
      layerBlockOp = op;
      return WalkResult::interrupt();
    });
    if (!layerBlockOp)
      return;

    // The module contains layer blocks and is instantiated under a layer block.
    // Walk up the instance hierarchy to find the first instance which is
    // directly under a layer block.
    SmallVector<igraph::ModuleOpInterface> worklist({moduleOp});
    while (!worklist.empty()) {
      auto current = worklist.pop_back_val();
      for (auto *instNode : instanceGraph.lookup(current)->uses()) {
        auto instanceOp = instNode->getInstance();
        auto parentModuleOp = instNode->getParent()->getModule();
        if (instanceOp->getParentOfType<LayerBlockOp>()) {
          auto moduleName = current.getModuleNameAttr();
          auto diag = emitError(instanceOp.getLoc())
                      << "cannot instantiate " << moduleName
                      << " under a layerblock, because " << moduleName
                      << " contains a layerblock";
          diag.attachNote(layerBlockOp->getLoc()) << "layerblock here";
          continue;
        }
        if (instanceInfo.anyInstanceUnderLayer(parentModuleOp))
          worklist.push_back(parentModuleOp);
      }
    }
  }

public:
  static LogicalResult run(InstanceGraph &instanceGraph,
                           InstanceInfo &instanceInfo) {
    CheckLayers checkLayers(instanceGraph, instanceInfo);
    DenseSet<InstanceGraphNode *> visited;
    for (auto *node : instanceGraph) {
      if (auto moduleOp = dyn_cast<FModuleOp>(node->getModule<Operation *>()))
        checkLayers.run(moduleOp);
    }
    return failure(checkLayers.error);
  }

private:
  InstanceGraph &instanceGraph;
  InstanceInfo &instanceInfo;
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
    if (failed(CheckLayers::run(getAnalysis<InstanceGraph>(),
                                getAnalysis<InstanceInfo>())))
      return signalPassFailure();
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<Pass> circt::firrtl::createCheckLayers() {
  return std::make_unique<CheckLayersPass>();
}
