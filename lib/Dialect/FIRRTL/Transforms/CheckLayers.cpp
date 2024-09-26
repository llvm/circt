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
#include "llvm/ADT/DepthFirstIterator.h"

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
      : iGraph(instanceGraph), iInfo(instanceInfo) {}

  /// Walk a module, reporting any illegal instantation of layers under layers,
  /// and record if this module contains any layerblocks.
  void run(FModuleOp moduleOp) {
    // No instance is under a layer block.  No further examination is necessary.
    if (!iInfo.anyInstanceUnderLayer(moduleOp))
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
    error = true;
    for (auto *node : llvm::inverse_depth_first(iGraph.lookup(moduleOp))) {
      auto modOp = node->getModule();
      if (previousErrors.contains(node) || !iInfo.anyInstanceUnderLayer(modOp))
        continue;
      for (auto *instNode : node->uses()) {
        auto instanceOp = instNode->getInstance();
        if (instanceOp->getParentOfType<LayerBlockOp>()) {
          auto moduleName = modOp.getModuleNameAttr();
          auto diag = emitError(instanceOp.getLoc())
                      << "cannot instantiate " << moduleName
                      << " under a layerblock, because " << moduleName
                      << " contains a layerblock";
          diag.attachNote(layerBlockOp->getLoc()) << "layerblock here";
          previousErrors.insert(node);
          continue;
        }
      }
    }
  }

public:
  static LogicalResult run(InstanceGraph &instanceGraph,
                           InstanceInfo &instanceInfo) {
    CheckLayers checkLayers(instanceGraph, instanceInfo);
    for (auto *node : instanceGraph)
      if (auto moduleOp = dyn_cast<FModuleOp>(node->getModule<Operation *>()))
        checkLayers.run(moduleOp);
    return failure(checkLayers.error);
  }

private:
  /// Pre-populated analyses
  InstanceGraph &iGraph;
  InstanceInfo &iInfo;

  /// This records modules for which we have already generated errors when doing
  /// a top-down walk.
  DenseSet<const InstanceGraphNode *> previousErrors;

  /// Indicates if this checker found an error.
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
