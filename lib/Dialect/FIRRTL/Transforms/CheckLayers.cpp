//===- CheckLayers.cpp - check layer legality -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/InstanceGraphInterface.h"
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
      : iGraph(instanceGraph), iInfo(instanceInfo) {}

  /// Walk a module and record any illegal layerblocks/Grand Central companions
  /// under layerblocks/Grand Central companions.  This function should be run
  /// on children before parents for accurate reporting.
  void run(FModuleOp moduleOp) {

    // The module is _never_ instantiated under a layer.  There is nothing to do
    // because erroneous instantiations are reported when examining the module.
    // Note: Grand Central companions are under a layer (because InstanceInfo
    // uses the inclusive definition of "under" to be consistent with how the
    // design-under-test module is "under" the design).
    if (!iInfo.anyInstanceUnderLayer(moduleOp))
      return;

    // Check if this module has any layerblock ops.  If these exist, then these
    // may be errors.
    SmallVector<Operation *> layerBlockOps;
    moduleOp->walk([&](LayerBlockOp layerBlockOp) {
      layerBlockOps.push_back(layerBlockOp);
    });

    bool isGCCompanion =
        AnnotationSet::hasAnnotation(moduleOp, companionAnnoClass);

    // Both Grand Central copmanions and modules that transitively instantiate
    // layerblocks/Grand Central companions require analysis of their
    // instantiation sites.  However, if this is a normal module instantiated
    // under a layer and it contains no layerblocks, then early exit to avoid
    // unnecessarily examining instantiation sites.
    if (!isGCCompanion && !transitiveModules.contains(moduleOp) &&
        layerBlockOps.empty())
      return;

    // Record instantiations of this module under layerblocks or modules that
    // are under layer blocks.  Update transitive modules.
    SmallVector<Operation *> instUnderLayerBlock, instUnderLayerModule;
    for (auto *instNode : iGraph.lookup(moduleOp)->uses()) {
      auto *instOp = instNode->getInstance().getOperation();
      if (instOp->getParentOfType<LayerBlockOp>())
        instUnderLayerBlock.push_back(instOp);
      else if (auto parent = instOp->getParentOfType<FModuleOp>();
               iInfo.anyInstanceUnderLayer(parent)) {
        transitiveModules.insert(parent);
        instUnderLayerModule.push_back(instOp);
      }
    }

    // The module _may_ contain no errors if it is a Grand Central companion or
    // a transitive module.  Do a final check to ensure that an error exists.
    if (layerBlockOps.empty() && instUnderLayerBlock.empty() &&
        instUnderLayerModule.empty())
      return;

    // Record that an error occurred and print out an error message on the
    // module with notes for more information.
    error = true;
    auto diag = moduleOp->emitOpError();
    if (isGCCompanion)
      diag
          << "is a Grand Central companion that either contains layerblocks or";

    else
      diag << "either contains layerblocks or";
    diag << " has at least one instance that is or contains a Grand Central "
            "companion or layerblocks";

    for (auto *layerBlockOp : layerBlockOps)
      diag.attachNote(layerBlockOp->getLoc()) << "illegal layerblock here";
    for (auto *instUnderLayerBlock : instUnderLayerBlock)
      diag.attachNote(instUnderLayerBlock->getLoc())
          << "illegal instantiation under a layerblock here";
    for (auto *instUnderLayerModule : instUnderLayerModule)
      diag.attachNote(instUnderLayerModule->getLoc())
          << "illegal instantiation in a module under a layer here";
  }

public:
  static LogicalResult run(InstanceGraph &instanceGraph,
                           InstanceInfo &instanceInfo) {
    CheckLayers checkLayers(instanceGraph, instanceInfo);
    instanceGraph.walkPostOrder([&](auto &node) {
      if (auto moduleOp = dyn_cast<FModuleOp>(*node.getModule()))
        checkLayers.run(moduleOp);
    });
    return failure(checkLayers.error);
  }

private:
  /// Pre-populated analyses
  InstanceGraph &iGraph;
  InstanceInfo &iInfo;

  /// A module whose instances (transitively) contain layerblocks or Grand
  /// Central companions.  This is used so that every illegal instantiation can
  /// be reported.  This is populated by `run` and requires child modules to be
  /// visited before parents.
  DenseSet<Operation *> transitiveModules;

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
