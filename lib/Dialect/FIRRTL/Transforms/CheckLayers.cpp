//===- CheckLayers.cpp - check layer legality -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/LayerSet.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

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
struct InstanceUnderLayerBlock {
  FInstanceLike instance;
  LayerBlockOp parent;
};
} // namespace

namespace {
struct InstanceUnderGCCompanion {
  FInstanceLike instance;
  FModuleLike parent;
};
} // namespace

using InstancesUnderLayerBlock = SmallVector<InstanceUnderLayerBlock>;
using InstancesUnderGCCompanion = SmallVector<InstanceUnderGCCompanion>;

using InstancesUnderLayerBlockTable =
    DenseMap<StringAttr, InstancesUnderLayerBlock>;
using InstancesUnderGCCompanionTable =
    DenseMap<StringAttr, InstancesUnderGCCompanion>;

static auto targetNames(igraph::InstanceOpInterface inst) {
  return inst.getReferencedModuleNamesAttr().getAsRange<StringAttr>();
}

namespace {
struct CheckLayersInModule {
  CheckLayersInModule(
      CircuitOp circuitOp, SymbolTableCollection &stc, FModuleLike moduleOp,
      InstancesUnderLayerBlockTable &instancesUnderLayerBlocksTable,
      InstancesUnderGCCompanionTable &instancesUnderGCCompanionsTable)
      : circuitOp(circuitOp), stc(stc), moduleOp(moduleOp),
        instancesUnderLayerBlocksTable(instancesUnderLayerBlocksTable),
        instancesUnderGCCompanionsTable(instancesUnderGCCompanionsTable) {
    auto name = moduleOp.getModuleNameAttr();
    instancesUnderLayerBlocks = instancesUnderLayerBlocksTable[name];
    instancesUnderGCCompanions = instancesUnderGCCompanionsTable[name];
    isCompanion = AnnotationSet::hasAnnotation(moduleOp, companionAnnoClass);
  }

  void noteChildLayerBlocks(InFlightDiagnostic &diag,
                            const SmallVector<LayerBlockOp> &childLayerBlocks) {
    for (auto child : childLayerBlocks)
      diag.attachNote(child.getLoc()) << "illegal layerblock here";
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void
  noteInstancesUnderLayerBlocks(InFlightDiagnostic &diag,
                                const InstancesUnderLayerBlock &instances) {
    for (auto entry : instances) {
      diag.attachNote(entry.instance.getLoc())
          << "illegal instantiation under a layerblock here";

      // If the instance is directly under a layerblock, point it out. Otherwise,
      // recursively report the chain of layers under a layerblock.
      if (entry.parent) {
        diag.attachNote(entry.parent.getLoc())
            << "enclosing bound layerblock here";
      } else {
        auto parent = entry.instance->getParentOfType<FModuleLike>();
        noteInstancesUnderLayerBlocks(diag, parent.getModuleNameAttr());
      }
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void noteInstancesUnderLayerBlocks(InFlightDiagnostic &diag,
                                     StringAttr target) {
    if (instancesUnderLayerBlocksTable.contains(target)) {
      auto &instances = instancesUnderLayerBlocksTable.at(target);
      noteInstancesUnderLayerBlocks(diag, instances);
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void
  noteInstancesUnderGCCompanions(InFlightDiagnostic &diag,
                                 const InstancesUnderGCCompanion &instances) {
    for (auto entry : instances) {
      diag.attachNote(entry.instance.getLoc())
          << "illegal instantiation under a grand central companion here";

      if (entry.parent) {
        diag.attachNote(entry.parent.getLoc())
            << "grand central companion parent module defined here";
      } else {
        auto parent = entry.instance->getParentOfType<FModuleLike>();
        noteInstancesUnderLayerBlocks(diag, parent.getModuleNameAttr());
      }
    }
  }

  void noteInstancesUnderGCCompanions(InFlightDiagnostic &diag, StringAttr target) {
    if (instancesUnderGCCompanionsTable.contains(target)) {
      auto &instances = instancesUnderGCCompanionsTable.at(target);
      noteInstancesUnderGCCompanions(diag, instances);
    }
  }

  void processOp(const LayerSet &enabled, LayerBlockOp parentLayerBlock,
                 FInstanceLike instance) {
    if (parentLayerBlock || !instancesUnderLayerBlocks.empty())
      for (auto target : targetNames(instance))
        instancesUnderLayerBlocksTable[target].push_back(
            {instance, parentLayerBlock});

    auto parentCompanion = isCompanion ? moduleOp : nullptr;
    if (parentCompanion || !instancesUnderGCCompanions.empty())
      for (auto target : targetNames(instance))
        instancesUnderGCCompanionsTable[target].push_back(
            {instance, parentCompanion});
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void processOp(const LayerSet &enabled, LayerBlockOp parentLayerBlock,
                 LayerBlockOp layerBlock) {
    // If this layer is hard-enabled according to the ambient layers, or if it
    // is an inline layerblock, then we don't treat it as a layerblock for the
    // purposes of detecting bind-under-bind.
    auto layerOp =
        cast<LayerOp>(stc.lookupSymbolIn(circuitOp, layerBlock.getLayerName()));
    auto convention = layerOp.getConvention();
    if (!isLayerCompatibleWith(layerBlock.getLayerNameAttr(), enabled) &&
        convention == LayerConvention::Bind) {
      parentLayerBlock = layerBlock;
      childLayerBlocks.push_back(layerBlock);
    }

    for (auto &op : *layerBlock.getBody())
      processOp(enabled, parentLayerBlock, &op);
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void processOp(const LayerSet &enabled, LayerBlockOp parent, Operation *op) {
    if (auto instance = dyn_cast<FInstanceLike>(op))
      return processOp(enabled, parent, instance);

    if (auto layerBlock = dyn_cast<LayerBlockOp>(op))
      return processOp(enabled, parent, layerBlock);

    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto &op : block)
          processOp(enabled, parent, &op);
  }

  void processModule() {
    LayerSet enabled = getAmbientLayersAt(moduleOp);
    for (auto &region : moduleOp.getOperation()->getRegions())
      for (auto &block : region)
        for (auto &op : block)
          processOp(enabled, nullptr, &op);
  }

  // Detect bind-under-bind errors in this module.
  LogicalResult checkModule() {
    bool failed = false;

    // If there are any instances of this module under a layerblock, and there
    // is a layerblock in this module, error.
    if (!instancesUnderLayerBlocks.empty() && !childLayerBlocks.empty()) {
      auto diag = moduleOp.emitError();
      diag << "module contains bound layer blocks and is instantiated "
              "under a bound layer block";
      noteInstancesUnderLayerBlocks(diag, instancesUnderLayerBlocks);
      noteChildLayerBlocks(diag, childLayerBlocks);
      failed = true;
    }

    // If there are any instances of this module under a GC companion, and there
    // is a layerblock in this module, error.
    if (!instancesUnderGCCompanions.empty() && !childLayerBlocks.empty()) {
      auto diag = moduleOp.emitError();
      diag << "module contains bound layerblocks and is instantiated "
              "under a Grand Central companion";
      noteInstancesUnderGCCompanions(diag, instancesUnderGCCompanions);
      noteChildLayerBlocks(diag, childLayerBlocks);
      failed = true;
    }

    // Handle "this module is a GC companion" case.
    if (isCompanion) {
      // This module cannot be instantiated under a layerblock.
      if (!instancesUnderLayerBlocks.empty()) {
        auto diag = moduleOp.emitError();
        diag << "Grand Central companion is instantiated under a bound "
                "layerblock";
        noteInstancesUnderLayerBlocks(diag, instancesUnderLayerBlocks);
        failed = true;
      }

      // This module cannot be instantiated under another GC companion.
      if (!instancesUnderGCCompanions.empty()) {
        auto diag = moduleOp.emitError();
        diag << "Grand Central companion is instantiated under another Grand "
                "Central companion";
        noteInstancesUnderGCCompanions(diag, instancesUnderGCCompanions);
        failed = true;
      }

      // This module cannot contain any layerblocks.
      if (!childLayerBlocks.empty()) {
        auto diag = moduleOp.emitError();
        diag << "Grand Central companion contains bound layerblocks";
        noteChildLayerBlocks(diag, childLayerBlocks);
        failed = true;
      }
    }

    return failure(failed);
  }

  LogicalResult run() {
    processModule();
    return checkModule();
  }

  CircuitOp circuitOp;
  SymbolTableCollection &stc;
  FModuleLike moduleOp;
  InstancesUnderLayerBlock instancesUnderLayerBlocks;
  InstancesUnderGCCompanion instancesUnderGCCompanions;
  InstancesUnderLayerBlockTable &instancesUnderLayerBlocksTable;
  InstancesUnderGCCompanionTable &instancesUnderGCCompanionsTable;
  SmallVector<LayerBlockOp> childLayerBlocks;
  bool isCompanion = false;
};
} // namespace

namespace {
class CheckLayersPass
    : public circt::firrtl::impl::CheckLayersBase<CheckLayersPass> {
public:
  void runOnOperation() override {
    auto &ig = getAnalysis<InstanceGraph>();
    SymbolTableCollection stc;
    InstancesUnderLayerBlockTable instancesUnderLayerBlocksTable;
    InstancesUnderGCCompanionTable instancesUnderGCCompanionsTable;
    ig.walkInversePostOrder([&](InstanceGraphNode &node) -> void {
      if (failed(CheckLayersInModule(getOperation(), stc,
                                     node.getModule<FModuleLike>(),
                                     instancesUnderLayerBlocksTable,
                                     instancesUnderGCCompanionsTable)
                     .run()))
        signalPassFailure();
    });
    markAllAnalysesPreserved();
  }
};
} // namespace
