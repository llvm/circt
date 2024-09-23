//===- FIRRTLInstanceInfo.cpp - Instance info analysis ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InstanceInfo analysis.  This is an analysis that
// depends on the InstanceGraph analysis, but provides additional information
// about FIRRTL operations.  This is useful if you find yourself needing to
// selectively iterate over parts of the design.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#ifndef NDEBUG
#include "llvm/ADT/DepthFirstIterator.h"
#endif

#define DEBUG_TYPE "firrtl-analysis-instanceinfo"

using namespace circt;
using namespace firrtl;

void InstanceInfo::LatticeValue::mergeIn(LatticeValue that) {
  if (kind > that.kind)
    return;

  if (kind < that.kind) {
    kind = that.kind;
    constant = that.constant;
    return;
  }

  if (kind == Constant && constant != that.constant)
    kind = Mixed;
}

InstanceInfo::InstanceInfo(Operation *op, mlir::AnalysisManager &am) {
  auto &iGraph = am.getAnalysis<InstanceGraph>();

  // Visit modules in reverse post-order (visit parents before children) because
  // information flows in this direction---the attributes of modules are
  // determinend by their instantiations.
  DenseSet<InstanceGraphNode *> visited;
  for (auto *root : iGraph) {
    for (auto *modIt : llvm::inverse_post_order_ext(root, visited)) {
      auto moduleOp = modIt->getModule();

      // Set baseline attributes for this module.
      ModuleAttributes attributes(
          {/*isDut=*/AnnotationSet(moduleOp).hasAnnotation(dutAnnoClass)});

      // Merge in attributes from instantiations one-by-one.
      for (auto *useIt : modIt->uses()) {
        auto parentAttrs =
            moduleAttributes.find(useIt->getParent()->getModule())->getSecond();
        // Merge underDut.
        if (parentAttrs.isDut || attributes.isDut)
          attributes.underDut.mergeIn({LatticeValue::Constant, true});
        else
          attributes.underDut.mergeIn(parentAttrs.underDut);
        // Merge underLayer.
        if (useIt->getInstance()->getParentOfType<LayerBlockOp>())
          attributes.underLayer.mergeIn({LatticeValue::Constant, true});
        else
          attributes.underLayer.mergeIn(parentAttrs.underLayer);
      }

      // If attributes are unknown at this point, set default values.
      if (attributes.underDut.kind == LatticeValue::Unknown)
        attributes.underDut.mergeIn({LatticeValue::Constant, false});
      if (attributes.underLayer.kind == LatticeValue::Unknown)
        attributes.underLayer.mergeIn({LatticeValue::Constant, false});

      // Record the attributes for the module.
      moduleAttributes[moduleOp] = attributes;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "InstanceInfo Analysis Results:\n";
    for (auto *node : llvm::depth_first(iGraph.getTopLevelNode())) {
      auto moduleOp = node->getModule();
      auto attributes = moduleAttributes[moduleOp];
      llvm::dbgs() << "  - module: " << moduleOp.getModuleName() << "\n"
                   << "    isDut:      "
                   << (attributes.isDut ? "true" : "false") << "\n"
                   << "    underDut:   " << attributes.underDut << "\n"
                   << "    underLayer: " << attributes.underLayer << "\n";
    }
  });
}

const InstanceInfo::ModuleAttributes &
InstanceInfo::getModuleAttributes(FModuleOp op) {
  return moduleAttributes.find(op)->getSecond();
}

bool InstanceInfo::isDut(FModuleOp op) { return getModuleAttributes(op).isDut; }

bool InstanceInfo::atLeastOneInstanceUnderDut(FModuleOp op) {
  return getModuleAttributes(op).underDut.kind == LatticeValue::Mixed ||
         allInstancesUnderDut(op);
}

bool InstanceInfo::allInstancesUnderDut(FModuleOp op) {
  auto underDut = getModuleAttributes(op).underDut;
  return underDut.kind == LatticeValue::Constant && underDut.constant;
}

bool InstanceInfo::atLeastOneInstanceUnderLayer(FModuleOp op) {
  return getModuleAttributes(op).underLayer.kind == LatticeValue::Mixed ||
         allInstancesUnderLayer(op);
}

bool InstanceInfo::allInstancesUnderLayer(FModuleOp op) {
  auto underLayer = getModuleAttributes(op).underLayer;
  return underLayer.kind == LatticeValue::Constant && underLayer.constant;
}
