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

bool InstanceInfo::LatticeValue::isUnknown() const { return kind == Unknown; }

bool InstanceInfo::LatticeValue::isConstant() const { return kind == Constant; }

bool InstanceInfo::LatticeValue::isMixed() const { return kind == Mixed; }

bool InstanceInfo::LatticeValue::getConstant() const {
  assert(isConstant());
  return value;
}

void InstanceInfo::LatticeValue::markConstant(bool constant) {
  kind = Constant;
  value = constant;
}

void InstanceInfo::LatticeValue::mergeIn(LatticeValue that) {
  if (kind > that.kind)
    return;

  if (kind < that.kind) {
    kind = that.kind;
    value = that.value;
    return;
  }

  if (isConstant() && getConstant() != that.value)
    kind = Mixed;
}

void InstanceInfo::LatticeValue::mergeIn(bool value) {
  LatticeValue latticeValue;
  latticeValue.markConstant(value);
  mergeIn(latticeValue);
}

InstanceInfo::InstanceInfo(Operation *op, mlir::AnalysisManager &am) {
  auto &iGraph = am.getAnalysis<InstanceGraph>();

  // Visit modules in reverse post-order (visit parents before children) because
  // information flows in this direction---the attributes of modules are
  // determinend by their instantiations.
  llvm::ReversePostOrderTraversal<InstanceGraph *> rpo(&iGraph);
  for (auto *modIt : rpo) {
    auto moduleOp = modIt->getModule();
    ModuleAttributes &attributes = moduleAttributes[moduleOp];

    // Set isDut.
    attributes.isDut = AnnotationSet(moduleOp).hasAnnotation(dutAnnoClass);

    // If the module is not instantiated, then set attributes and early exit.
    if (modIt->noUses()) {
      attributes.underDut.markConstant(false);
      attributes.underLayer.markConstant(false);
      continue;
    }

    // Merge in attributes from modules that instantiate this module.
    for (auto *useIt : modIt->uses()) {
      auto parentAttrs =
          moduleAttributes.find(useIt->getParent()->getModule())->getSecond();
      // Merge underDut.
      if (parentAttrs.isDut || attributes.isDut)
        attributes.underDut.mergeIn(true);
      else
        attributes.underDut.mergeIn(parentAttrs.underDut);
      // Merge underLayer.
      if (useIt->getInstance()->getParentOfType<LayerBlockOp>())
        attributes.underLayer.mergeIn(true);
      else
        attributes.underLayer.mergeIn(parentAttrs.underLayer);
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
  auto underDut = getModuleAttributes(op).underDut;
  return underDut.isMixed() || allInstancesUnderDut(op);
}

bool InstanceInfo::allInstancesUnderDut(FModuleOp op) {
  auto underDut = getModuleAttributes(op).underDut;
  return underDut.isConstant() && underDut.getConstant();
}

bool InstanceInfo::atLeastOneInstanceUnderLayer(FModuleOp op) {
  auto underLayer = getModuleAttributes(op).underLayer;
  return underLayer.isMixed() || allInstancesUnderLayer(op);
}

bool InstanceInfo::allInstancesUnderLayer(FModuleOp op) {
  auto underLayer = getModuleAttributes(op).underLayer;
  return underLayer.isConstant() && underLayer.getConstant();
}
