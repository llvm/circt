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
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Debug.h"
#include "circt/Support/InstanceGraph.h"
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

void InstanceInfo::LatticeValue::markMixed() { kind = Mixed; }

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

InstanceInfo::LatticeValue InstanceInfo::LatticeValue::operator!() {
  if (isUnknown() || isMixed())
    return *this;

  auto invert = LatticeValue();
  invert.markConstant(!getConstant());
  return invert;
}

InstanceInfo::InstanceInfo(Operation *op, mlir::AnalysisManager &am) {
  auto &iGraph = am.getAnalysis<InstanceGraph>();

  // Setup circuit attributes based on presence of annotations.
  circuitAttributes.effectiveDut = iGraph.getTopLevelNode()->getModule();
  for (auto *node : iGraph) {
    auto moduleOp = node->getModule();
    AnnotationSet annotations(moduleOp);
    if (annotations.hasAnnotation(dutAnnoClass)) {
      circuitAttributes.dut = moduleOp;
      circuitAttributes.effectiveDut = moduleOp;
    }
  }

  // Setup boundary conditions for any modules without users.
  for (auto *node : iGraph) {
    if (!node->noUses())
      continue;

    auto moduleOp = node->getModule();
    ModuleAttributes &attributes = moduleAttributes[moduleOp];

    attributes.underDut.mergeIn(isDut(moduleOp));
    attributes.inDesign.mergeIn(isDut(moduleOp));
    attributes.inEffectiveDesign.mergeIn(isEffectiveDut(moduleOp) || !hasDut());
    attributes.underLayer.mergeIn(false);
  }

  // Visit modules in reverse post-order (visit parents before children) to
  // merge parent attributes and per-instance attributes into children.
  DenseSet<InstanceGraphNode *> visited;
  for (auto *root : iGraph) {
    for (auto *modIt : llvm::inverse_post_order_ext(root, visited)) {
      visited.insert(modIt);
      auto moduleOp = modIt->getModule();
      ModuleAttributes &attributes = moduleAttributes[moduleOp];

      AnnotationSet annotations(moduleOp);
      auto isDut = annotations.hasAnnotation(dutAnnoClass);
      auto isGCCompanion = annotations.hasAnnotation(companionAnnoClass);

      if (isDut) {
        attributes.underDut.markConstant(true);
        attributes.inDesign.markConstant(true);
        attributes.inEffectiveDesign.markConstant(true);
      }

      if (isGCCompanion) {
        attributes.inDesign.mergeIn(false);
        attributes.inEffectiveDesign.mergeIn(false);
        attributes.underLayer.mergeIn(true);
      }

      // Merge in values based on the instantiations of this module.
      for (auto *useIt : modIt->uses()) {
        auto parentOp = useIt->getParent()->getModule();
        auto parentAttrs = moduleAttributes.find(parentOp)->getSecond();

        // Update underDut.
        if (!isDut)
          attributes.underDut.mergeIn(parentAttrs.underDut);

        // Update underLayer.
        bool underLayer = false;
        if (auto instanceOp = useIt->getInstance<InstanceOp>()) {
          if (instanceOp.getLowerToBind() || instanceOp.getDoNotPrint() ||
              instanceOp->getParentOfType<LayerBlockOp>() ||
              instanceOp->getParentOfType<sv::IfDefOp>())
            underLayer = true;
        }

        if (!isGCCompanion) {
          if (underLayer)
            attributes.underLayer.mergeIn(true);
          else
            attributes.underLayer.mergeIn(parentAttrs.underLayer);
        }

        // Update inDesign and inEffectiveDesign.
        if (underLayer) {
          attributes.inDesign.mergeIn(false);
          attributes.inEffectiveDesign.mergeIn(false);
        } else if (!isDut && !isGCCompanion) {
          attributes.inDesign.mergeIn(parentAttrs.inDesign);
          attributes.inEffectiveDesign.mergeIn(parentAttrs.inEffectiveDesign);
        }
      }
    }
  }

  LLVM_DEBUG({
    mlir::OpPrintingFlags flags;
    flags.skipRegions();
    debugHeader("FIRRTL InstanceInfo Analysis")
        << "\n"
        << llvm::indent(2) << "circuit attributes:\n"
        << llvm::indent(4) << "hasDut: " << (hasDut() ? "true" : "false")
        << "\n"
        << llvm::indent(4) << "dut: ";
    if (auto dut = circuitAttributes.dut)
      dut->print(llvm::dbgs(), flags);
    else
      llvm::dbgs() << "null";
    llvm::dbgs() << "\n" << llvm::indent(4) << "effectiveDut: ";
    circuitAttributes.effectiveDut->print(llvm::dbgs(), flags);
    llvm::dbgs() << "\n" << llvm::indent(2) << "module attributes:\n";
    visited.clear();
    for (auto *root : iGraph) {
      for (auto *modIt : llvm::inverse_post_order_ext(root, visited)) {
        visited.insert(modIt);
        auto moduleOp = modIt->getModule();
        auto attributes = moduleAttributes[moduleOp];
        llvm::dbgs().indent(4)
            << "- module: " << moduleOp.getModuleName() << "\n"
            << llvm::indent(6)
            << "isDut: " << (isDut(moduleOp) ? "true" : "false") << "\n"
            << llvm::indent(6) << "isEffectiveDue: "
            << (isEffectiveDut(moduleOp) ? "true" : "false") << "\n"
            << llvm::indent(6) << "underDut: " << attributes.underDut << "\n"
            << llvm::indent(6) << "underLayer: " << attributes.underLayer
            << "\n"
            << llvm::indent(6) << "inDesign: " << attributes.inDesign << "\n"
            << llvm::indent(6)
            << "inEffectiveDesign: " << attributes.inEffectiveDesign << "\n";
      }
    }
  });
}

const InstanceInfo::ModuleAttributes &
InstanceInfo::getModuleAttributes(igraph::ModuleOpInterface op) {
  return moduleAttributes.find(op)->getSecond();
}

bool InstanceInfo::hasDut() { return circuitAttributes.dut; }

bool InstanceInfo::isDut(igraph::ModuleOpInterface op) {
  if (hasDut())
    return op == circuitAttributes.dut;
  return false;
}

bool InstanceInfo::isEffectiveDut(igraph::ModuleOpInterface op) {
  if (hasDut())
    return isDut(op);
  return op == circuitAttributes.effectiveDut;
}

igraph::ModuleOpInterface InstanceInfo::getDut() {
  return circuitAttributes.dut;
}

igraph::ModuleOpInterface InstanceInfo::getEffectiveDut() {
  return circuitAttributes.effectiveDut;
}

bool InstanceInfo::anyInstanceUnderDut(igraph::ModuleOpInterface op) {
  auto underDut = getModuleAttributes(op).underDut;
  return underDut.isMixed() || allInstancesUnderDut(op);
}

bool InstanceInfo::allInstancesUnderDut(igraph::ModuleOpInterface op) {
  auto underDut = getModuleAttributes(op).underDut;
  return underDut.isConstant() && underDut.getConstant();
}

bool InstanceInfo::anyInstanceUnderEffectiveDut(igraph::ModuleOpInterface op) {
  return !hasDut() || anyInstanceUnderDut(op);
}

bool InstanceInfo::allInstancesUnderEffectiveDut(igraph::ModuleOpInterface op) {
  return !hasDut() || allInstancesUnderDut(op);
}

bool InstanceInfo::anyInstanceUnderLayer(igraph::ModuleOpInterface op) {
  auto underLayer = getModuleAttributes(op).underLayer;
  return underLayer.isMixed() || allInstancesUnderLayer(op);
}

bool InstanceInfo::allInstancesUnderLayer(igraph::ModuleOpInterface op) {
  auto underLayer = getModuleAttributes(op).underLayer;
  return underLayer.isConstant() && underLayer.getConstant();
}

bool InstanceInfo::anyInstanceInDesign(igraph::ModuleOpInterface op) {
  auto inDesign = getModuleAttributes(op).inDesign;
  return inDesign.isMixed() || allInstancesInDesign(op);
}

bool InstanceInfo::allInstancesInDesign(igraph::ModuleOpInterface op) {
  auto inDesign = getModuleAttributes(op).inDesign;
  return inDesign.isConstant() && inDesign.getConstant();
}

bool InstanceInfo::anyInstanceInEffectiveDesign(igraph::ModuleOpInterface op) {
  auto inEffectiveDesign = getModuleAttributes(op).inEffectiveDesign;
  return inEffectiveDesign.isMixed() || allInstancesInEffectiveDesign(op);
}

bool InstanceInfo::allInstancesInEffectiveDesign(igraph::ModuleOpInterface op) {
  auto inEffectiveDesign = getModuleAttributes(op).inEffectiveDesign;
  return inEffectiveDesign.isConstant() && inEffectiveDesign.getConstant();
}
