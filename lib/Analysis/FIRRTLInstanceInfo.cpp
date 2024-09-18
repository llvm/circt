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
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-analysis-instanceinfo"

using namespace circt;
using namespace firrtl;
using namespace detail;

void InstanceInfo::LatticeValue::merge(LatticeValue that) {
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

void InstanceInfo::LatticeValue::merge(bool property) {
  merge({/*kind=*/Constant, /*constant=*/property});
}

namespace {
/// This struct is one frame of a worklist.  This is used to track what instance
/// is being visited and with what information about its instantiation needs to
/// be known.
struct Frame {

  /// The instance that will be visited.
  InstanceGraphNode *node;

  /// Indicates that the current hierarchy is under the DUT.
  bool underDut = false;

  /// Indicates that the current hierarchy is under a layer.
  bool underLayer = false;
};
} // namespace

InstanceInfo::InstanceInfo(Operation *op, mlir::AnalysisManager &am) {
  auto &iGraph = am.getAnalysis<InstanceGraph>();

  SmallVector<Frame> worklist({{iGraph.getTopLevelNode()}});
  while (!worklist.empty()) {
    auto frame = worklist.pop_back_val();
    auto *node = frame.node;
    auto moduleOp = node->getModule();

    // Compute information about this instance.
    bool isDut = AnnotationSet(moduleOp).hasAnnotation(dutAnnoClass);

    // Set the baseline attributes.  This is what the attributes will be if
    // there is no other, existing attributes.
    ModuleAttributes attributes({/*isDut=*/isDut});
    attributes.underDut.merge(frame.underDut);
    attributes.underLayer.merge(frame.underLayer);

    // Merge the baseline attributes with the existing attributes, if they
    // exist.
    auto it = moduleAttributes.find(moduleOp);
    if (it == moduleAttributes.end())
      moduleAttributes[moduleOp] = attributes;
    else {
      auto &oldAttributes = it->getSecond();
      oldAttributes.underDut.merge(attributes.underDut);
      oldAttributes.underLayer.merge(attributes.underLayer);
      attributes = oldAttributes;
    }

    for (auto *inst : *frame.node) {
      auto underDut = frame.underDut || isDut;
      auto underLayer = frame.underLayer ||
                        inst->getInstance()->getParentOfType<LayerBlockOp>();
      worklist.push_back({inst->getTarget(),
                          /*underDut=*/underDut,
                          /*underLayer=*/underLayer});
    }
  }
  for (auto *node : llvm::depth_first(iGraph.getTopLevelNode())) {
    auto moduleOp = node->getModule();

    auto &attributes = moduleAttributes[moduleOp];

    if (AnnotationSet(moduleOp).hasAnnotation(dutAnnoClass)) {
      attributes.isDut = true;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "InstanceInfo Analysis Results:\n";
    DenseSet<Operation *> visited;
    for (auto *node : llvm::depth_first(iGraph.getTopLevelNode())) {
      auto moduleOp = node->getModule();
      if (!visited.insert(moduleOp.getOperation()).second)
        continue;
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

bool InstanceInfo::isDut(FModuleOp op) {
  return getModuleAttributes(op).isDut;
}

bool InstanceInfo::isUnderDut(FModuleOp op) {
  auto underDut = getModuleAttributes(op).underDut;
  return underDut.kind == LatticeValue::Mixed ||
         (underDut.kind == LatticeValue::Constant && underDut.constant);
}

bool InstanceInfo::isFullyUnderDut(FModuleOp op) {
  auto underDut = getModuleAttributes(op).underDut;
  return underDut.kind == LatticeValue::Constant && underDut.constant;
}

bool InstanceInfo::isUnderLayer(FModuleOp op) {
  auto underLayer = getModuleAttributes(op).underLayer;
  return underLayer.kind == LatticeValue::Mixed ||
         (underLayer.kind == LatticeValue::Constant && underLayer.constant);
}

bool InstanceInfo::isFullyUnderLayer(FModuleOp op) {
  auto underLayer = getModuleAttributes(op).underLayer;
  return underLayer.kind == LatticeValue::Constant && underLayer.constant;
}
