//===- HWInstanceGraph.cpp - Instance Graph ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"

using namespace circt;
using namespace hw;

InstanceGraph::InstanceGraph(Operation *operation)
    : igraph::InstanceGraph(operation) {
  for (auto &node : nodes) {
    // Note: we dyn_cast here because we cannot assume that _all_ nodes are
    // HWModuleLike - there may be cases where hw.module's are mixed with
    // other ops that implement the igraph::ModuleOpInterface.
    auto hwModuleLikeNode =
        dyn_cast<HWModuleLike>(node.getModule().getOperation());
    if (hwModuleLikeNode && hwModuleLikeNode.isPublic())
      entry.addInstance({}, &node);
  }
}

igraph::InstanceGraphNode *InstanceGraph::addHWModule(HWModuleLike module) {
  auto *node = igraph::InstanceGraph::addModule(
      cast<igraph::ModuleOpInterface>(module.getOperation()));
  if (module.isPublic())
    entry.addInstance({}, node);
  return node;
}

void InstanceGraph::erase(igraph::InstanceGraphNode *node) {
  for (auto *instance : llvm::make_early_inc_range(entry)) {
    if (instance->getTarget() == node)
      instance->erase();
  }
  igraph::InstanceGraph::erase(node);
}
