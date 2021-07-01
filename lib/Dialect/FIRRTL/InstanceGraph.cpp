//===- InstanceGraph.cpp - Instance Graph -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/InstanceGraph.h"

using namespace circt;
using namespace firrtl;

InstanceRecord *InstanceGraphNode::recordInstance(InstanceOp instance,
                                                  InstanceGraphNode *target) {
  moduleInstances.emplace_back(instance, this, target);
  return &moduleInstances.back();
}

void InstanceGraphNode::recordUse(InstanceRecord *record) {
  moduleUses.push_back(record);
}

InstanceGraph::InstanceGraph(Operation *operation) {
  auto circuitOp = cast<CircuitOp>(operation);

  // We insert the top level module first in to the node map.  Getting the node
  // here is enough to ensure that it is the first one added.
  getOrAddNode(circuitOp.name());

  for (auto &op : *circuitOp.getBody()) {
    if (auto extModule = dyn_cast<FExtModuleOp>(op)) {
      auto *currentNode = getOrAddNode(extModule.getName());
      currentNode->module = extModule;
    }
    if (auto module = dyn_cast<FModuleOp>(op)) {
      auto *currentNode = getOrAddNode(module.getName());
      currentNode->module = module;
      // Find all instance operations in the module body.
      module.body().walk([&](InstanceOp instanceOp) {
        // Add an edge to indicate that this module instantiates the target.
        auto *targetNode = getOrAddNode(instanceOp.moduleName());
        auto *instanceRecord =
            currentNode->recordInstance(instanceOp, targetNode);
        targetNode->recordUse(instanceRecord);
      });
    }
  }
}

InstanceGraphNode *InstanceGraph::getTopLevelNode() {
  // The graph always puts the top level module in the array first.
  if (!nodes.size())
    return nullptr;
  return &nodes[0];
}

InstanceGraphNode *InstanceGraph::lookup(StringRef name) {
  auto it = nodeMap.find(name);
  assert(it != nodeMap.end() && "Module not in InstanceGraph!");
  return &nodes[it->second];
}

InstanceGraphNode *InstanceGraph::lookup(Operation *op) {
  if (auto extModule = dyn_cast<FExtModuleOp>(op)) {
    return lookup(extModule.getName());
  }
  if (auto module = dyn_cast<FModuleOp>(op)) {
    return lookup(module.getName());
  }
  llvm_unreachable("Can only look up module operations.");
}

InstanceGraphNode *InstanceGraph::getOrAddNode(StringRef name) {
  // Try to insert an InstanceGraphNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto itAndInserted = nodeMap.try_emplace(name, 0);
  auto &index = itAndInserted.first->second;
  if (itAndInserted.second) {
    // This is a new node, we have to add an element to the NodeVec.
    nodes.emplace_back();
    // Store the node storage index in to the map.
    index = nodes.size() - 1;
    return &nodes.back();
  }
  return &nodes[index];
}

Operation *InstanceGraph::getReferencedModule(InstanceOp op) {
  return lookup(op.moduleName())->getModule();
}
