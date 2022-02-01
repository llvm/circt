//===- InstanceGraph.cpp - Instance Graph -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace firrtl;

void InstanceRecord::erase() {
  // Update the prev node to point to the next node.
  if (prevUse)
    prevUse->nextUse = nextUse;
  else
    target->firstUse = nextUse;
  // Update the next node to point to the prev node.
  if (nextUse)
    nextUse->prevUse = prevUse;
  getParent()->instances.erase(this);
}

InstanceRecord *InstanceGraphNode::addInstance(InstanceOp instance,
                                               InstanceGraphNode *target) {
  auto *instanceRecord = new InstanceRecord(this, instance, target);
  target->recordUse(instanceRecord);
  instances.push_back(instanceRecord);
  return instanceRecord;
}

void InstanceGraphNode::recordUse(InstanceRecord *record) {
  record->nextUse = firstUse;
  if (firstUse)
    firstUse->prevUse = record;
  firstUse = record;
}

InstanceGraphNode *InstanceGraph::getOrAddNode(StringAttr name) {
  // Try to insert an InstanceGraphNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto *&node = nodeMap[name];
  if (!node) {
    node = new InstanceGraphNode();
    nodes.push_back(node);
  }
  return node;
}

InstanceGraph::InstanceGraph(Operation *operation) {
  if (auto mod = dyn_cast<mlir::ModuleOp>(operation))
    for (auto &op : *mod.getBody())
      if ((operation = dyn_cast<CircuitOp>(&op)))
        break;

  auto circuit = cast<CircuitOp>(operation);
  auto topModuleName = circuit.nameAttr();
  for (auto &op : *circuit.getBody()) {
    auto module = dyn_cast<FModuleLike>(op);
    if (!module)
      continue;
    auto name = module.moduleNameAttr();
    auto *currentNode = getOrAddNode(name);
    currentNode->module = module;
    if (name == topModuleName)
      topLevelNode = currentNode;
    // Find all instance operations in the module body.
    module.walk([&](InstanceOp instanceOp) {
      // Add an edge to indicate that this module instantiates the target.
      auto *targetNode = getOrAddNode(instanceOp.moduleNameAttr().getAttr());
      currentNode->addInstance(instanceOp, targetNode);
    });
  }
}

void InstanceGraph::erase(InstanceGraphNode *node) {
  assert(node->noUses() &&
         "all instances of this module must have been erased.");
  // Erase all instances inside this module.
  for (auto *instance : llvm::make_early_inc_range(*node))
    instance->erase();
  nodeMap.erase(cast<FModuleLike>(node->getModule()).moduleNameAttr());
  nodes.erase(node);
}

InstanceGraphNode *InstanceGraph::getTopLevelNode() { return topLevelNode; }

FModuleLike InstanceGraph::getTopLevelModule() {
  return getTopLevelNode()->getModule();
}

InstanceGraphNode *InstanceGraph::lookup(StringAttr name) {
  auto it = nodeMap.find(name);
  assert(it != nodeMap.end() && "Module not in InstanceGraph!");
  return it->second;
}

InstanceGraphNode *InstanceGraph::lookup(Operation *op) {
  return lookup(cast<FModuleLike>(op).moduleNameAttr());
}

Operation *InstanceGraph::getReferencedModule(InstanceOp op) {
  return lookup(op.moduleNameAttr().getAttr())->getModule();
}

void InstanceGraph::replaceInstance(InstanceOp inst, InstanceOp newInst) {
  assert(inst.moduleName() == newInst.moduleName() &&
         "Both instances must be targeting the same module");

  // Find the instance record of this instance.
  auto *node = lookup(inst.moduleNameAttr().getAttr());
  auto it = llvm::find_if(node->uses(), [&](InstanceRecord *record) {
    return record->getInstance() == inst;
  });
  assert(it != node->usesEnd() && "Instance of module not recorded in graph");

  // We can just replace the instance op in the InstanceRecord without updating
  // any instance lists.
  (*it)->instance = newInst;
}

bool InstanceGraph::isAncestor(FModuleLike child, FModuleOp parent) {
  DenseSet<InstanceGraphNode *> seen;
  SmallVector<InstanceGraphNode *> worklist;
  auto *cn = lookup(child);
  worklist.push_back(cn);
  seen.insert(cn);
  while (!worklist.empty()) {
    auto *node = worklist.back();
    worklist.pop_back();
    if (node->getModule() == parent)
      return true;
    for (auto *use : node->uses()) {
      auto *mod = use->getParent();
      if (!seen.count(mod)) {
        seen.insert(mod);
        worklist.push_back(mod);
      }
    }
  }
  return false;
}

ArrayRef<InstancePath> InstancePathCache::getAbsolutePaths(Operation *op) {
  assert((isa<FModuleOp, FExtModuleOp>(op))); // extra parens makes parser smile

  // If we have reached the circuit root, we're done.
  if (op == instanceGraph.getTopLevelNode()->getModule()) {
    static InstancePath empty{};
    return empty; // array with single empty path
  }

  // Fast path: hit the cache.
  auto cached = absolutePathsCache.find(op);
  if (cached != absolutePathsCache.end())
    return cached->second;

  // For each instance, collect the instance paths to its parent and append the
  // instance itself to each.
  SmallVector<InstancePath, 8> extendedPaths;
  for (auto inst : instanceGraph[op]->uses()) {
    auto instPaths = getAbsolutePaths(inst->getParent()->getModule());
    extendedPaths.reserve(instPaths.size());
    for (auto path : instPaths) {
      extendedPaths.push_back(appendInstance(path, inst->getInstance()));
    }
  }

  // Move the list of paths into the bump allocator for later quick retrieval.
  ArrayRef<InstancePath> pathList;
  if (!extendedPaths.empty()) {
    auto paths = allocator.Allocate<InstancePath>(extendedPaths.size());
    std::copy(extendedPaths.begin(), extendedPaths.end(), paths);
    pathList = ArrayRef<InstancePath>(paths, extendedPaths.size());
  }

  absolutePathsCache.insert({op, pathList});
  return pathList;
}

InstancePath InstancePathCache::appendInstance(InstancePath path,
                                               InstanceOp inst) {
  size_t n = path.size() + 1;
  auto newPath = allocator.Allocate<InstanceOp>(n);
  std::copy(path.begin(), path.end(), newPath);
  newPath[path.size()] = inst;
  return InstancePath(newPath, n);
}
