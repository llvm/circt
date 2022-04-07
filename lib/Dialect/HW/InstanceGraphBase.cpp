//===- InstanceGraphBase.cpp - Instance Graph -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/InstanceGraphBase.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace hw;

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

InstanceRecord *InstanceGraphNode::addInstance(HWInstanceLike instance,
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

InstanceGraphNode *InstanceGraphBase::getOrAddNode(StringAttr name) {
  // Try to insert an InstanceGraphNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto *&node = nodeMap[name];
  if (!node) {
    node = new InstanceGraphNode();
    nodes.push_back(node);
  }
  return node;
}

InstanceGraphBase::InstanceGraphBase(Operation *parent) : parent(parent) {
  parent->walk([&](HWModuleLike module) {
    auto name = module.moduleNameAttr();
    auto *currentNode = getOrAddNode(name);
    currentNode->module = module;

    // Find all instance operations in the module body.
    module.walk([&](HWInstanceLike instanceOp) {
      // Add an edge to indicate that this module instantiates the target.
      auto *targetNode = getOrAddNode(instanceOp.referencedModuleNameAttr());
      currentNode->addInstance(instanceOp, targetNode);
    });
  });
}

InstanceGraphNode *InstanceGraphBase::addModule(HWModuleLike module) {
  assert(!nodeMap.count(module.moduleNameAttr()) && "module already added");
  auto *node = new InstanceGraphNode();
  node->module = module;
  nodeMap[module.moduleNameAttr()] = node;
  nodes.push_back(node);
  return node;
}

void InstanceGraphBase::erase(InstanceGraphNode *node) {
  assert(node->noUses() &&
         "all instances of this module must have been erased.");
  // Erase all instances inside this module.
  for (auto *instance : llvm::make_early_inc_range(*node))
    instance->erase();
  nodeMap.erase(node->getModule().moduleNameAttr());
  nodes.erase(node);
}

InstanceGraphNode *InstanceGraphBase::lookup(StringAttr name) {
  auto it = nodeMap.find(name);
  assert(it != nodeMap.end() && "Module not in InstanceGraph!");
  return it->second;
}

InstanceGraphNode *InstanceGraphBase::lookup(HWModuleLike op) {
  return lookup(cast<HWModuleLike>(op).moduleNameAttr());
}

HWModuleLike InstanceGraphBase::getReferencedModule(HWInstanceLike op) {
  return lookup(op.referencedModuleNameAttr())->getModule();
}

InstanceGraphBase::~InstanceGraphBase() {}

void InstanceGraphBase::replaceInstance(HWInstanceLike inst,
                                        HWInstanceLike newInst) {
  assert(inst.referencedModuleName() == newInst.referencedModuleName() &&
         "Both instances must be targeting the same module");

  // Find the instance record of this instance.
  auto *node = lookup(inst.referencedModuleNameAttr());
  auto it = llvm::find_if(node->uses(), [&](InstanceRecord *record) {
    return record->getInstance() == inst;
  });
  assert(it != node->usesEnd() && "Instance of module not recorded in graph");

  // We can just replace the instance op in the InstanceRecord without updating
  // any instance lists.
  (*it)->instance = newInst;
}

bool InstanceGraphBase::isAncestor(HWModuleLike child, HWModuleLike parent) {
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
