//===- OwningModuleCache.h - Memoized cache of owning modules ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OWNINGMODULECACHE_H
#define CIRCT_DIALECT_FIRRTL_OWNINGMODULECACHE_H

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

namespace circt {
namespace firrtl {

/// This implements an analysis to determine which module owns a given path
/// operation. The owning module of a path is a module which contains the
/// path op, or instantiates it through children classes.  A path operation
/// is generally only allowed to have a single owning module, and must target
/// an entity underneath that module's hierarchy.
struct OwningModuleCache {

  OwningModuleCache(InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {}

  /// Return this operation's owning module.  If there is none or more than
  /// one, this returns null.
  FModuleOp lookup(ClassOp classOp) {
    auto op = classOp;

    auto [it, inserted] = cache.try_emplace(op);
    if (!inserted)
      return it->second;

    // Slow path.
    struct StackElement {
      StackElement(InstanceGraphNode *node)
          : current(node->getModule()), it(node->usesBegin()),
            end(node->usesEnd()) {}
      bool first = true;
      FModuleOp owner = nullptr;
      Operation *current;
      InstanceGraphNode::UseIterator it;
      InstanceGraphNode::UseIterator end;
    };

    auto combine = [](FModuleOp current, FModuleOp parent, bool first) {
      if (first)
        current = parent;
      else if (current != parent)
        current = nullptr;
      return current;
    };

    auto *node = instanceGraph.lookup(op);
    SmallVector<StackElement> stack;

    stack.emplace_back(node);

    auto returning = false;
    FModuleOp result = nullptr;
    while (!stack.empty()) {

      auto &elt = stack.back();
      auto &it = elt.it;
      auto &end = elt.end;

      if (returning) {
        elt.owner = combine(elt.owner, result, elt.first);
        elt.first = false;
        returning = false;
      }

      // Try to get each child.
      while (true) {
        if (it == end) {
          // Set up to return the result.
          returning = true;
          result = elt.owner;
          // Cache the result.
          cache[elt.current] = elt.owner;
          stack.pop_back();
          break;
        }

        auto *parentNode = (*it)->getParent();
        auto parent = parentNode->getModule();

        // Advance past the current element.
        ++it;

        if (auto parentModule = dyn_cast<FModuleOp>(parent.getOperation())) {
          elt.owner = combine(elt.owner, parentModule, elt.first);
          elt.first = false;
          continue;
        }
        auto [pIt, pInserted] = cache.try_emplace(parent);
        if (!pInserted) {
          elt.owner = combine(elt.owner, pIt->second, elt.first);
          elt.first = false;
          continue;
        }

        // Set it up to iterate the child.
        stack.emplace_back(parentNode);
        returning = false;
        result = nullptr;
        break;
      }
    }

    assert(returning && "must be returning a result");
    return result;
  }

  /// Return this operation's owning module.  If there is none or more than
  /// one, this returns null.
  FModuleOp lookup(Operation *op) {
    while (op) {
      if (auto module = dyn_cast<FModuleOp>(op))
        return module;
      if (auto classOp = dyn_cast<ClassOp>(op))
        return lookup(classOp);
      op = op->getParentOp();
    }
    return nullptr;
  }

private:
  DenseMap<Operation *, FModuleOp> cache;
  InstanceGraph &instanceGraph;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_OWNINGMODULECACHE_H
