//===- FIRRTLInstanceGraph.h - Instance graph -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FIRRTL InstanceGraph, which is similar to a CallGraph.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEGRAPH_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEGRAPH_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/InstanceGraphBase.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"

namespace circt {
namespace firrtl {
using InstanceRecord = hw::InstanceRecord;
using InstanceGraphNode = hw::InstanceGraphNode;

/// This graph tracks modules and where they are instantiated. This is intended
/// to be used as a cached analysis on FIRRTL circuits.  This class can be used
/// to walk the modules efficiently in a bottom-up or top-down order.
///
/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &instanceGraph = getAnalysis<InstanceGraph>(getOperation());
class InstanceGraph : public hw::InstanceGraphBase {
public:
  /// Create a new module graph of a circuit.  This must be called on a FIRRTL
  /// CircuitOp or MLIR ModuleOp.
  explicit InstanceGraph(Operation *operation);

  /// Get the node corresponding to the top-level module of a circuit.
  InstanceGraphNode *getTopLevelNode() override { return topLevelNode; }

  /// Get the module corresponding to the top-lebel module of a circuit.
  FModuleLike getTopLevelModule() {
    return cast<FModuleLike>(*getTopLevelNode()->getModule());
  }

private:
  InstanceGraphNode *topLevelNode;
};

/// An absolute instance path.
using InstancePath = ArrayRef<InstanceOp>;

template <typename T>
inline static T &formatInstancePath(T &into, const InstancePath &path) {
  into << "$root";
  for (auto inst : path)
    into << "/" << inst.name() << ":" << inst.moduleName();
  return into;
}

template <typename T>
static T &operator<<(T &os, const InstancePath &path) {
  return formatInstancePath(os, path);
}

/// A data structure that caches and provides absolute paths to module instances
/// in the IR.
struct InstancePathCache {
  /// The instance graph of the IR.
  InstanceGraph &instanceGraph;

  explicit InstancePathCache(InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {}
  ArrayRef<InstancePath> getAbsolutePaths(Operation *op);

private:
  /// An allocator for individual instance paths and entire path lists.
  llvm::BumpPtrAllocator allocator;

  /// Cached absolute instance paths.
  DenseMap<Operation *, ArrayRef<InstancePath>> absolutePathsCache;

  /// Append an instance to a path.
  InstancePath appendInstance(InstancePath path, InstanceOp inst);
};

} // namespace firrtl
} // namespace circt

template <>
struct llvm::GraphTraits<circt::firrtl::InstanceGraph *>
    : public llvm::GraphTraits<circt::hw::InstanceGraphBase *> {};

template <>
struct llvm::DOTGraphTraits<circt::firrtl::InstanceGraph *>
    : public llvm::DOTGraphTraits<circt::hw::InstanceGraphBase *> {
  using llvm::DOTGraphTraits<circt::hw::InstanceGraphBase *>::DOTGraphTraits;
};

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEGRAPH_H
