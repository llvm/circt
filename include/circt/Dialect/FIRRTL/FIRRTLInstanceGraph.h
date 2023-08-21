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
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"

namespace circt {
namespace firrtl {
using InstanceRecord = igraph::InstanceRecord;
using InstanceGraphNode = igraph::InstanceGraphNode;
using InstancePathCache = igraph::InstancePathCache;

/// This graph tracks modules and where they are instantiated. This is intended
/// to be used as a cached analysis on FIRRTL circuits.  This class can be used
/// to walk the modules efficiently in a bottom-up or top-down order.
///
/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &instanceGraph = getAnalysis<InstanceGraph>(getOperation());
class InstanceGraph : public igraph::InstanceGraph {
public:
  /// Create a new module graph of a circuit.  This must be called on a FIRRTL
  /// CircuitOp or MLIR ModuleOp.
  explicit InstanceGraph(Operation *operation);

  /// Get the node corresponding to the top-level module of a circuit.
  igraph::InstanceGraphNode *getTopLevelNode() override { return topLevelNode; }

  /// Get the module corresponding to the top-level module of a circuit.
  FModuleLike getTopLevelModule() {
    return cast<FModuleLike>(*getTopLevelNode()->getModule());
  }

private:
  InstanceGraphNode *topLevelNode;
};

bool allUnder(ArrayRef<InstanceRecord *> nodes, InstanceGraphNode *top);

} // namespace firrtl
} // namespace circt

template <>
struct llvm::GraphTraits<circt::firrtl::InstanceGraph *>
    : public llvm::GraphTraits<circt::igraph::InstanceGraph *> {};

template <>
struct llvm::DOTGraphTraits<circt::firrtl::InstanceGraph *>
    : public llvm::DOTGraphTraits<circt::igraph::InstanceGraph *> {
  using llvm::DOTGraphTraits<circt::igraph::InstanceGraph *>::DOTGraphTraits;
};

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEGRAPH_H
