//===- GAAInstanceGraph.h - Instance graph ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GAA InstanceGraph.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_GAA_GAAINSTANCEGRAPH_H
#define CIRCT_DIALECT_GAA_GAAINSTANCEGRAPH_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"

#include "circt/Dialect/GAA/GAAOps.h"
#include "circt/Dialect/HW/InstanceGraphBase.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace gaa {
using InstanceRecord = hw::InstanceRecord;
using InstanceGraphNode = hw::InstanceGraphNode;

class InstanceGraph : public hw::InstanceGraphBase {
public:
  /// Create a new module graph of a circuit.  This must be called on a GAA
  /// CircuitOp or MLIR ModuleOp.
  explicit InstanceGraph(Operation *operation);

  /// Get the node corresponding to the top-level module of a circuit.
  InstanceGraphNode *getTopLevelNode() override { return topLevelNode; }

  GAAModuleLike getTopLevelModule() {
    return cast<GAAModuleLike>(*getTopLevelNode()->getModule());
  }

private:
  InstanceGraphNode *topLevelNode;
};
} // namespace gaa
} // namespace circt

template <>
struct llvm::GraphTraits<circt::gaa::InstanceGraph *>
    : public llvm::GraphTraits<circt::hw::InstanceGraphBase *> {};

template <>
struct llvm::DOTGraphTraits<circt::gaa::InstanceGraph *>
    : public llvm::DOTGraphTraits<circt::hw::InstanceGraphBase *> {
  using llvm::DOTGraphTraits<circt::hw::InstanceGraphBase *>::DOTGraphTraits;
};

#endif // CIRCT_DIALECT_GAA_GAAINSTANCEGRAPH_H
