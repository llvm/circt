//===- FIRRTLInstanceGraph.cpp - Instance Graph -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace firrtl;

static CircuitOp findCircuitOp(Operation *operation) {
  if (auto mod = dyn_cast<mlir::ModuleOp>(operation))
    for (auto &op : *mod.getBody())
      if (auto circuit = dyn_cast<CircuitOp>(&op))
        return circuit;
  return cast<CircuitOp>(operation);
}

InstanceGraph::InstanceGraph(Operation *operation)
    : igraph::InstanceGraph(findCircuitOp(operation)) {
  topLevelNode = lookup(cast<CircuitOp>(getParent()).getNameAttr());
}
