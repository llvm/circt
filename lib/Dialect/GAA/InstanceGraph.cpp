//===- GAAInstanceGraph.cpp - Instance Graph -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/GAA/InstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace circt;
using namespace gaa;

InstanceGraph::InstanceGraph(Operation *operation)
    : InstanceGraphBase(cast<CircuitOp>(operation)) {
  auto circuit = cast<CircuitOp>(getParent());
  topLevelNode = lookup(circuit.getNameAttr());
}