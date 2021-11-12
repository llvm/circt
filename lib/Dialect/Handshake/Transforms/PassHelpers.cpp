//===- PassHelpers.cpp - handshake pass helper functions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions for various helper functions used in handshake
// passes.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace circt {
namespace handshake {

/// Iterates over the handshake::FuncOp's in the program to build an instance
/// graph. In doing so, we detect whether there are any cycles in this graph, as
/// well as infer a top module for the design by performing a topological sort
/// of the instance graph. The result of this sort is placed in sortedFuncs.
LogicalResult resolveInstanceGraph(ModuleOp moduleOp,
                                   InstanceGraph &instanceGraph,
                                   std::string &topLevel,
                                   SmallVectorImpl<std::string> &sortedFuncs) {
  // Create use graph
  auto walkFuncOps = [&](handshake::FuncOp funcOp) {
    auto &funcUses = instanceGraph[funcOp.getName().str()];
    funcOp.walk([&](handshake::InstanceOp instanceOp) {
      funcUses.insert(instanceOp.getModule().str());
    });
  };
  moduleOp.walk(walkFuncOps);

  // find top-level (and cycles) using a topological sort. Initialize all
  // instances as candidate top level modules; these will be pruned whenever
  // they are referenced by another module.
  std::set<std::string> visited, marked, candidateTopLevel;
  SmallVector<std::string> cycleTrace;
  bool cyclic = false;
  llvm::transform(instanceGraph,
                  std::inserter(candidateTopLevel, candidateTopLevel.begin()),
                  [](auto it) { return it.first; });
  std::function<void(const std::string &, SmallVector<std::string>)> cycleUtil =
      [&](const std::string &node, SmallVector<std::string> trace) {
        if (cyclic || visited.count(node))
          return;
        trace.push_back(node);
        if (marked.count(node)) {
          cyclic = true;
          cycleTrace = trace;
          return;
        }
        marked.insert(node);
        for (auto use : instanceGraph[node]) {
          candidateTopLevel.erase(use);
          cycleUtil(use, trace);
        }
        marked.erase(node);
        visited.insert(node);
        sortedFuncs.insert(sortedFuncs.begin(), node);
      };
  for (auto it : instanceGraph) {
    if (visited.count(it.first) == 0)
      cycleUtil(it.first, {});
    if (cyclic)
      break;
  }

  if (cyclic) {
    auto err = moduleOp.emitOpError();
    err << "cannot lower handshake program - cycle "
           "detected in instance graph (";
    llvm::interleave(
        cycleTrace, err, [&](auto node) { err << node; }, "->");
    err << ").";
    return err;
  }
  assert(!candidateTopLevel.empty() &&
         "if non-cyclic, there should be at least 1 candidate top level");

  if (candidateTopLevel.size() > 1) {
    auto err = moduleOp.emitOpError();
    err << "multiple candidate top-level modules detected (";
    llvm::interleaveComma(candidateTopLevel, err,
                          [&](auto topLevel) { err << topLevel; });
    err << "). Please remove one of these from the source program.";
    return err;
  }
  topLevel = *candidateTopLevel.begin();
  return success();
}

} // namespace handshake
} // namespace circt
