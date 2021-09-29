//===- DependenceAnalysis.cpp - memory dependence analyses ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis involving memory access
// dependences.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include <mlir/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h>

using namespace mlir;
using namespace circt::analysis;

/// Helper to iterate through memory operation pairs and check for dependences
/// at a given loop nesting depth.
static void checkMemrefDependence(SmallVectorImpl<Operation *> &memoryOps,
                                  unsigned depth,
                                  MemoryDependenceResult &results) {
  for (auto *source : memoryOps) {
    for (auto *destination : memoryOps) {
      if (source == destination)
        continue;

      MemRefAccess src(source);
      MemRefAccess dst(destination);
      FlatAffineValueConstraints dependenceConstraints;
      SmallVector<DependenceComponent, 2> depComps;
      DependenceResult result = checkMemrefAccessDependence(
          src, dst, depth, &dependenceConstraints, &depComps, true);

      if (results.count(destination) == 0)
        results[destination] = SmallVector<MemoryDependence>();

      results[destination].emplace_back(source, result.value, depComps);

      // Also consider intra-iteration dependences.
      if (src == dst && source->isBeforeInBlock(destination)) {
        SmallVector<DependenceComponent> intraDeps;
        for (size_t i = 1; i <= depth; ++i) {
          // TODO: for completeness, this should also find the for loop and
          // initialize the Operation * in depComp, even though we won't use it.
          DependenceComponent depComp;
          depComp.lb = 0;
          depComp.ub = 0;
          intraDeps.emplace_back(depComp);
        }
        results[destination].emplace_back(
            source, DependenceResult::HasDependence, intraDeps);
      }
    }
  }
}

/// MemoryDependenceAnalysis traverses any AffineForOps in the FuncOp body and
/// checks for memory access dependences. Results are captured in a
/// MemoryDependenceResult, which can by queried by Operation.
circt::analysis::MemoryDependenceAnalysis::MemoryDependenceAnalysis(
    mlir::FuncOp funcOp) {
  // Collect affine loops grouped by nesting depth.
  std::vector<SmallVector<AffineForOp, 2>> depthToLoops;
  mlir::gatherLoops(funcOp, depthToLoops);

  // Collect load and store operations to check.
  SmallVector<Operation *> memoryOps;
  funcOp.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      memoryOps.push_back(op);
  });

  // For each depth, check memref accesses.
  for (unsigned depth = 1, e = depthToLoops.size(); depth <= e; ++depth)
    checkMemrefDependence(memoryOps, depth, results);
}

/// Returns the dependences, if any, that the given Operation depends on.
ArrayRef<MemoryDependence>
circt::analysis::MemoryDependenceAnalysis::getDependences(Operation *op) {
  return results[op];
}
