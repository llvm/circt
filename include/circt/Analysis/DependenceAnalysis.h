//===- DependenceAnalysis.h - memory dependence analyses ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving memory access dependences.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_DEPENDENCE_ANALYSIS_H
#define CIRCT_ANALYSIS_DEPENDENCE_ANALYSIS_H

#include "circt/Support/LLVM.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include <utility>

namespace mlir {
class DependenceComponent;
class FuncOp;
} // namespace mlir

namespace circt {
namespace analysis {

/// MemoryDependence captures a dependence from one memory operation to another.
/// It represents the destination of the dependence edge, the type of the
/// dependence, and the components associated with each enclosing loop.
struct MemoryDependence {
  MemoryDependence(Operation *destination,
                   mlir::DependenceResult::ResultEnum dependenceType,
                   ArrayRef<mlir::DependenceComponent> dependenceComponents)
      : destination(destination), dependenceType(dependenceType),
        dependenceComponents(dependenceComponents.begin(),
                             dependenceComponents.end()) {}

  // The dependence is from some source operation to this destination operation.
  Operation *destination;

  // The dependence type denotes whether or not there is a dependence.
  mlir::DependenceResult::ResultEnum dependenceType;

  // The dependence components include lower and upper bounds for each loop.
  SmallVector<mlir::DependenceComponent> dependenceComponents;
};

/// MemoryDependenceResult captures a set of memory dependences. The map key is
/// the operation from which the dependence originates, and the map value is
/// zero or more MemoryDependences for that operation.
using MemoryDependenceResult =
    DenseMap<Operation *, SmallVector<MemoryDependence>>;

/// getMemoryAccessDependences traverses any AffineForOps in the FuncOp body and
/// checks for memory access dependences. Results are output into the 'results'
/// argument.
void getMemoryAccessDependences(mlir::FuncOp funcOp,
                                MemoryDependenceResult &results);

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_DEPENDENCE_ANALYSIS_H
