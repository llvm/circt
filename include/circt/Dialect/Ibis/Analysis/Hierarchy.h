//===- Hierarchy.h - Ibis hierarchy analysis --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_HIERARCHY_H
#define CIRCT_DIALECT_IBIS_HIERARCHY_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"

#include "circt/Dialect/Ibis/IbisOps.h"

namespace circt {
namespace ibis {

class Hierarchy {

  // Defines a dependency of an Ibis container to another Ibis container.
  struct Dependence {};

  // Parent dependencies are used to track dependencies going from a child
  // to a parent in the instance hierarchy. Parent dependencies are optionally
  // typed.
  struct ParentDependence : public Dependence {};

  // Child dependencies are used to track dependencies going from a parent
  // to a child in the instance hierarchy.
  // - Child dependencies are always typed
  // - Child dependencies always refer to a specific instance name
  struct ChildDependence : public Dependence {
    mlir::StringAttr instanceName;
  };

  struct Dependency {
    llvm::SmallVector<std::unique_ptr<Dependence>> path;
  };

public:
  explicit Hierarchy(mlir::Operation *operation);

  // Lookup the dependencies of a given scope-like op.
  const Dependency &lookup(ScopeOpInterface scopeOp);

private:
  llvm::DenseMap<ScopeOpInterface, Dependency> analysis;
};

} // namespace ibis
} // namespace circt
#endif // CIRCT_DIALECT_IBIS_HIERARCHY_H
