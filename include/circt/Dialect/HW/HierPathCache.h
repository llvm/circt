//===- HierPathCache.h - HierPathOp Caching Utility -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The HierPathCache is a utility for creating hierarchical paths at a
// pre-defined location in a circuit.  This exists to help with a common pattern
// where you are running a transform and you need to build HierPathOps, but you
// don't know when you are going to do it.  You also don't want to create the
// same HierPathOp multiple times.  This utility will maintain a cache of
// existing ops and only create new ones when necessary.  Additionally, this
// creates the ops in nice, predictable order.  I.e., all the ops are inserted
// into the IR in the order they are created, not in reverse order.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HIERPATHCACHE_H
#define CIRCT_DIALECT_HW_HIERPATHCACHE_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include <mlir/IR/Attributes.h>

namespace circt {
namespace hw {

class HierPathCache {
public:
  HierPathCache(Namespace *ns, OpBuilder::InsertPoint insertionPoint)
      : ns(ns), pathInsertPoint(insertionPoint) {}

  /// Get an existing `hw::HierPathOp` at the default location in the circuit.
  HierPathOp getOrCreatePath(ArrayAttr pathArray, Location loc,
                             StringRef nameHint = "xmrPath");

  /// Get an existing `hw::HierPathOp` at a specific location in the circuit.
  /// The insertion point will be updated to allow for this method to be used
  /// repeatedly to create the ops predictably, one after the other.
  HierPathOp getOrCreatePath(ArrayAttr pathArray, Location loc,
                             OpBuilder::InsertPoint &insertPoint,
                             StringRef nameHint = "xmrPath");

private:
  /// A namespace in which symbols for hierarchical path ops will be created.
  Namespace *ns;

  /// A cache of already created HierPathOps.  This is used to avoid repeatedly
  /// creating the same HierPathOp.
  DenseMap<mlir::Attribute, hw::HierPathOp> pathCache;

  /// The insertion point where the pass inserts HierPathOps.
  OpBuilder::InsertPoint pathInsertPoint;
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_HIERPATHCACHE_H
