//===- HierPathBuilder.cpp - HierPathOp Builder Utility -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of a utility for creating Hierarchical Path operation.s
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HierPathBuilder.h"
#include "circt/Dialect/HW/HWOps.h"

namespace circt {
namespace hw {

HierPathOp HierPathBuilder::getOrCreatePath(ArrayAttr pathArray, Location loc,
                                            StringRef nameHint) {

  assert(pathArray && !pathArray.empty());
  // Return an existing HierPathOp if one exists with the same path.  Add
  // location information to the existing HierPathOp if it is being reused.
  auto pathIter = pathCache.find(pathArray);
  if (pathIter != pathCache.end()) {
    auto &hierPathOp = pathIter->getSecond();
    hierPathOp->setLoc(
        FusedLoc::get(loc.getContext(), hierPathOp->getLoc(), loc));
    return hierPathOp;
  }

  // Create a builder and move its insertion point to the original insertion
  // point.
  OpBuilder builder(pathInsertPoint.getBlock(), pathInsertPoint.getPoint());

  // Create the new HierPathOp and insert it into the pathCache.
  hw::HierPathOp path =
      pathCache
          .insert({pathArray, builder.create<hw::HierPathOp>(
                                  loc, ns->newName(nameHint), pathArray)})
          .first->second;
  path.setVisibility(SymbolTable::Visibility::Private);

  // Save the insertion point so other unique HierPathOps will be created
  // after this one.
  pathInsertPoint = builder.saveInsertionPoint();

  // Return the new path.
  return path;
}

} // namespace hw
} // namespace circt
