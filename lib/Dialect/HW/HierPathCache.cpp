//===- HierPathCache.h - HierPathOp Caching Utility -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of a utility for creating Hierarchical Path operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HierPathCache.h"
#include "circt/Dialect/HW/HWOps.h"

namespace circt {
namespace hw {

HierPathOp HierPathCache::getOrCreatePath(ArrayAttr pathArray, Location loc,
                                          StringRef nameHint) {
  return getOrCreatePath(pathArray, loc, pathInsertPoint, nameHint);
}

HierPathOp HierPathCache::getOrCreatePath(ArrayAttr pathArray, Location loc,
                                          OpBuilder::InsertPoint &insertPoint,
                                          StringRef nameHint) {

  assert(pathArray && !pathArray.empty());
  // Return an existing HierPathOp if one exists with the same path.  Add
  // location information to the existing HierPathOp if it is being reused.
  auto pathIter = pathCache.find(pathArray);
  if (pathIter != pathCache.end()) {
    auto &hierPathOp = pathIter->getSecond();
    auto oldLoc = hierPathOp->getLoc();
    // Fuse the location of all old locations and the new location.
    SmallVector<Location> locations;
    if (auto fusedLoc = dyn_cast<FusedLoc>(oldLoc)) {
      auto oldLocs = fusedLoc.getLocations();
      locations.append(oldLocs.begin(), oldLocs.end());
    } else {
      locations.push_back(oldLoc);
    }
    locations.push_back(loc);
    // Update the location on the original HierPathOp.
    hierPathOp->setLoc(FusedLoc::get(loc.getContext(), locations));
    return hierPathOp;
  }

  // Create a builder and move its insertion point to the original insertion
  // point.
  OpBuilder builder(insertPoint.getBlock(), insertPoint.getPoint());

  // Create the new HierPathOp and insert it into the pathCache.
  hw::HierPathOp path =
      pathCache
          .insert({pathArray, builder.create<hw::HierPathOp>(
                                  loc, ns->newName(nameHint), pathArray)})
          .first->second;
  path.setVisibility(SymbolTable::Visibility::Private);

  // Save the insertion point so other unique HierPathOps will be created
  // after this one.
  insertPoint = builder.saveInsertionPoint();

  // Return the new path.
  return path;
}

} // namespace hw
} // namespace circt
