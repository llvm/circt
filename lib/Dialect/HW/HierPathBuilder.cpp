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

HierPathOp HierPathBuilder::getOrCreatePath(ArrayAttr pathArray,
                                            ImplicitLocOpBuilder &builder) {

  assert(pathArray && !pathArray.empty());
  // Return an existing HierPathOp if one exists with the same path.
  auto pathIter = pathCache.find(pathArray);
  if (pathIter != pathCache.end())
    return pathIter->second;

  // Reset the insertion point after this function returns.
  OpBuilder::InsertionGuard guard(builder);

  builder.restoreInsertionPoint(pathInsertPoint);

  // Create the new HierPathOp and insert it into the pathCache.
  hw::HierPathOp path =
      pathCache
          .insert({pathArray, builder.create<hw::HierPathOp>(
                                  ns->newName("xmrPath"), pathArray)})
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
