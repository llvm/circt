//===- MSFTOpInterfaces.cpp - Implement MSFT OpInterfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

using namespace circt;
using namespace msft;

LogicalResult circt::msft::verifyUnaryDynInstDataOp(Operation *op) {
  auto inst = dyn_cast<DynamicInstanceOp>(op->getParentOp());
  FlatSymbolRefAttr pathRef =
      cast<UnaryDynInstDataOpInterface>(op).getPathSym();

  if (inst && pathRef)
    return op->emitOpError("cannot both have a global ref symbol and be a "
                           "child of a dynamic instance op");
  if (!inst && !pathRef)
    return op->emitOpError("must have either a global ref symbol of belong to "
                           "a dynamic instance op");
  return success();
}

Operation *circt::msft::getHierPathTopModule(Location loc,
                                             circt::hw::HWSymbolCache &symCache,
                                             FlatSymbolRefAttr pathSym) {
  assert(pathSym && "pathSym must be non-null");
  auto ref = dyn_cast_or_null<hw::HierPathOp>(symCache.getDefinition(pathSym));
  if (!ref) {
    emitError(loc) << "could not find hw.hierpath " << pathSym;
    return nullptr;
  }
  if (ref.getNamepath().empty())
    return nullptr;
  auto modSym = FlatSymbolRefAttr::get(
      cast<hw::InnerRefAttr>(ref.getNamepath()[0]).getRoot());
  return symCache.getDefinition(modSym);
}

namespace circt {
namespace msft {
#include "circt/Dialect/MSFT/MSFTOpInterfaces.cpp.inc"
} // namespace msft
} // namespace circt
