//===- DeviceDB.cpp - Implement a device database -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/DeviceDB.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// NOTE: Nothing in this implementation is in any way the most optimal
// implementation. We put off deciding what the correct data structure is until
// we have a better handle of the operations it must accelerate. Performance is
// not an immediate goal.
//===----------------------------------------------------------------------===//

DeviceDB::DeviceDB() {}

/// Assign an instance to a primitive. Return false if another instance is
/// already placed at that location.
LogicalResult DeviceDB::addPrimitive(PhysLocationAttr loc) {
  DenseSet<PrimitiveType> primsAtLoc =
      placements[loc.getX()][loc.getY()][loc.getNum()];
  PrimitiveType prim = loc.getPrimitiveType().getValue();
  primsAtLoc.insert(prim);
  return success();
}
