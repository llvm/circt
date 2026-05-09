//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ProceduralRegionTrait.h"

using namespace llvm;
using namespace mlir;

namespace circt {

LogicalResult verifyNotInProceduralRegion(Operation *op) {
  auto *parent = op;
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<NonProceduralRegion>())
      return success();
    if (parent->hasTrait<ProceduralRegion>())
      return op->emitOpError("must not be in a procedural region");
  }
  return success();
}

LogicalResult verifyNotInNonProceduralRegion(Operation *op) {
  auto *parent = op;
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<ProceduralRegion>())
      return success();
    if (parent->hasTrait<NonProceduralRegion>())
      return op->emitOpError("must not be in a non-procedural region");
  }
  return success();
}

bool isProceduralRegionOp(Operation *op) {
  for (; op; op = op->getParentOp()) {
    if (op->hasTrait<NonProceduralRegion>())
      return false;
    if (op->hasTrait<ProceduralRegion>())
      return true;
  }
  return false;
}

} // namespace circt
