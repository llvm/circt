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
    if (parent->hasTrait<ProceduralRegion>()) {
      op->emitError() << op->getName() << " must not be in a procedural region";
      return failure();
    }
  }
  return success();
}

LogicalResult verifyNotInNonProceduralRegion(Operation *op) {
  auto *parent = op;
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<ProceduralRegion>())
      return success();
    if (parent->hasTrait<NonProceduralRegion>()) {
      op->emitError() << op->getName()
                      << " must not be in a non-procedural region";
      return failure();
    }
  }
  return success();
}

} // namespace circt
