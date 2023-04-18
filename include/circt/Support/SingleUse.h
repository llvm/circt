//===- SingleUse.h - --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility for checking that all values within a given function-like op are
// only used once.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SINGLEUSE_H
#define CIRCT_SUPPORT_SINGLEUSE_H

#include "mlir/IR/FunctionInterfaces.h"

namespace circt {

LogicalResult verifyAllValuesHasOneUse(mlir::FunctionOpInterface funcOp) {
  if (funcOp.isExternal())
    return success();

  auto checkUseFunc = [&](Operation *op, Value v, Twine desc,
                          unsigned idx) -> LogicalResult {
    auto numUses = std::distance(v.getUses().begin(), v.getUses().end());
    if (numUses == 0)
      return op->emitOpError() << desc << " " << idx << " has no uses.";
    if (numUses > 1)
      return op->emitOpError() << desc << " " << idx << " has multiple uses.";
    return success();
  };

  // Validate blocks within the function
  for (auto [idx, block] : enumerate(funcOp.getBlocks())) {
    // Validate ops within the block
    for (auto &subOp : block) {
      for (auto res : llvm::enumerate(subOp.getResults())) {
        if (failed(checkUseFunc(&subOp, res.value(), "result", res.index())))
          return failure();
      }
    }

    for (auto &barg : block.getArguments()) {
      if (failed(checkUseFunc(funcOp.getOperation(), barg,
                              "block #" + Twine(idx) + " argument",
                              barg.getArgNumber())))
        return failure();
    }
  }
  return success();
}

} // namespace circt

#endif // CIRCT_SUPPORT_SINGLEUSE_H
