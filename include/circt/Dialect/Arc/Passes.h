//===- Passes.h - Arc dialect passes --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_PASSES_H
#define CIRCT_DIALECT_ARC_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace arc {

std::unique_ptr<mlir::Pass> createDedupPass();
std::unique_ptr<mlir::Pass> createInferMemoriesPass();
std::unique_ptr<mlir::Pass> createInlineModulesPass();
std::unique_ptr<mlir::Pass> createMakeTablesPass();
std::unique_ptr<mlir::Pass> createSimplifyVariadicOpsPass();
std::unique_ptr<mlir::Pass> createSinkInputsPass();
std::unique_ptr<mlir::Pass> createSplitLoopsPass();
std::unique_ptr<mlir::Pass> createStripSVPass();

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Arc/Passes.h.inc"

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_PASSES_H
