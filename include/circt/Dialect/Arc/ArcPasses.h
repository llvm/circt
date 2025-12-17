//===- ArcPasses.h - Arc dialect passes -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCPASSES_H
#define CIRCT_DIALECT_ARC_ARCPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

#include "circt/Dialect/HW/HWOps.h"

namespace mlir {
class Pass;
} // namespace mlir

#include "circt/Dialect/Arc/ArcPassesEnums.h.inc"

namespace circt {
namespace arc {

#define GEN_PASS_DECL
#include "circt/Dialect/Arc/ArcPasses.h.inc"

std::unique_ptr<mlir::Pass>
createAddTapsPass(const AddTapsOptions &options = {});
std::unique_ptr<mlir::Pass> createAllocateStatePass();
std::unique_ptr<mlir::Pass> createArcCanonicalizerPass();
std::unique_ptr<mlir::Pass> createDedupPass();
std::unique_ptr<mlir::Pass> createFindInitialVectorsPass();
std::unique_ptr<mlir::Pass>
createInferMemoriesPass(const InferMemoriesOptions &options = {});
std::unique_ptr<mlir::Pass> createInlineArcsPass();
std::unique_ptr<mlir::Pass> createIsolateClocksPass();
std::unique_ptr<mlir::Pass> createLatencyRetimingPass();
std::unique_ptr<mlir::Pass> createLowerArcsToFuncsPass();
std::unique_ptr<mlir::Pass> createLowerClocksToFuncsPass();
std::unique_ptr<mlir::Pass> createLowerLUTPass();
std::unique_ptr<mlir::Pass> createLowerVectorizationsPass(
    LowerVectorizationsModeEnum mode = LowerVectorizationsModeEnum::Full);
std::unique_ptr<mlir::Pass> createMakeTablesPass();
std::unique_ptr<mlir::Pass> createMuxToControlFlowPass();
std::unique_ptr<mlir::Pass> createPrintCostModelPass();
std::unique_ptr<mlir::Pass> createSimplifyVariadicOpsPass();
std::unique_ptr<mlir::Pass> createSplitLoopsPass();
std::unique_ptr<mlir::Pass> createStripSVPass(bool ignoreAsync = false);

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Arc/ArcPasses.h.inc"

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCPASSES_H
