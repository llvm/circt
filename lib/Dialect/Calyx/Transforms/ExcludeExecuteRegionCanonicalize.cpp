//===- ExcludeExecuteRegion.cpp
//----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_EXCLUDEEXECUTEREGIONCANONICALIZE
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace mlir::affine;
using namespace circt;

namespace {
class ExcludeExecuteRegionCanonicalizePass
    : public circt::calyx::impl::ExcludeExecuteRegionCanonicalizeBase<
          ExcludeExecuteRegionCanonicalizePass> {
  void runOnOperation() override;
};
} // namespace

void ExcludeExecuteRegionCanonicalizePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  // Add dialect-level canonicalization patterns
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);

  // Add op-specific canonicalization patterns
  for (const RegisteredOperationName &op : ctx->getRegisteredOperations()) {
    if (op.getStringRef() == "scf.execute_region" ||
        op.getStringRef() == "scf.parallel")
      continue;

    op.getCanonicalizationPatterns(patterns, ctx);
  }

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError("Failed to apply canonicalization.");
    signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect, affine::AffineDialect>();
}

std::unique_ptr<mlir::Pass>
circt::calyx::createExcludeExecuteRegionCanonicalizePass() {
  return std::make_unique<ExcludeExecuteRegionCanonicalizePass>();
}
