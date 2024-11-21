//===- LowerContracts.cpp - Formal Preparations --*- C++ -*----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Lower verif.formal to hw.module.
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_LOWERCONTRACTSPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace verif;
using namespace hw;

namespace {
struct LowerContractsPass
    : verif::impl::LowerContractsPassBase<LowerContractsPass> {
  void runOnOperation() override;
};

struct HWOpRewritePattern : public OpRewritePattern<HWModuleOp> {
  using OpRewritePattern<HWModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HWModuleOp op,
                                PatternRewriter &rewriter) const override {
    auto formalOp =
        rewriter.create<verif::FormalOp>(op.getLoc(), op.getNameAttr());

    rewriter.cloneRegionBefore(op.getRegion(), formalOp.getBody(),
                               formalOp.getBody().end());

    auto *bodyBlock = &formalOp.getBody().front();
    // Erase hw.output
    rewriter.eraseOp(bodyBlock->getTerminator());

    // Convert block args to symbolic values
    rewriter.setInsertionPointToStart(&formalOp.getBody().front());
    for (auto arg : llvm::make_early_inc_range(bodyBlock->getArguments())) {
      auto sym =
          rewriter.create<verif::SymbolicValueOp>(arg.getLoc(), arg.getType());
      rewriter.replaceAllUsesWith(arg, sym);
    }
    bodyBlock->eraseArguments(0, bodyBlock->getNumArguments());

    for (auto contractOp :
         llvm::make_early_inc_range(bodyBlock->getOps<verif::ContractOp>())) {
      rewriter.setInsertionPointToEnd(&contractOp.getBody().front());
      for (auto ensureOp : llvm::make_early_inc_range(
               contractOp.getBody().front().getOps<EnsureOp>())) {
        rewriter.create<verif::AssertOp>(
            contractOp.getLoc(), ensureOp.getProperty(), nullptr, nullptr);
        rewriter.eraseOp(ensureOp);
      }
      rewriter.inlineBlockBefore(&contractOp.getBody().front(),
                                 &formalOp.getBody().front(),
                                 formalOp.getBody().front().end());
      for (auto [input, result] :
           llvm::zip(contractOp.getResults(), contractOp.getInputs())) {
        rewriter.replaceAllUsesWith(input, result);
      }
      rewriter.eraseOp(contractOp);
    }

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void LowerContractsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<HWOpRewritePattern>(patterns.getContext());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
