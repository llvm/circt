//===- LowerContracts.cpp - Formal Preparations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower contracts into verif.formal tests.
//
//===----------------------------------------------------------------------===//

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

template <typename TO>
void replaceContractOp(PatternRewriter &rewriter, RequireLike &op) {
  StringAttr labelAttr;
  if (auto label = op.getLabel())
    labelAttr = rewriter.getStringAttr(label.value());

  rewriter.replaceOpWithNewOp<TO>(op, op.getProperty(), op.getEnable(),
                                  labelAttr);
}

struct HWOpRewritePattern : public OpRewritePattern<HWModuleOp> {
  using OpRewritePattern<HWModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HWModuleOp op,
                                PatternRewriter &rewriter) const override {
    auto contracts = op.getBody().front().getOps<ContractOp>();
    if (contracts.empty()) {
      return failure();
    }

    auto name =
        rewriter.getStringAttr(op.getNameAttr().getValue() + "_CheckContract");
    auto formalOp = rewriter.create<verif::FormalOp>(
        op.getLoc(), name, rewriter.getDictionaryAttr({}));

    // Clone module body into formal op body
    rewriter.cloneRegionBefore(op.getRegion(), formalOp.getBody(),
                               formalOp.getBody().end());

    auto *bodyBlock = &formalOp.getBody().front();

    // Erase hw.output
    rewriter.eraseOp(bodyBlock->getTerminator());

    // Convert block args to symbolic values
    rewriter.setInsertionPointToStart(bodyBlock);
    for (auto arg : llvm::make_early_inc_range(bodyBlock->getArguments())) {
      auto sym =
          rewriter.create<verif::SymbolicValueOp>(arg.getLoc(), arg.getType());
      rewriter.replaceAllUsesWith(arg, sym);
    }
    bodyBlock->eraseArguments(0, bodyBlock->getNumArguments());

    // Inline contract ops
    for (auto contractOp :
         llvm::make_early_inc_range(bodyBlock->getOps<ContractOp>())) {

      // Convert ensure to assert, require to assume
      Block *contractBlock = &contractOp.getBody().front();
      rewriter.setInsertionPointToEnd(contractBlock);

      WalkResult result = contractBlock->walk([&](RequireLike op) {
        if (isa<EnsureOp>(op)) {
          replaceContractOp<AssertOp>(rewriter, op);
        } else if (isa<RequireOp>(op)) {
          replaceContractOp<AssumeOp>(rewriter, op);
        } else {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (result.wasInterrupted()) {
        return failure();
      }

      // Inline body
      rewriter.inlineBlockBefore(contractBlock, bodyBlock, bodyBlock->end());

      // Replace results with inputs and erase
      for (auto [input, result] :
           llvm::zip(contractOp.getResults(), contractOp.getInputs())) {
        rewriter.replaceAllUsesWith(input, result);
      }
      rewriter.eraseOp(contractOp);
    }

    return failure();
  }
};
} // namespace

void LowerContractsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<HWOpRewritePattern>(patterns.getContext());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
