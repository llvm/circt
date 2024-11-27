//===- KanagawaInlineSBlocksPass.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iterator>

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWAINLINESBLOCKS
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace kanagawa;

namespace {

struct InlineSBlocksPass
    : public circt::kanagawa::impl::KanagawaInlineSBlocksBase<
          InlineSBlocksPass> {
  void runOnOperation() override;
};

class InlineSBlocksPattern
    : public OpConversionPattern<kanagawa::StaticBlockOp> {
public:
  using OpConversionPattern<kanagawa::StaticBlockOp>::OpConversionPattern;
  using OpAdaptor =
      typename OpConversionPattern<kanagawa::StaticBlockOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(kanagawa::StaticBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // We only create block markers in case the sblock has attributes.
    bool hasAttributes = !op->getAttrs().empty();

    if (hasAttributes) {
      // Start the inline block...
      auto inlineStart =
          rewriter.create<kanagawa::InlineStaticBlockBeginOp>(loc);
      inlineStart->setAttrs(op->getAttrs());
    }

    // Inline the sblock.
    Block *sblockBody = op.getBodyBlock();
    BlockReturnOp ret = cast<BlockReturnOp>(sblockBody->getTerminator());
    rewriter.inlineBlockBefore(sblockBody, op->getNextNode());

    // Replace the kanagawa.sblock return values with the values that were
    // defined (returned from) within the sblock body, and erase the return op.
    for (auto [res, val] : llvm::zip(op.getResults(), ret.getRetValues()))
      rewriter.replaceAllUsesWith(res, val);

    if (hasAttributes) {
      // Close the inline block
      rewriter.setInsertionPoint(ret);
      rewriter.create<kanagawa::InlineStaticBlockEndOp>(loc);
    }

    rewriter.eraseOp(ret);
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

void InlineSBlocksPass::runOnOperation() {
  MethodOp parent = getOperation();
  ConversionTarget target(getContext());
  target.addIllegalOp<kanagawa::StaticBlockOp>();
  // Mark everything but `kanagawa.sblock` as legal - we need a legalization
  // pattern for any possible op that gets inlined from a block.
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  RewritePatternSet patterns(&getContext());
  patterns.add<InlineSBlocksPattern>(&getContext());

  if (failed(applyPartialConversion(parent, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> circt::kanagawa::createInlineSBlocksPass() {
  return std::make_unique<InlineSBlocksPass>();
}
