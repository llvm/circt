//===- IbisArgifyBlocksPass.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace ibis;

namespace {

// A mapping between values and op operands who use them. A MapVector is used to
// ensure determinism.
using ValueMapping = llvm::MapVector<Value, llvm::SmallVector<OpOperand *>>;

// Returns a map of values to op operands, of values that are defined
// outside of the block op.
static void getExternallyDefinedOperands(StaticBlockOp blockOp,
                                         ValueMapping &mapping) {
  Block *blockBodyBlock = blockOp.getBodyBlock();
  for (Operation &op : *blockBodyBlock) {
    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      if (v.getParentBlock() != blockBodyBlock)
        mapping[v].push_back(&operand);
    }
  }
}

struct BlockConversionPattern : public OpConversionPattern<StaticBlockOp> {
  using OpConversionPattern<StaticBlockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticBlockOp blockOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueMapping mapping;
    getExternallyDefinedOperands(blockOp, mapping);
    Block *bodyBlock = blockOp.getBodyBlock();

    rewriter.updateRootInPlace(blockOp, [&]() {
      // Add inputs and block arguments to the block, and replace the operand
      // uses.
      for (auto &[value, uses] : mapping) {
        blockOp.getInputsMutable().append({value});
        auto newArg = bodyBlock->addArgument(value.getType(), value.getLoc());
        for (OpOperand *operand : uses)
          operand->set(newArg);
      }
    });

    return success();
  }
};

struct ArgifyBlocksPass : public IbisArgifyBlocksBase<ArgifyBlocksPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ArgifyBlocksPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<StaticBlockOp>([](StaticBlockOp op) {
    ValueMapping mapping;
    getExternallyDefinedOperands(op, mapping);
    return mapping.empty();
  });

  RewritePatternSet patterns(ctx);
  patterns.add<BlockConversionPattern>(ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createArgifyBlocksPass() {
  return std::make_unique<ArgifyBlocksPass>();
}
