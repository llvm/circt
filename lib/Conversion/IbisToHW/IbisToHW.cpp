//===- IbisToHW.cpp - Ibis to HW Conversion Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/IbisToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Ibis/Analysis/Hierarchy.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace ibis;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

struct ConvertContainerPattern : OpConversionPattern<ContainerOp> {
  ConvertContainerPattern(MLIRContext *ctx, Hierarchy &hierarchy)
      : OpConversionPattern(ctx), hierarchy(hierarchy) {}

  using OpAdaptor = typename ContainerOp::Adaptor;

  LogicalResult
  matchAndRewrite(ContainerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return success();
  }

  Hierarchy &hierarchy;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct IbisToHWPass : public ConvertIbisToHWBase<IbisToHWPass> {
  void runOnOperation() override;
};
} // namespace

void IbisToHWPass::runOnOperation() {
  auto &analysis = getAnalysis<Hierarchy>();

  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addIllegalOp<ContainerOp>();
  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertContainerPattern>(&getContext(), analysis);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
circt::createConvertIbisToHWPass() {
  return std::make_unique<IbisToHWPass>();
}
