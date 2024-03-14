//===- IbisMethodsToContainers.cpp - Implementation of containerizing -----===//
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

#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace ibis;

namespace {

struct DataflowMethodOpConversion
    : public OpConversionPattern<DataflowMethodOp> {
  using OpConversionPattern<DataflowMethodOp>::OpConversionPattern;
  using OpAdaptor = typename DataflowMethodOp::Adaptor;

  LogicalResult
  matchAndRewrite(DataflowMethodOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the class by a container of the same name.
    auto newContainer = rewriter.create<ContainerOp>(
        op.getLoc(), op.getInnerSym().getSymName());
    rewriter.setInsertionPointToStart(newContainer.getBodyBlock());

    // Create mandatory %this
    // TODO @mortbopet: this will most likely already be present at the
    // method.df level soon...
    rewriter.create<ThisOp>(op.getLoc(), newContainer.getSymNameAttr());

    // Create in- and output ports.
    llvm::SmallVector<Value> argValues;
    for (auto [arg, name] : llvm::zip_equal(
             op.getArguments(), op.getArgNames().getAsRange<StringAttr>())) {
      auto port = rewriter.create<InputPortOp>(
          arg.getLoc(), hw::InnerSymAttr::get(name), arg.getType());
      argValues.push_back(rewriter.create<PortReadOp>(arg.getLoc(), port));
    }

    ReturnOp returnOp = cast<ReturnOp>(op.getBodyBlock()->getTerminator());
    for (auto [idx, resType] : llvm::enumerate(
             cast<MethodLikeOpInterface>(op.getOperation()).getResultTypes())) {
      auto port = rewriter.create<OutputPortOp>(
          op.getLoc(),
          hw::InnerSymAttr::get(
              rewriter.getStringAttr("out" + std::to_string(idx))),
          resType);
      rewriter.create<PortWriteOp>(op.getLoc(), port, returnOp.getOperand(idx));
    }

    rewriter.mergeBlocks(op.getBodyBlock(), newContainer.getBodyBlock(),
                         argValues);
    rewriter.eraseOp(op);
    rewriter.eraseOp(returnOp);
    return success();
  }
};

struct MethodsToContainersPass
    : public IbisConvertMethodsToContainersBase<MethodsToContainersPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void MethodsToContainersPass::runOnOperation() {
  auto *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<IbisDialect>();
  target.addIllegalOp<DataflowMethodOp>();
  target.addIllegalOp<ReturnOp>();
  RewritePatternSet patterns(context);

  patterns.insert<DataflowMethodOpConversion>(context);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createConvertMethodsToContainersPass() {
  return std::make_unique<MethodsToContainersPass>();
}
