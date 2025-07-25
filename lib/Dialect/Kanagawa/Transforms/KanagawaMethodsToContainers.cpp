//===- KanagawaMethodsToContainers.cpp - Implementation of containerizing -===//
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

#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWACONVERTMETHODSTOCONTAINERS
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace circt;
using namespace kanagawa;

namespace {

struct DataflowMethodOpConversion
    : public OpConversionPattern<DataflowMethodOp> {
  using OpConversionPattern<DataflowMethodOp>::OpConversionPattern;
  using OpAdaptor = typename DataflowMethodOp::Adaptor;

  LogicalResult
  matchAndRewrite(DataflowMethodOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the class by a container of the same name.
    auto newContainer = ContainerOp::create(
        rewriter, op.getLoc(), op.getInnerSym(), /*isTopLevel=*/false);
    rewriter.setInsertionPointToStart(newContainer.getBodyBlock());

    // Create in- and output ports.
    llvm::SmallVector<Value> argValues;
    for (auto [arg, name] : llvm::zip_equal(
             op.getArguments(), op.getArgNames().getAsRange<StringAttr>())) {
      auto port =
          InputPortOp::create(rewriter, arg.getLoc(),
                              hw::InnerSymAttr::get(name), arg.getType(), name);
      argValues.push_back(PortReadOp::create(rewriter, arg.getLoc(), port));
    }

    ReturnOp returnOp = cast<ReturnOp>(op.getBodyBlock()->getTerminator());
    for (auto [idx, resType] : llvm::enumerate(
             cast<MethodLikeOpInterface>(op.getOperation()).getResultTypes())) {
      auto portName = rewriter.getStringAttr("out" + std::to_string(idx));
      auto port = OutputPortOp::create(rewriter, op.getLoc(),
                                       hw::InnerSymAttr::get(portName), resType,
                                       portName);
      PortWriteOp::create(rewriter, op.getLoc(), port,
                          returnOp.getOperand(idx));
    }

    rewriter.mergeBlocks(op.getBodyBlock(), newContainer.getBodyBlock(),
                         argValues);
    rewriter.eraseOp(op);
    rewriter.eraseOp(returnOp);
    return success();
  }
};

struct MethodsToContainersPass
    : public circt::kanagawa::impl::KanagawaConvertMethodsToContainersBase<
          MethodsToContainersPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void MethodsToContainersPass::runOnOperation() {
  auto *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<KanagawaDialect>();
  target.addIllegalOp<DataflowMethodOp>();
  target.addIllegalOp<ReturnOp>();
  RewritePatternSet patterns(context);

  patterns.insert<DataflowMethodOpConversion>(context);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::kanagawa::createConvertMethodsToContainersPass() {
  return std::make_unique<MethodsToContainersPass>();
}
