//===- Buffers.cpp - buffer materialization passes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of buffer materialization passes.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace {

struct RemoveHandshakeBuffers : public OpRewritePattern<handshake::BufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BufferOp bufferOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(bufferOp, bufferOp.operand());
    return success();
  }
};

struct HandshakeRemoveBuffersPass
    : public HandshakeRemoveBuffersBase<HandshakeRemoveBuffersPass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();
    ConversionTarget target(getContext());
    target.addIllegalOp<handshake::BufferOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<RemoveHandshakeBuffers>(&getContext());

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  };
};

} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeRemoveBuffersPass() {
  return std::make_unique<HandshakeRemoveBuffersPass>();
}
