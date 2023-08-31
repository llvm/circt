//===- IbisCleanSelfdrivers.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace ibis;

// Returns true if the given input port is self-driven, i.e. there exists
// a PortWriteOp that writes to it.
static bool isSelfDriven(InputPortOp op) {
  return llvm::any_of(op->getUsers(), [&](Operation *user) {
    auto writer = dyn_cast<PortWriteOp>(user);
    return writer && writer.getPort() == op.getPort();
  });
}

namespace {

struct InputPortOpConversionPattern : public OpConversionPattern<InputPortOp> {
  using OpConversionPattern<InputPortOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InputPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Locate the readers and writers of the port.
    PortWriteOp writer = nullptr;
    llvm::SmallVector<PortReadOp> readers;
    for (auto *user : op->getUsers()) {
      auto res = llvm::TypeSwitch<Operation *, LogicalResult>(user)
                     .Case<PortWriteOp>([&](auto op) {
                       if (writer)
                         return rewriter.notifyMatchFailure(
                             user, "found multiple drivers of the self-driven "
                                   "input port");
                       writer = op;
                       return success();
                     })
                     .Case<PortReadOp>([&](auto op) {
                       readers.push_back(op);
                       return success();
                     })
                     .Default([&](auto) {
                       return rewriter.notifyMatchFailure(
                           user, "unhandled user of the self-driven "
                                 "input port");
                     });

      if (failed(res))
        return failure();
    }

    // Create a `hw.wire` to ensure that the input port name is maintained.
    auto wire = rewriter.create<hw::WireOp>(op.getLoc(), writer.getValue(),
                                            op.getSymName());

    // Replace all reads of the input port with the wire.
    for (auto reader : readers)
      rewriter.replaceOp(reader, wire);

    rewriter.eraseOp(op);
    rewriter.eraseOp(writer);
    return success();
  }
};

struct CleanSelfdriversPass
    : public IbisCleanSelfdriversBase<CleanSelfdriversPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void CleanSelfdriversPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<InputPortOp>();
  target.addLegalOp<hw::WireOp>();

  target.addDynamicallyLegalOp<InputPortOp>(
      [](InputPortOp op) { return !isSelfDriven(op); });

  RewritePatternSet patterns(ctx);
  patterns.add<InputPortOpConversionPattern>(ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createCleanSelfdriversPass() {
  return std::make_unique<CleanSelfdriversPass>();
}
