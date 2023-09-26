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
#include "circt/Support/InstanceGraph.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace ibis;
using namespace circt::igraph;

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
  InputPortOpConversionPattern(MLIRContext *context, InstanceGraph &ig)
      : OpConversionPattern<InputPortOp>(context), ig(ig) {}

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

    // Since Ibis allows for input ports to be read from outside the container,
    // we need to check the instance graph to see whether this is the case.
    // If so, we need to add an output port to the container and connect it to
    // the assigned value.
    auto parentModuleOp = dyn_cast<ModuleOpInterface>(op->getParentOp());
    if (parentModuleOp) {
      InstanceGraphNode *node = ig.lookup(parentModuleOp);
      bool anyOutsideReads = llvm::any_of(node->uses(), [&](auto use) {
        Block *userBlock = use->getInstance()->getBlock();
        for (auto getPortOp : userBlock->getOps<GetPortOp>()) {
          if (getPortOp.getPortSymbol() == op.getPortName())
            return true;
        }
        return false;
      });

      if (anyOutsideReads) {
        auto outputPort = rewriter.create<OutputPortOp>(
            op.getLoc(), op.getSymName(), op.getType());
        rewriter.create<PortWriteOp>(op.getLoc(), outputPort, wire);
      }
    }

    // Finally, erase the writer and input port.
    rewriter.eraseOp(op);
    rewriter.eraseOp(writer);
    return success();
  }

protected:
  InstanceGraph &ig;
};

struct CleanSelfdriversPass
    : public IbisCleanSelfdriversBase<CleanSelfdriversPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void CleanSelfdriversPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addLegalDialect<IbisDialect>();
  target.addLegalOp<hw::WireOp>();
  target.addDynamicallyLegalOp<InputPortOp>(
      [](InputPortOp op) { return !isSelfDriven(op); });

  auto &ig = getAnalysis<InstanceGraph>();
  RewritePatternSet patterns(ctx);
  patterns.add<InputPortOpConversionPattern>(ctx, ig);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createCleanSelfdriversPass() {
  return std::make_unique<CleanSelfdriversPass>();
}
