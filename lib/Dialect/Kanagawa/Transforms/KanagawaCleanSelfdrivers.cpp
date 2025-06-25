//===- KanagawaCleanSelfdrivers.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"
#include "circt/Support/InstanceGraph.h"
#include "llvm/Support/Debug.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "kanagawa-clean-selfdrivers"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWACLEANSELFDRIVERS
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace circt;
using namespace kanagawa;
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

// Rewrites cases where an input port is being read in the instantiating module.
// Replaces the input port read by the assignment value of the input port.
static LogicalResult replaceReadsOfWrites(ContainerOp containerOp) {
  // Partition out all of the get_port's wrt. their target port symbol.
  struct PortAccesses {
    GetPortOp getAsInput;
    GetPortOp getAsOutput;
    llvm::SmallVector<PortReadOp> reads;
    PortWriteOp writer;
  };
  llvm::DenseMap</*instance*/ Value,
                 /*portName*/ llvm::DenseMap<StringAttr, PortAccesses>>
      instancePortAccessMap;

  for (auto getPortOp : containerOp.getOps<GetPortOp>()) {
    PortAccesses &portAccesses =
        instancePortAccessMap[getPortOp.getInstance()]
                             [getPortOp.getPortSymbolAttr().getAttr()];
    if (getPortOp.getDirection() == Direction::Input) {
      if (portAccesses.getAsInput)
        return portAccesses.getAsInput.emitError("multiple input get_ports")
                   .attachNote(getPortOp.getLoc())
               << "redundant get_port here";
      portAccesses.getAsInput = getPortOp;
      for (auto *user : getPortOp->getUsers()) {
        if (auto writer = dyn_cast<PortWriteOp>(user)) {
          if (portAccesses.writer)
            return getPortOp.emitError(
                "multiple writers of the same input port");
          portAccesses.writer = writer;
        }
      }
    } else {
      if (portAccesses.getAsOutput)
        return portAccesses.getAsOutput.emitError("multiple get_port as output")
                   .attachNote(getPortOp.getLoc())
               << "redundant get_port here";
      portAccesses.getAsOutput = getPortOp;

      for (auto *user : getPortOp->getUsers()) {
        if (auto reader = dyn_cast<PortReadOp>(user))
          portAccesses.reads.push_back(reader);
      }
    }
  }

  for (auto &[instance, portAccessMap] : instancePortAccessMap) {
    for (auto &[portName, portAccesses] : portAccessMap) {
      // If the port is not written to, nothing to do.
      if (!portAccesses.writer)
        continue;

      // if the port is not read, nothing to do.
      if (!portAccesses.getAsOutput)
        continue;

      // If the input port is self-driven, we need to replace all reads of the
      // input port with the value that is being written to it.
      LLVM_DEBUG(llvm::dbgs() << "Writer is: " << portAccesses.writer << "\n";);
      for (auto reader : portAccesses.reads) {
        LLVM_DEBUG(llvm::dbgs() << "Replacing: " << reader << "\n";);
        reader.replaceAllUsesWith(portAccesses.writer.getValue());
        reader.erase();
      }
      portAccesses.getAsOutput.erase();
    }
  }

  return success();
}

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
                                            op.getInnerSymAttrName());

    // Replace all reads of the input port with the wire.
    for (auto reader : readers)
      rewriter.replaceOp(reader, wire);

    // Since Kanagawa allows for input ports to be read from outside the
    // container, we need to check the instance graph to see whether this is the
    // case. If so, we need to add an output port to the container and connect
    // it to the assigned value.
    auto parentModuleOp = dyn_cast<ModuleOpInterface>(op->getParentOp());
    if (parentModuleOp) {
      InstanceGraphNode *node = ig.lookup(parentModuleOp);
      bool anyOutsideReads = llvm::any_of(node->uses(), [&](auto use) {
        Block *userBlock = use->getInstance()->getBlock();
        for (auto getPortOp : userBlock->getOps<GetPortOp>()) {
          if (getPortOp.getPortSymbol() == *op.getInnerName()) {
            return true;
          }
        }
        return false;
      });

      if (anyOutsideReads) {
        auto outputPort = rewriter.create<OutputPortOp>(
            op.getLoc(), op.getInnerSym(), op.getType(), op.getNameAttr());
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
    : public circt::kanagawa::impl::KanagawaCleanSelfdriversBase<
          CleanSelfdriversPass> {
  void runOnOperation() override;

  LogicalResult cleanInstanceSide();
  LogicalResult cleanContainerSide();
};
} // anonymous namespace

LogicalResult CleanSelfdriversPass::cleanInstanceSide() {
  for (ContainerOp containerOp : getOperation().getOps<ContainerOp>())
    if (failed(replaceReadsOfWrites(containerOp)))
      return failure();

  return success();
}

LogicalResult CleanSelfdriversPass::cleanContainerSide() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addLegalDialect<KanagawaDialect>();
  target.addLegalOp<hw::WireOp>();
  target.addDynamicallyLegalOp<InputPortOp>(
      [](InputPortOp op) { return !isSelfDriven(op); });

  auto &ig = getAnalysis<InstanceGraph>();
  RewritePatternSet patterns(ctx);
  patterns.add<InputPortOpConversionPattern>(ctx, ig);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return failure();

  return success();
}

void CleanSelfdriversPass::runOnOperation() {
  if (failed(cleanInstanceSide()) || failed(cleanContainerSide()))
    return signalPassFailure();
}

std::unique_ptr<Pass> circt::kanagawa::createCleanSelfdriversPass() {
  return std::make_unique<CleanSelfdriversPass>();
}
