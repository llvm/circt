//===- IbisPortrefLowering.cpp - Implementation of PortrefLowering --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ibis-lower-portrefs"

namespace circt {
namespace ibis {
#define GEN_PASS_DEF_IBISPORTREFLOWERING
#include "circt/Dialect/Ibis/IbisPasses.h.inc"
} // namespace ibis
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace ibis;

namespace {

class InputPortConversionPattern : public OpConversionPattern<InputPortOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<InputPortOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(InputPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PortRefType innerPortRefType = cast<PortRefType>(op.getType());
    Type innerType = innerPortRefType.getPortType();
    Direction d = innerPortRefType.getDirection();

    // CSE check - CSE should have ensured that only a single port unwrapper was
    // present, so if this is not the case, the user should run
    // CSE. This goes for other assumptions in the following code -
    // we require a CSEd form to avoid having to deal with a bunch of edge
    // cases.
    auto portrefUsers = op.getResult().getUsers();
    size_t nPortrefUsers =
        std::distance(portrefUsers.begin(), portrefUsers.end());
    if (nPortrefUsers != 1)
      return rewriter.notifyMatchFailure(
          op, "expected a single ibis.port.read as the only user of the input "
              "port reference, but found multiple readers - please run CSE "
              "prior to this pass");

    // A single PortReadOp should be present, which unwraps the portref<portref>
    // into a portref.
    PortReadOp portUnwrapper = dyn_cast<PortReadOp>(*portrefUsers.begin());
    if (!portUnwrapper)
      return rewriter.notifyMatchFailure(
          op, "expected a single ibis.port.read as the only user of the input "
              "port reference");

    // Replace the inner portref + port access with a "raw" port.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    if (d == Direction::Input) {
      // references to inputs becomes outputs (write from this container)
      auto rawOutput = rewriter.create<OutputPortOp>(
          op.getLoc(), op.getInnerSym(), innerType, op.getNameAttr());

      // Replace writes to the unwrapped port with writes to the new port.
      for (auto *unwrappedPortUser :
           llvm::make_early_inc_range(portUnwrapper.getResult().getUsers())) {
        PortWriteOp portWriter = dyn_cast<PortWriteOp>(unwrappedPortUser);
        if (!portWriter || portWriter.getPort() != portUnwrapper.getResult())
          continue;

        // Replace the source port of the write op with the new port.
        rewriter.replaceOpWithNewOp<PortWriteOp>(portWriter, rawOutput,
                                                 portWriter.getValue());
      }
    } else {
      // References to outputs becomes inputs (read from this container)
      auto rawInput = rewriter.create<InputPortOp>(
          op.getLoc(), op.getInnerSym(), innerType, op.getNameAttr());
      // TODO: RewriterBase::replaceAllUsesWith is not currently supported by
      // DialectConversion. Using it may lead to assertions about mutating
      // replaced/erased ops. For now, do this RAUW directly, until
      // ConversionPatternRewriter properly supports RAUW.
      // See https://github.com/llvm/circt/issues/6795.
      portUnwrapper.getResult().replaceAllUsesWith(rawInput);

      // Replace all ibis.port.read ops with a read of the new input.
      for (auto *portUser :
           llvm::make_early_inc_range(portUnwrapper.getResult().getUsers())) {
        PortReadOp portReader = dyn_cast<PortReadOp>(portUser);
        if (!portReader || portReader.getPort() != portUnwrapper.getResult())
          continue;

        rewriter.replaceOpWithNewOp<PortReadOp>(portReader, rawInput);
      }
    }

    // Finally, remove the port unwrapper and the original input port.
    rewriter.eraseOp(portUnwrapper);
    rewriter.eraseOp(op);

    return success();
  }
};

class OutputPortConversionPattern : public OpConversionPattern<OutputPortOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OutputPortOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OutputPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PortRefType innerPortRefType = cast<PortRefType>(op.getType());
    Type innerType = innerPortRefType.getPortType();
    Direction d = innerPortRefType.getDirection();

    // Locate the portwrapper - this is a writeOp with the output portref as
    // the portref value.
    PortWriteOp portWrapper;
    for (auto *user : op.getResult().getUsers()) {
      auto writeOp = dyn_cast<PortWriteOp>(user);
      if (writeOp && writeOp.getPort() == op.getResult()) {
        if (portWrapper)
          return rewriter.notifyMatchFailure(
              op, "expected a single ibis.port.write to wrap the output "
                  "portref, but found multiple");
        portWrapper = writeOp;
        break;
      }
    }

    if (!portWrapper)
      return rewriter.notifyMatchFailure(
          op, "expected an ibis.port.write to wrap the output portref");

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    if (d == Direction::Input) {
      // Outputs of inputs are inputs (external driver into this container).
      // Create the raw input port and write the input port reference with a
      // read of the raw input port.
      auto rawInput = rewriter.create<InputPortOp>(
          op.getLoc(), op.getInnerSym(), innerType, op.getNameAttr());
      rewriter.create<PortWriteOp>(
          op.getLoc(), portWrapper.getValue(),
          rewriter.create<PortReadOp>(op.getLoc(), rawInput));
    } else {
      // Outputs of outputs are outputs (external driver out of this container).
      // Create the raw output port and do a read of the input port reference.
      auto rawOutput = rewriter.create<OutputPortOp>(
          op.getLoc(), op.getInnerSym(), innerType, op.getNameAttr());
      rewriter.create<PortWriteOp>(
          op.getLoc(), rawOutput,
          rewriter.create<PortReadOp>(op.getLoc(), portWrapper.getValue()));
    }

    // Finally, remove the port wrapper and the original output port.
    rewriter.eraseOp(portWrapper);
    rewriter.eraseOp(op);

    return success();
  }
};

class GetPortConversionPattern : public OpConversionPattern<GetPortOp> {
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<GetPortOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(GetPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PortRefType outerPortRefType = cast<PortRefType>(op.getType());
    PortRefType innerPortRefType =
        cast<PortRefType>(outerPortRefType.getPortType());
    Type innerType = innerPortRefType.getPortType();

    Direction outerDirection = outerPortRefType.getDirection();
    Direction innerDirection = innerPortRefType.getDirection();

    StringAttr portName = op.getPortSymbolAttr().getAttr();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    Operation *wrapper;
    if (outerDirection == Direction::Input) {
      // Locate the get_port wrapper - this is a WriteOp with the get_port
      // result as the portref value.
      PortWriteOp getPortWrapper;
      for (auto *user : op.getResult().getUsers()) {
        auto writeOp = dyn_cast<PortWriteOp>(user);
        if (!writeOp || writeOp.getPort() != op.getResult())
          continue;

        getPortWrapper = writeOp;
        break;
      }

      if (!getPortWrapper)
        return rewriter.notifyMatchFailure(
            op, "expected an ibis.port.write to wrap the get_port result");
      wrapper = getPortWrapper;
      LLVM_DEBUG(llvm::dbgs() << "Found wrapper: " << *wrapper);
      if (innerDirection == Direction::Input) {
        // The portref<in portref<in T>> is now an output port.
        auto newGetPort =
            rewriter.create<GetPortOp>(op.getLoc(), op.getInstance(), portName,
                                       innerType, Direction::Output);
        auto newGetPortVal =
            rewriter.create<PortReadOp>(op.getLoc(), newGetPort);
        rewriter.create<PortWriteOp>(op.getLoc(), getPortWrapper.getValue(),
                                     newGetPortVal);
      } else {
        // The portref<in portref<out T>> is now an input port.
        auto newGetPort =
            rewriter.create<GetPortOp>(op.getLoc(), op.getInstance(), portName,
                                       innerType, Direction::Input);
        auto writeValue =
            rewriter.create<PortReadOp>(op.getLoc(), getPortWrapper.getValue());
        rewriter.create<PortWriteOp>(op.getLoc(), newGetPort, writeValue);
      }
    } else {
      PortReadOp getPortUnwrapper;
      for (auto *user : op.getResult().getUsers()) {
        auto readOp = dyn_cast<PortReadOp>(user);
        if (!readOp || readOp.getPort() != op.getResult())
          continue;

        getPortUnwrapper = readOp;
        break;
      }

      if (!getPortUnwrapper)
        return rewriter.notifyMatchFailure(
            op, "expected an ibis.port.read to unwrap the get_port result");
      wrapper = getPortUnwrapper;

      LLVM_DEBUG(llvm::dbgs() << "Found unwrapper: " << *wrapper);
      if (innerDirection == Direction::Input) {
        // In this situation, we're retrieving an input port that is sent as an
        // output of the container: %rr = ibis.get_port %c %c_in :
        // !ibis.scoperef<...> -> !ibis.portref<out !ibis.portref<in T>>
        //
        // Thus we expect one of these cases:
        // (always). a read op which unwraps the portref<out portref<in T>> into
        // a portref<in T>
        //    %r = ibis.port.read %rr : !ibis.portref<out !ibis.portref<in T>>
        // either:
        // 1. A write to %r which drives the target input port
        //    ibis.port.write %r, %someValue : !ibis.portref<in T>
        // 2. A write using %r which forwards the input port reference
        //    ibis.port.write %r_fw, %r : !ibis.portref<out !ibis.portref<in
        //    T>>
        //
        PortWriteOp portDriver;
        PortWriteOp portForwardingDriver;
        for (auto *user : getPortUnwrapper.getResult().getUsers()) {
          auto writeOp = dyn_cast<PortWriteOp>(user);
          if (!writeOp)
            continue;

          bool isForwarding = writeOp.getPort() != getPortUnwrapper.getResult();
          if (isForwarding) {
            if (portForwardingDriver)
              return rewriter.notifyMatchFailure(
                  op, "expected a single ibis.port.write to use the unwrapped "
                      "get_port result, but found multiple");
            portForwardingDriver = writeOp;
            LLVM_DEBUG(llvm::dbgs()
                       << "Found forwarding driver: " << *portForwardingDriver);
          } else {
            if (portDriver)
              return rewriter.notifyMatchFailure(
                  op, "expected a single ibis.port.write to use the unwrapped "
                      "get_port result, but found multiple");
            portDriver = writeOp;
            LLVM_DEBUG(llvm::dbgs() << "Found driver: " << *portDriver);
          }
        }

        if (!portDriver && !portForwardingDriver)
          return rewriter.notifyMatchFailure(
              op, "expected an ibis.port.write to drive the unwrapped get_port "
                  "result");

        Value portDriverValue;
        if (portForwardingDriver) {
          // In the case of forwarding, it is simplest to just create a new
          // input port, and write the forwarded value to it. This will allow
          // this pattern to recurse and eventually reach the case where the
          // forwarding is resolved through reading/writing the intermediate
          // inputs.
          auto fwPortName = rewriter.getStringAttr(portName.strref() + "_fw");
          auto forwardedInputPort = rewriter.create<InputPortOp>(
              op.getLoc(), hw::InnerSymAttr::get(fwPortName), innerType,
              fwPortName);

          // TODO: RewriterBase::replaceAllUsesWith is not currently supported
          // by DialectConversion. Using it may lead to assertions about
          // mutating replaced/erased ops. For now, do this RAUW directly, until
          // ConversionPatternRewriter properly supports RAUW.
          // See https://github.com/llvm/circt/issues/6795.
          getPortUnwrapper.getResult().replaceAllUsesWith(forwardedInputPort);
          portDriverValue = rewriter.create<PortReadOp>(
              op.getLoc(), forwardedInputPort.getPort());
        } else {
          // Direct assignmenet - the driver value will be the value of
          // the driver.
          portDriverValue = portDriver.getValue();
          rewriter.eraseOp(portDriver);
        }

        // Perform assignment to the input port of the target instance using
        // the driver value.
        auto rawPort =
            rewriter.create<GetPortOp>(op.getLoc(), op.getInstance(), portName,
                                       innerType, Direction::Input);
        rewriter.create<PortWriteOp>(op.getLoc(), rawPort, portDriverValue);
      } else {
        // In this situation, we're retrieving an output port that is sent as an
        // output of the container: %rr = ibis.get_port %c %c_in :
        // !ibis.scoperef<...> -> !ibis.portref<out !ibis.portref<out T>>
        //
        // Thus we expect two ops to be present:
        // 1. a read op which unwraps the portref<out portref<in T>> into a
        //    portref<in T>
        //      %r = ibis.port.read %rr : !ibis.portref<out !ibis.portref<in T>>
        // 2. one (or multiple, if not CSEd)
        //
        // We then replace the read op with the actual output port of the
        // container.
        auto rawPort =
            rewriter.create<GetPortOp>(op.getLoc(), op.getInstance(), portName,
                                       innerType, Direction::Output);

        // TODO: RewriterBase::replaceAllUsesWith is not currently supported by
        // DialectConversion. Using it may lead to assertions about mutating
        // replaced/erased ops. For now, do this RAUW directly, until
        // ConversionPatternRewriter properly supports RAUW.
        // See https://github.com/llvm/circt/issues/6795.
        getPortUnwrapper.getResult().replaceAllUsesWith(rawPort);
      }
    }

    // Finally, remove the get_port op.
    rewriter.eraseOp(wrapper);
    rewriter.eraseOp(op);

    return success();
  }
};

struct PortrefLoweringPass
    : public circt::ibis::impl::IbisPortrefLoweringBase<PortrefLoweringPass> {
  void runOnOperation() override;
};

} // anonymous namespace

void PortrefLoweringPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<InputPortOp, OutputPortOp>();
  target.addLegalDialect<IbisDialect>();

  // Ports are legal when they do not have portref types anymore.
  target.addDynamicallyLegalOp<InputPortOp, OutputPortOp>([&](auto op) {
    PortRefType portType =
        cast<PortRefType>(cast<PortOpInterface>(op).getPort().getType());
    return !isa<PortRefType>(portType.getPortType());
  });

  PortReadOp op;

  // get_port's are legal when they do not have portref types anymore.
  target.addDynamicallyLegalOp<GetPortOp>([&](GetPortOp op) {
    PortRefType portType = cast<PortRefType>(op.getPort().getType());
    return !isa<PortRefType>(portType.getPortType());
  });

  RewritePatternSet patterns(ctx);
  patterns.add<InputPortConversionPattern, OutputPortConversionPattern,
               GetPortConversionPattern>(ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createPortrefLoweringPass() {
  return std::make_unique<PortrefLoweringPass>();
}
