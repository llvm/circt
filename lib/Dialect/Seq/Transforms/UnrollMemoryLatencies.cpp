//===- UnrollMemoryLatencies.cpp - seq.shiftreg lowering ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_UNROLLMEMORYLATENCIES
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {

/// Delays all of the addresses in the given value range using the given delay
/// and stores the delayed addresses in the given vector.
static void delayAddresses(ConversionPatternRewriter &rewriter, Location loc,
                           Value clk, Value ce, ValueRange addrs,
                           IntegerAttr delayAttr,
                           llvm::SmallVector<Value> &delayedAddrs) {

  // My understanding is that when multiple addresses are given, the same
  // data is written to each location. Therefore we must delay all of the
  // addresses individually.
  for (auto addr : addrs) {
    // delay the addr
    auto shiftreg = seq::ShiftRegOp::create(rewriter, loc, delayAttr, addr, clk,
                                            ce, {}, {}, {}, {}, {});
    // store new delayed addr
    delayedAddrs.push_back(shiftreg);
  }
}

/// Unrolling for the HLMem write ports. This reduces the latency attribute
/// of the port to 1 and implements that latency as a delay of n-1 on its
/// data and address inputs.
struct WritePortUnrolling : public OpConversionPattern<seq::WritePortOp> {
public:
  using OpConversionPattern::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(seq::WritePortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Compute the new latency attributes
    IntegerAttr latencyAttr = adaptor.getLatencyAttr();
    IntegerAttr delayAttr =
        IntegerAttr::get(latencyAttr.getType(), latencyAttr.getInt() - 1);
    IntegerAttr oneAttr = IntegerAttr::get(latencyAttr.getType(), 1);

    // Retrieve all useful arguments
    Value data = adaptor.getInData();
    ValueRange addrs = adaptor.getAddresses();
    Value clk = adaptor.getMemory().getDefiningOp()->getOperand(0);

    // Stores the generated shiftregisters for the addresses.
    llvm::SmallVector<Value> delayedAddrs;

    // A global clock enable for all ports that is always true
    Value ce = rewriter.createOrFold<hw::ConstantOp>(op.getLoc(),
                                                     rewriter.getI1Type(), 1);

    // Delay all of the addresses
    delayAddresses(rewriter, op.getLoc(), clk, ce, addrs, delayAttr,
                   delayedAddrs);

    // delay the data
    auto delayedData = seq::ShiftRegOp::create(
        rewriter, op.getLoc(), delayAttr, data, clk, ce, {}, {}, {}, {}, {});

    // Replace the old write port with the new one
    rewriter.replaceOpWithNewOp<seq::WritePortOp>(op, adaptor.getMemory(),
                                                  delayedAddrs, delayedData,
                                                  adaptor.getWrEn(), oneAttr);

    return success();
  }
};

/// Unrolling for the HLMem Read ports. This reduces the latency attribute
/// of the port to 0 and implements that latency as a delay of n, modeled
/// as a shift-register, on its address input.
struct ReadPortUnrolling : public OpConversionPattern<seq::ReadPortOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::ReadPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Compute the new latency attributes
    IntegerAttr latencyAttr = adaptor.getLatencyAttr();
    IntegerAttr zeroAttr = IntegerAttr::get(latencyAttr.getType(), 0);

    // Retrieve all useful arguments
    ValueRange addrs = adaptor.getAddresses();
    Value clk = adaptor.getMemory().getDefiningOp()->getOperand(0);

    // Stores the generated shiftregisters for the addresses.
    llvm::SmallVector<Value> delayedAddrs;

    // A global clock enable for all ports that is always true
    Value ce = rewriter.createOrFold<hw::ConstantOp>(op.getLoc(),
                                                     rewriter.getI1Type(), 1);

    // Same as write port, delay all of the addresses
    delayAddresses(rewriter, op.getLoc(), clk, ce, addrs, latencyAttr,
                   delayedAddrs);

    // Replace the old read port with the new one
    rewriter.replaceOpWithNewOp<seq::ReadPortOp>(
        op, op.getReadData().getType(), adaptor.getMemory(), delayedAddrs,
        adaptor.getRdEn(), zeroAttr);

    return success();
  }
};

/// Unrolling for FirMem latencies. Unlike HLMem, FirMem stores
/// latencies it the memory declaration op rather than the ports themselves.
/// This makes the unrolling a little more annoying, as we need to do fetch
/// all of the ports from the memory itself, i.e. walk its users to find the
/// all of the relevant ports to delay before updating the memory declaration
/// to have its new delays. The actual port updates are identical to those done
/// in the HLMem ports.
struct FirMemUnrolling : public OpConversionPattern<seq::FirMemOp> {
public:
  using OpConversionPattern::OpConversionPattern;

private:
  /// Performs the latency unrolling of an individual write port
  void unrollFirMemWritePort(seq::FirMemOp mem, seq::FirMemWriteOp &op,
                             IntegerAttr delayAttr, IntegerAttr oneAttr,
                             Value ce,
                             ConversionPatternRewriter &rewriter) const {

    // Retrieve all of the relevant arguments
    Value addr = op.getAddress();
    Value data = op.getData();
    Value clk = op.getClk();

    // Delay the address and data inputs
    auto delayedData = seq::ShiftRegOp::create(
        rewriter, op.getLoc(), delayAttr, data, clk, ce, {}, {}, {}, {}, {});

    auto delayedAddr = seq::ShiftRegOp::create(
        rewriter, op.getLoc(), delayAttr, addr, clk, ce, {}, {}, {}, {}, {});

    // Replace the current port with a delayed port
    rewriter.replaceOpWithNewOp<seq::FirMemWriteOp>(
        op, mem, delayedAddr, clk, op.getEnable(), delayedData, op.getMask());
  }

  /// Performs the latency unrolling of an individual read port
  void unrollFirMemReadPort(seq::FirMemOp mem, seq::FirMemReadOp &op,
                            IntegerAttr delayRdAttr, IntegerAttr zeroAttr,
                            Value ce,
                            ConversionPatternRewriter &rewriter) const {
    // Retrieve all of the relevant arguments
    Value addr = op.getAddress();
    Value clk = op.getClk();

    auto delayedAddr = seq::ShiftRegOp::create(
        rewriter, op.getLoc(), delayRdAttr, addr, clk, ce, {}, {}, {}, {}, {});

    // Replace the current port with a delayed port
    rewriter.replaceOpWithNewOp<seq::FirMemReadOp>(
        op, op.getResult().getType(), mem, delayedAddr, clk, op.getEnable());
  }

  /// Performs the latency unrolling of a read-write port.
  /// These are annoying as they require implementing both the read and write
  /// unrolling and then muxing the two delays using the mode as a selector.
  void unrollFirMemReadWritePort(seq::FirMemOp mem, seq::FirMemReadWriteOp &op,
                                 IntegerAttr delayWrAttr,
                                 IntegerAttr delayRdAttr, IntegerAttr zeroAttr,
                                 IntegerAttr oneAttr, Value ce,
                                 ConversionPatternRewriter &rewriter) const {
    // Retrieve all of the relevant arguments
    Value addr = op.getAddress();
    Value data = op.getWriteData();
    Value mode = op.getMode();
    Value clk = op.getClk();

    // Create two versions of the delayed address, one using the write latency,
    // one using the read latency
    auto delayedReadAddr = seq::ShiftRegOp::create(
        rewriter, op.getLoc(), delayRdAttr, addr, clk, ce, {}, {}, {}, {}, {});

    auto delayedWriteAddr = seq::ShiftRegOp::create(
        rewriter, op.getLoc(), delayWrAttr, addr, clk, ce, {}, {}, {}, {}, {});

    // Delay the write data, this will only be used in case of read so no need
    // for a second mux.
    auto delayedWriteData = seq::ShiftRegOp::create(
        rewriter, op.getLoc(), delayWrAttr, data, clk, ce, {}, {}, {}, {}, {});

    // Select which delayed version we want based on the mode
    auto selectAddr = comb::MuxOp::create(rewriter, op.getLoc(), mode,
                                          delayedWriteAddr, delayedReadAddr);

    // Replace the current port with a delayed port
    rewriter.replaceOpWithNewOp<seq::FirMemReadWriteOp>(
        op, op.getResult().getType(), mem, selectAddr, clk, op.getEnable(),
        delayedWriteData, mode, op.getMask());
  }

public:
  LogicalResult
  matchAndRewrite(seq::FirMemOp mem, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Retrieve read and write latencies
    IntegerAttr wrLatency = mem.getWriteLatencyAttr();
    IntegerAttr rdLatency = mem.getReadLatencyAttr();
    IntegerAttr delayRdAttr = rewriter.getI64IntegerAttr(rdLatency.getInt());
    IntegerAttr delayWrAttr =
        rewriter.getI64IntegerAttr(wrLatency.getInt() - 1);
    IntegerAttr zeroAttr = rewriter.getI32IntegerAttr(0);
    IntegerAttr oneAttr = rewriter.getI32IntegerAttr(1);

    // A global clock enable for all ports that is always true
    Value ce = rewriter.createOrFold<hw::ConstantOp>(mem.getLoc(),
                                                     rewriter.getI1Type(), 1);

    // Find all of the relevant ports and update them
    for (auto user : mem->getUsers()) {
      // Figure out what type of port it is an unroll it
      TypeSwitch<Operation *, void>(user)
          .Case<seq::FirMemWriteOp>([&](auto port) {
            unrollFirMemWritePort(mem, port, delayWrAttr, oneAttr, ce,
                                  rewriter);
          })
          .Case<seq::FirMemReadOp>([&](auto port) {
            unrollFirMemReadPort(mem, port, delayRdAttr, zeroAttr, ce,
                                 rewriter);
          })
          .Case<seq::FirMemReadWriteOp>([&](auto port) {
            unrollFirMemReadWritePort(mem, port, delayWrAttr, delayRdAttr,
                                      zeroAttr, oneAttr, ce, rewriter);
          })
          .Default([](auto) { /*ignore non-port users*/ });
    }

    // Update the memory's latency attributes
    mem.setReadLatencyAttr(zeroAttr);
    mem.setWriteLatencyAttr(oneAttr);

    return success();
  }
};

struct UnrollMemoryLatenciesPass
    : public circt::seq::impl::UnrollMemoryLatenciesBase<
          UnrollMemoryLatenciesPass> {
  void runOnOperation() override;
};

} // namespace

void UnrollMemoryLatenciesPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  target.addLegalDialect<seq::SeqDialect, hw::HWDialect, comb::CombDialect>();

  // resulting ports must have their latencies reset.
  target.addDynamicallyLegalOp<seq::ReadPortOp>(
      [](seq::ReadPortOp op) { return op.getLatencyAttr().getInt() == 0; });

  target.addDynamicallyLegalOp<seq::WritePortOp>(
      [](seq::WritePortOp op) { return op.getLatencyAttr().getInt() == 1; });

  target.addDynamicallyLegalOp<seq::FirMemOp>([](seq::FirMemOp op) {
    return op.getReadLatencyAttr().getInt() == 0 &&
           op.getWriteLatencyAttr().getInt() == 1;
  });

  RewritePatternSet patterns(&ctxt);
  patterns.add<WritePortUnrolling, ReadPortUnrolling, FirMemUnrolling>(&ctxt);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
