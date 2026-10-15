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
#include "circt/Support/BackedgeBuilder.h"
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

    // A global clock enable that is always true
    Value ce =
        hw::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI1Type(), 1);

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

    // A global clock enable that is always true
    Value ce =
        hw::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI1Type(), 1);

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

/// Unrolling for FirMem latencies. Unlike HLMem, FirMem stores latencies
/// it the memory declaration op rather than the ports themselves. This
/// makes the unrolling a little more annoying, as we need to fetch the
/// latencies in the mem op, and then walk its users to find the all of the
/// relevant ports to delay. The delays are then indentical to those done
/// in the HLMem ports.
struct FirMemUnrolling : public OpConversionPattern<seq::FirMemOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::FirMemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
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

  target.addLegalDialect<seq::SeqDialect, hw::HWDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<WritePortUnrolling, ReadPortUnrolling, FirMemUnrolling>(&ctxt);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
