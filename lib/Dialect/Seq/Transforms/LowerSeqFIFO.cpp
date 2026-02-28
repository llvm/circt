//===- LowerSeqFIFO.cpp - seq.fifo lowering -------------------------------===//
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
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_LOWERSEQFIFO
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {

struct FIFOLowering : public OpConversionPattern<seq::FIFOOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::FIFOOp mem, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = mem.getLoc();
    Type eltType = adaptor.getInput().getType();
    Value clk = adaptor.getClk();
    Value rst = adaptor.getRst();
    BackedgeBuilder bb(rewriter, loc);
    size_t depth = mem.getDepth();
    Type countType = rewriter.getIntegerType(llvm::Log2_64_Ceil(depth + 1));
    Type ptrType = rewriter.getIntegerType(llvm::Log2_64_Ceil(depth));
    Backedge rdAddrNext = bb.get(ptrType);
    Backedge wrAddrNext = bb.get(ptrType);
    Backedge nextCount = bb.get(countType);

    // ====== Some constants ======
    Value countTcFull = hw::ConstantOp::create(rewriter, loc, countType, depth);
    Value countTc1 = hw::ConstantOp::create(rewriter, loc, countType, 1);
    Value countTc0 = hw::ConstantOp::create(rewriter, loc, countType, 0);
    Value ptrTc0 = hw::ConstantOp::create(rewriter, loc, ptrType, 0);
    Value ptrTc1 = hw::ConstantOp::create(rewriter, loc, ptrType, 1);
    Value ptrTcFull = hw::ConstantOp::create(rewriter, loc, ptrType, depth);

    // ====== Hardware units ======
    Value count = seq::CompRegOp::create(
        rewriter, loc, nextCount, clk, rst,
        hw::ConstantOp::create(rewriter, loc, countType, 0), "fifo_count");
    seq::HLMemOp hlmem = seq::HLMemOp::create(
        rewriter, loc, clk, rst, "fifo_mem",
        llvm::SmallVector<int64_t>{static_cast<int64_t>(depth)}, eltType);
    Value rdAddr = seq::CompRegOp::create(
        rewriter, loc, rdAddrNext, clk, rst,
        hw::ConstantOp::create(rewriter, loc, ptrType, 0), "fifo_rd_addr");
    Value wrAddr = seq::CompRegOp::create(
        rewriter, loc, wrAddrNext, clk, rst,
        hw::ConstantOp::create(rewriter, loc, ptrType, 0), "fifo_wr_addr");

    Value readData = seq::ReadPortOp::create(
        rewriter, loc, hlmem, llvm::SmallVector<Value>{rdAddr},
        adaptor.getRdEn(), mem.getRdLatency());
    seq::WritePortOp::create(rewriter, loc, hlmem,
                             llvm::SmallVector<Value>{wrAddr},
                             adaptor.getInput(), adaptor.getWrEn(),
                             /*latency*/ 1);

    // ====== some more constants =====
    comb::ICmpOp fifoFull = comb::ICmpOp::create(
        rewriter, loc, comb::ICmpPredicate::eq, count, countTcFull);
    fifoFull->setAttr("sv.namehint", rewriter.getStringAttr("fifo_full"));
    comb::ICmpOp fifoEmpty = comb::ICmpOp::create(
        rewriter, loc, comb::ICmpPredicate::eq, count, countTc0);
    fifoEmpty->setAttr("sv.namehint", rewriter.getStringAttr("fifo_empty"));

    // ====== Next-state count ======
    auto notRdEn = comb::createOrFoldNot(loc, adaptor.getRdEn(), rewriter);
    auto notWrEn = comb::createOrFoldNot(loc, adaptor.getWrEn(), rewriter);
    Value rdEnNandWrEn = comb::AndOp::create(rewriter, loc, notRdEn, notWrEn);
    Value rdEnAndNotWrEn =
        comb::AndOp::create(rewriter, loc, adaptor.getRdEn(), notWrEn);
    Value wrEnAndNotRdEn =
        comb::AndOp::create(rewriter, loc, adaptor.getWrEn(), notRdEn);

    auto countEqTcFull = comb::ICmpOp::create(
        rewriter, loc, comb::ICmpPredicate::eq, count, countTcFull);
    auto addCountTc1 = comb::AddOp::create(rewriter, loc, count, countTc1);
    Value wrEnNext = comb::MuxOp::create(rewriter, loc, countEqTcFull,
                                         // keep value
                                         count,
                                         // increment
                                         addCountTc1);
    auto countEqTc0 = comb::ICmpOp::create(
        rewriter, loc, comb::ICmpPredicate::eq, count, countTc0);
    auto subCountTc1 = comb::SubOp::create(rewriter, loc, count, countTc1);

    Value rdEnNext = comb::MuxOp::create(rewriter, loc, countEqTc0,
                                         // keep value
                                         count,
                                         // decrement
                                         subCountTc1);

    auto nextInnerMux =
        comb::MuxOp::create(rewriter, loc, rdEnAndNotWrEn, rdEnNext, count);
    auto nextMux = comb::MuxOp::create(rewriter, loc, wrEnAndNotRdEn, wrEnNext,
                                       nextInnerMux);
    nextCount.setValue(comb::MuxOp::create(rewriter, loc, rdEnNandWrEn,
                                           /*keep value*/ count, nextMux));
    static_cast<Value>(nextCount).getDefiningOp()->setAttr(
        "sv.namehint", rewriter.getStringAttr("fifo_count_next"));

    // ====== Read/write pointers ======
    Value wrAndNotFull =
        comb::AndOp::create(rewriter, loc, adaptor.getWrEn(),
                            comb::createOrFoldNot(loc, fifoFull, rewriter));
    auto addWrAddrPtrTc1 = comb::AddOp::create(rewriter, loc, wrAddr, ptrTc1);

    auto wrAddrNextNoRollover = comb::MuxOp::create(rewriter, loc, wrAndNotFull,
                                                    addWrAddrPtrTc1, wrAddr);
    auto isMaxAddrWr =
        comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::eq,
                             wrAddrNextNoRollover, ptrTcFull);
    wrAddrNext.setValue(comb::MuxOp::create(rewriter, loc, isMaxAddrWr, ptrTc0,
                                            wrAddrNextNoRollover));
    static_cast<Value>(wrAddrNext)
        .getDefiningOp()
        ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_wr_addr_next"));

    auto notFifoEmpty = comb::createOrFoldNot(loc, fifoEmpty, rewriter);
    Value rdAndNotEmpty =
        comb::AndOp::create(rewriter, loc, adaptor.getRdEn(), notFifoEmpty);
    auto addRdAddrPtrTc1 = comb::AddOp::create(rewriter, loc, rdAddr, ptrTc1);
    auto rdAddrNextNoRollover = comb::MuxOp::create(
        rewriter, loc, rdAndNotEmpty, addRdAddrPtrTc1, rdAddr);
    auto isMaxAddrRd =
        comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::eq,
                             rdAddrNextNoRollover, ptrTcFull);
    rdAddrNext.setValue(comb::MuxOp::create(rewriter, loc, isMaxAddrRd, ptrTc0,
                                            rdAddrNextNoRollover));
    static_cast<Value>(rdAddrNext)
        .getDefiningOp()
        ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_rd_addr_next"));

    // ====== Result values ======
    llvm::SmallVector<Value> results;

    // Data
    results.push_back(readData);
    // Full
    results.push_back(fifoFull);
    // Empty
    results.push_back(fifoEmpty);

    if (auto almostFull = mem.getAlmostFullThreshold()) {
      results.push_back(
          comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::uge, count,
                               hw::ConstantOp::create(rewriter, loc, countType,
                                                      almostFull.value())));
      static_cast<Value>(results.back())
          .getDefiningOp()
          ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_almost_full"));
    }

    if (auto almostEmpty = mem.getAlmostEmptyThreshold()) {
      results.push_back(
          comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::ule, count,
                               hw::ConstantOp::create(rewriter, loc, countType,
                                                      almostEmpty.value())));
      static_cast<Value>(results.back())
          .getDefiningOp()
          ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_almost_empty"));
    }

    // ====== Protocol checks =====
    Value clkI1 = seq::FromClockOp::create(rewriter, loc, clk);
    Value notEmptyAndRden = comb::createOrFoldNot(
        loc, comb::AndOp::create(rewriter, loc, adaptor.getRdEn(), fifoEmpty),
        rewriter);
    verif::ClockedAssertOp::create(
        rewriter, loc, notEmptyAndRden, verif::ClockEdge::Pos, clkI1,
        /*enable=*/Value(),
        rewriter.getStringAttr("FIFO empty when read enabled"));
    Value notFullAndWren = comb::createOrFoldNot(
        loc, comb::AndOp::create(rewriter, loc, adaptor.getWrEn(), fifoFull),
        rewriter);
    verif::ClockedAssertOp::create(
        rewriter, loc, notFullAndWren, verif::ClockEdge::Pos, clkI1,
        /*enable=*/Value(),
        rewriter.getStringAttr("FIFO full when write enabled"));

    rewriter.replaceOp(mem, results);
    return success();
  }
};

struct LowerSeqFIFOPass
    : public circt::seq::impl::LowerSeqFIFOBase<LowerSeqFIFOPass> {
  void runOnOperation() override;
};

} // namespace

void LowerSeqFIFOPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  // Lowering patterns must lower away all HLMem-related operations.
  target.addIllegalOp<seq::FIFOOp>();
  target.addLegalDialect<seq::SeqDialect, hw::HWDialect, comb::CombDialect,
                         verif::VerifDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<FIFOLowering>(&ctxt);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::seq::createLowerSeqFIFOPass() {
  return std::make_unique<LowerSeqFIFOPass>();
}
