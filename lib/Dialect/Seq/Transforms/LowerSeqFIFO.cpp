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
#include "llvm/ADT/DenseMap.h"
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

  // Every status output is a register, and every comparison is of a register
  // against a constant. The caller's enables, which usually arrive late, only
  // select between values computed from registers, and the occupancy count,
  // which is not an output, does not see them at all. Reading while empty and
  // writing while full are protocol violations, checked by assertions rather
  // than guarded against, so the enables are used as they are.
  LogicalResult
  matchAndRewrite(seq::FIFOOp mem, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = mem.getLoc();
    Type eltType = adaptor.getInput().getType();
    Value clk = adaptor.getClk();
    Value rst = adaptor.getRst();
    Value rdEn = adaptor.getRdEn();
    Value wrEn = adaptor.getWrEn();
    BackedgeBuilder bb(rewriter, loc);
    uint64_t depth = mem.getDepth();
    Type i1Type = rewriter.getI1Type();
    Type countType = rewriter.getIntegerType(llvm::Log2_64_Ceil(depth + 1));
    // A depth one short of a power of two (from 3) gets a memory of that power
    // of two. The spare slot is the one at the write pointer, which never holds
    // an entry, so the memory is written every cycle, without a write enable,
    // and the pointers wrap on their own. In FPGA block RAM the extra entry is
    // free.
    bool spareSlot = depth >= 3 && llvm::isPowerOf2_64(depth + 1);
    uint64_t memDepth = spareSlot ? depth + 1 : depth;
    unsigned ptrWidth = llvm::Log2_64_Ceil(memDepth);
    Type ptrType = rewriter.getIntegerType(ptrWidth);

    // Operands that create ops are built into locals before they are combined:
    // the order in which C++ evaluates arguments is unspecified, and the order
    // of the ops must not depend on the compiler.
    auto constant = [&](Type type, uint64_t value) -> Value {
      return hw::ConstantOp::create(rewriter, loc, type, value);
    };
    auto mux = [&](Value cond, Value trueValue, Value falseValue) -> Value {
      return comb::MuxOp::create(rewriter, loc, cond, trueValue, falseValue);
    };
    auto setName = [&](Value value, StringRef name) {
      value.getDefiningOp()->setAttr("sv.namehint",
                                     rewriter.getStringAttr(name));
    };
    auto makeReg = [&](Value next, Type type, uint64_t resetValue,
                       StringRef name) -> Value {
      return seq::CompRegOp::create(rewriter, loc, next, clk, rst,
                                    constant(type, resetValue), name);
    };

    // ====== Hardware units ======
    Backedge countNext = bb.get(countType);
    Backedge rdAddrNext = bb.get(ptrType);
    Backedge wrAddrNext = bb.get(ptrType);
    Value countLag = makeReg(countNext, countType, 0, "fifo_count_lag");
    // The memory is not reset, which keeps the reset off its write enable. The
    // FIFO leaves reset empty, so nothing written during reset is ever read.
    seq::HLMemOp hlmem = seq::HLMemOp::create(
        rewriter, loc, clk, constant(i1Type, 0), "fifo_mem",
        llvm::SmallVector<int64_t>{static_cast<int64_t>(memDepth)}, eltType);
    Value rdAddr = makeReg(rdAddrNext, ptrType, 0, "fifo_rd_addr");
    Value wrAddr = makeReg(wrAddrNext, ptrType, 0, "fifo_wr_addr");

    // The memory is read in the cycle an entry is popped, with latency 1 at
    // most; any further latency is registers on the read data. So a popped
    // slot is free at once: no read of it is still to come when it is
    // rewritten. Delaying the read address instead would also leave a RAM
    // block's output register empty.
    uint64_t rdLatency = mem.getRdLatency();
    Value readData = seq::ReadPortOp::create(
        rewriter, loc, hlmem, llvm::SmallVector<Value>{rdAddr}, rdEn,
        std::min<uint64_t>(rdLatency, 1));
    for (uint64_t i = 1; i < rdLatency; ++i)
      readData = seq::CompRegOp::create(
          rewriter, loc, readData, clk,
          rewriter.getStringAttr("fifo_mem_rd_dly" + std::to_string(i)));
    seq::WritePortOp::create(
        rewriter, loc, hlmem, llvm::SmallVector<Value>{wrAddr},
        adaptor.getInput(), spareSlot ? constant(i1Type, 1) : wrEn,
        /*latency*/ 1);

    // ====== Occupancy ======
    Value countUp = comb::AndOp::create(
        rewriter, loc, wrEn, comb::createOrFoldNot(rewriter, loc, rdEn));
    Value countDown = comb::AndOp::create(
        rewriter, loc, rdEn, comb::createOrFoldNot(rewriter, loc, wrEn));
    // The next value of `current`, a function of the count, given its values
    // after the count goes up or down by one.
    auto step = [&](Value current, Value afterUp, Value afterDown) -> Value {
      return mux(countUp, afterUp, mux(countDown, afterDown, current));
    };
    Value trueValue = constant(i1Type, 1);
    Value falseValue = constant(i1Type, 0);

    // The count register lags a cycle behind: it is stepped by registered
    // copies of `countUp` and `countDown`, so the enables never reach its
    // adder. The count this cycle is the register plus that step.
    Value countUpLag = makeReg(countUp, i1Type, 0, "fifo_count_up");
    Value countDownLag = makeReg(countDown, i1Type, 0, "fifo_count_down");
    Value lagPlus1 =
        comb::AddOp::create(rewriter, loc, countLag, constant(countType, 1));
    Value lagMinus1 =
        comb::SubOp::create(rewriter, loc, countLag, constant(countType, 1));
    countNext.setValue(
        mux(countUpLag, lagPlus1, mux(countDownLag, lagMinus1, countLag)));
    setName(countNext, "fifo_count");

    // Whether the count this cycle is `value`, without the adder: a compare of
    // the register against `value` adjusted by the registered step. Values
    // outside [0, depth] are never reached; comparing against them would wrap.
    llvm::SmallDenseMap<int64_t, Value> lagEqs, countEqs;
    auto lagEq = [&](int64_t value) -> Value {
      if (value < 0 || value > static_cast<int64_t>(depth))
        return falseValue;
      Value &eq = lagEqs[value];
      if (!eq)
        eq = comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::eq,
                                  countLag, constant(countType, value));
      return eq;
    };
    auto countEq = [&](int64_t value) -> Value {
      Value &eq = countEqs[value];
      if (!eq) {
        Value afterUp = lagEq(value - 1);
        Value afterDown = lagEq(value + 1);
        Value unchanged = lagEq(value);
        eq = mux(countUpLag, afterUp, mux(countDownLag, afterDown, unchanged));
      }
      return eq;
    };

    Backedge fullNext = bb.get(i1Type);
    Value full = makeReg(fullNext, i1Type, 0, "fifo_full");
    fullNext.setValue(step(full, countEq(depth - 1), falseValue));

    Backedge emptyNext = bb.get(i1Type);
    Value empty = makeReg(emptyNext, i1Type, 1, "fifo_empty");
    emptyNext.setValue(step(empty, falseValue, countEq(1)));

    // ====== Read/write pointers ======
    // The wrap test looks at the current pointer, in parallel with the
    // increment. With a power-of-two memory the increment wraps on its own.
    auto advance = [&](Backedge ptrNext, Value ptr, Value enable,
                       StringRef name) {
      if (ptrWidth == 0) {
        ptrNext.setValue(ptr);
        return;
      }
      Value next =
          comb::AddOp::create(rewriter, loc, ptr, constant(ptrType, 1));
      if (!llvm::isPowerOf2_64(memDepth)) {
        Value wrap =
            comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::eq, ptr,
                                 constant(ptrType, memDepth - 1));
        next = mux(wrap, constant(ptrType, 0), next);
      }
      ptrNext.setValue(mux(enable, next, ptr));
      setName(ptrNext, name);
    };
    advance(wrAddrNext, wrAddr, wrEn, "fifo_wr_addr_next");
    advance(rdAddrNext, rdAddr, rdEn, "fifo_rd_addr_next");

    // ====== Result values ======
    llvm::SmallVector<Value> results;

    // Data
    results.push_back(readData);
    // Full
    results.push_back(full);
    // Empty
    results.push_back(empty);

    // Almost full is `count >= threshold`. A threshold of `depth` makes it
    // `full`, and one of 0 makes it constant.
    if (auto threshold = mem.getAlmostFullThreshold()) {
      uint64_t af = *threshold;
      if (af == 0) {
        results.push_back(trueValue);
      } else if (af == depth || mem.getAlmostFull().use_empty()) {
        results.push_back(full);
      } else {
        Backedge next = bb.get(i1Type);
        Value almostFull = makeReg(next, i1Type, 0, "fifo_almost_full");
        Value notAtThreshold =
            comb::createOrFoldNot(rewriter, loc, countEq(af));
        Value afterUp =
            comb::OrOp::create(rewriter, loc, almostFull, countEq(af - 1));
        Value afterDown =
            comb::AndOp::create(rewriter, loc, almostFull, notAtThreshold);
        next.setValue(step(almostFull, afterUp, afterDown));
        results.push_back(almostFull);
      }
    }

    // Almost empty is `count <= threshold`. A threshold of 0 makes it `empty`,
    // and one of `depth` makes it constant.
    if (auto threshold = mem.getAlmostEmptyThreshold()) {
      uint64_t ae = *threshold;
      if (ae == depth) {
        results.push_back(trueValue);
      } else if (ae == 0 || mem.getAlmostEmpty().use_empty()) {
        results.push_back(empty);
      } else {
        Backedge next = bb.get(i1Type);
        Value almostEmpty = makeReg(next, i1Type, 1, "fifo_almost_empty");
        Value notAtThreshold =
            comb::createOrFoldNot(rewriter, loc, countEq(ae));
        Value afterUp =
            comb::AndOp::create(rewriter, loc, almostEmpty, notAtThreshold);
        Value afterDown =
            comb::OrOp::create(rewriter, loc, almostEmpty, countEq(ae + 1));
        next.setValue(step(almostEmpty, afterUp, afterDown));
        results.push_back(almostEmpty);
      }
    }

    // ====== Protocol checks =====
    // Not checked in reset, before the enables need to mean anything.
    Value clkI1 = seq::FromClockOp::create(rewriter, loc, clk);
    Value notInReset = comb::createOrFoldNot(rewriter, loc, rst);
    Value notEmptyAndRden = comb::createOrFoldNot(
        rewriter, loc, comb::AndOp::create(rewriter, loc, rdEn, empty));
    verif::ClockedAssertOp::create(
        rewriter, loc, notEmptyAndRden, verif::ClockEdge::Pos, clkI1,
        notInReset, rewriter.getStringAttr("FIFO empty when read enabled"));
    Value notFullAndWren = comb::createOrFoldNot(
        rewriter, loc, comb::AndOp::create(rewriter, loc, wrEn, full));
    verif::ClockedAssertOp::create(
        rewriter, loc, notFullAndWren, verif::ClockEdge::Pos, clkI1, notInReset,
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
