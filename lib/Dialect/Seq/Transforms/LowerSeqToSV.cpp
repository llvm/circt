//===- LowerSeqToSV.cpp - Seq to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq ops to SV.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;

namespace {
#define GEN_PASS_DEF_LOWERSEQTOSV
#include "circt/Dialect/Seq/SeqPasses.h.inc"

struct SeqToSVPass : public impl::LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
  using LowerSeqToSVBase::lowerToAlwaysFF;
};
} // anonymous namespace

/// Create the assign.
static void createAssign(ConversionPatternRewriter &rewriter, sv::RegOp svReg,
                         CompRegOp reg) {
  rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
}
/// Create the assign inside of an if block.
static void createAssign(ConversionPatternRewriter &rewriter, sv::RegOp svReg,
                         CompRegClockEnabledOp reg) {
  Location loc = reg.getLoc();
  rewriter.create<sv::IfOp>(loc, reg.getClockEnable(), [&]() {
    rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
  });
}

namespace {
/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
template <typename OpTy>
class CompRegLower : public OpConversionPattern<OpTy> {
public:
  CompRegLower(MLIRContext *context, bool lowerToAlwaysFF)
      : OpConversionPattern<OpTy>(context), lowerToAlwaysFF(lowerToAlwaysFF) {}

  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy reg, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg =
        rewriter.create<sv::RegOp>(loc, reg.getResult().getType(),
                                   reg.getNameAttr(), reg.getInnerSymAttr());
    svReg->setDialectAttrs(reg->getDialectAttrs());

    circt::sv::setSVAttributes(svReg, circt::sv::getSVAttributes(reg));

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);

    auto assignValue = [&] { createAssign(rewriter, svReg, reg); };
    auto assignReset = [&] {
      rewriter.create<sv::PAssignOp>(loc, svReg, adaptor.getResetValue());
    };

    if (adaptor.getReset() && adaptor.getResetValue()) {
      if (lowerToAlwaysFF) {
        rewriter.create<sv::AlwaysFFOp>(
            loc, sv::EventControl::AtPosEdge, adaptor.getClk(),
            ResetType::SyncReset, sv::EventControl::AtPosEdge,
            adaptor.getReset(), assignValue, assignReset);
      } else {
        rewriter.create<sv::AlwaysOp>(
            loc, sv::EventControl::AtPosEdge, adaptor.getClk(), [&] {
              rewriter.create<sv::IfOp>(loc, adaptor.getReset(), assignReset,
                                        assignValue);
            });
      }
    } else {
      if (lowerToAlwaysFF) {
        rewriter.create<sv::AlwaysFFOp>(loc, sv::EventControl::AtPosEdge,
                                        adaptor.getClk(), assignValue);
      } else {
        rewriter.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge,
                                      adaptor.getClk(), assignValue);
      }
    }

    rewriter.replaceOp(reg, regVal);
    return success();
  }

private:
  bool lowerToAlwaysFF;
};

// Lower seq.clock_gate to a fairly standard clock gate implementation.
//
class ClockGateLowering : public OpConversionPattern<ClockGateOp> {
public:
  using OpConversionPattern<ClockGateOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ClockGateOp>::OpAdaptor;
  LogicalResult
  matchAndRewrite(ClockGateOp clockGate, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = clockGate.getLoc();
    Value clk = adaptor.getInput();

    // enable in
    Value enable = adaptor.getEnable();
    if (auto te = adaptor.getTestEnable())
      enable = rewriter.create<comb::OrOp>(loc, enable, te);

    // Enable latch.
    Value enableLatch = rewriter.create<sv::RegOp>(
        loc, rewriter.getI1Type(), rewriter.getStringAttr("cg_en_latch"));

    // Latch the enable signal using an always @* block.
    rewriter.create<sv::AlwaysOp>(
        loc, llvm::SmallVector<sv::EventControl>{}, llvm::SmallVector<Value>{},
        [&]() {
          rewriter.create<sv::IfOp>(
              loc, comb::createOrFoldNot(loc, clk, rewriter), [&]() {
                rewriter.create<sv::PAssignOp>(loc, enableLatch, enable);
              });
        });

    // Create the gated clock signal.
    Value gclk = rewriter.create<comb::AndOp>(
        loc, clk, rewriter.create<sv::ReadInOutOp>(loc, enableLatch));
    clockGate.replaceAllUsesWith(gclk);
    rewriter.eraseOp(clockGate);
    return success();
  }
};

// Lower seq.clock_mux to a `comb.mux` op
//
class ClockMuxLowering : public OpConversionPattern<ClockMuxOp> {
public:
  using OpConversionPattern<ClockMuxOp>::OpConversionPattern;
  using OpConversionPattern<ClockMuxOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ClockMuxOp clockMux, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<comb::MuxOp>(clockMux, adaptor.getCond(),
                                             adaptor.getTrueClock(),
                                             adaptor.getFalseClock(), true);
    return success();
  }
};
} // namespace

void SeqToSVPass::runOnOperation() {
  ModuleOp top = getOperation();

  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);
  target.addIllegalDialect<SeqDialect>();
  target.addLegalDialect<sv::SVDialect, comb::CombDialect, hw::HWDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<CompRegLower<CompRegOp>>(&ctxt, lowerToAlwaysFF);
  patterns.add<CompRegLower<CompRegClockEnabledOp>>(&ctxt, lowerToAlwaysFF);
  patterns.add<ClockGateLowering>(&ctxt);
  patterns.add<ClockMuxLowering>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass>
circt::seq::createSeqLowerToSVPass(std::optional<bool> lowerToAlwaysFF) {
  auto pass = std::make_unique<SeqToSVPass>();
  if (lowerToAlwaysFF)
    pass->lowerToAlwaysFF = *lowerToAlwaysFF;
  return pass;
}
