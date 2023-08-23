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

#include "circt/Conversion/SeqToSV.h"
#include "../PassDetail.h"
#include "FirMemLowering.h"
#include "FirRegLowering.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "lower-seq-to-sv"

using namespace circt;
using namespace seq;
using hw::HWModuleOp;
using llvm::MapVector;

namespace {
#define GEN_PASS_DEF_LOWERSEQTOSV
#include "circt/Conversion/Passes.h.inc"

struct SeqToSVPass : public impl::LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
  using LowerSeqToSVBase<SeqToSVPass>::lowerToAlwaysFF;
  using LowerSeqToSVBase<SeqToSVPass>::disableRegRandomization;
  using LowerSeqToSVBase<SeqToSVPass>::emitSeparateAlwaysBlocks;
  using LowerSeqToSVBase<SeqToSVPass>::LowerSeqToSVBase;
  using LowerSeqToSVBase<SeqToSVPass>::numSubaccessRestored;
};
} // anonymous namespace

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

  // Helper to create an assignment based on the register type.
  void createAssign(ConversionPatternRewriter &rewriter, sv::RegOp svReg,
                    OpTy reg) const;

private:
  bool lowerToAlwaysFF;
};

/// Create the assign.
template <>
void CompRegLower<CompRegOp>::createAssign(ConversionPatternRewriter &rewriter,
                                           sv::RegOp svReg,
                                           CompRegOp reg) const {
  rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
}
/// Create the assign inside of an if block.
template <>
void CompRegLower<CompRegClockEnabledOp>::createAssign(
    ConversionPatternRewriter &rewriter, sv::RegOp svReg,
    CompRegClockEnabledOp reg) const {
  Location loc = reg.getLoc();
  rewriter.create<sv::IfOp>(loc, reg.getClockEnable(), [&]() {
    rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
  });
}

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
  auto circuit = getOperation();
  MLIRContext *context = &getContext();

  auto modules = llvm::to_vector(circuit.getOps<HWModuleOp>());

  FirMemLowering memLowering(circuit);

  // Identify memories and group them by module.
  auto uniqueMems = memLowering.collectMemories(modules);
  MapVector<HWModuleOp, SmallVector<FirMemLowering::MemoryConfig>> memsByModule;
  for (auto &[config, memOps] : uniqueMems) {
    // Create the `HWModuleGeneratedOp`s for each unique configuration.
    auto genOp = memLowering.createMemoryModule(config, memOps);

    // Group memories by their parent module for parallelism.
    for (auto memOp : memOps) {
      auto parent = memOp->getParentOfType<HWModuleOp>();
      memsByModule[parent].emplace_back(&config, genOp, memOp);
    }
  }

  // Lower memories and registers in modules in parallel.
  mlir::parallelForEach(&getContext(), modules, [&](HWModuleOp module) {
    FirRegLowering regLowering(module, disableRegRandomization,
                               emitSeparateAlwaysBlocks);
    regLowering.lower();
    numSubaccessRestored += regLowering.numSubaccessRestored;

    if (auto *it = memsByModule.find(module); it != memsByModule.end())
      memLowering.lowerMemoriesInModule(module, it->second);
  });

  ConversionTarget target(*context);
  target.addIllegalDialect<SeqDialect>();
  target.addLegalDialect<sv::SVDialect, comb::CombDialect, hw::HWDialect>();

  RewritePatternSet patterns(context);
  patterns.add<CompRegLower<CompRegOp>>(context, lowerToAlwaysFF);
  patterns.add<CompRegLower<CompRegClockEnabledOp>>(context, lowerToAlwaysFF);
  patterns.add<ClockGateLowering>(context);
  patterns.add<ClockMuxLowering>(context);
  if (failed(applyPartialConversion(circuit, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass>
circt::createLowerSeqToSVPass(const LowerSeqToSVOptions &options) {
  return std::make_unique<SeqToSVPass>(options);
}
