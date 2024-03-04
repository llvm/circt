//===- LowerSimToSV.cpp - Sim to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translates Sim ops to SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SimToSV.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "lower-sim-to-sv"

using namespace circt;
using namespace sim;

namespace {

struct SimConversionState {
  std::atomic<bool> usedSynthesisMacro = false;
};

template <typename T>
struct SimConversionPattern : public OpConversionPattern<T> {
  explicit SimConversionPattern(MLIRContext *context, SimConversionState &state)
      : OpConversionPattern<T>(context), state(state) {}

  SimConversionState &state;
};

} // namespace

// Lower `sim.plusargs.test` to a standard SV implementation.
//
class PlusArgsTestLowering : public SimConversionPattern<PlusArgsTestOp> {
public:
  using SimConversionPattern<PlusArgsTestOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(PlusArgsTestOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = rewriter.getIntegerType(1);
    auto str = rewriter.create<sv::ConstantStrOp>(loc, op.getFormatString());
    auto reg = rewriter.create<sv::RegOp>(loc, resultType,
                                          rewriter.getStringAttr("_pargs"));
    rewriter.create<sv::InitialOp>(loc, [&] {
      auto call = rewriter.create<sv::SystemFunctionOp>(
          loc, resultType, "test$plusargs", ArrayRef<Value>{str});
      rewriter.create<sv::BPAssignOp>(loc, reg, call);
    });

    rewriter.replaceOpWithNewOp<sv::ReadInOutOp>(op, reg);
    return success();
  }
};

// Lower `sim.plusargs.value` to a standard SV implementation.
//
class PlusArgsValueLowering : public SimConversionPattern<PlusArgsValueOp> {
public:
  using SimConversionPattern<PlusArgsValueOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(PlusArgsValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto i1ty = rewriter.getIntegerType(1);
    auto type = op.getResult().getType();

    auto regv = rewriter.create<sv::RegOp>(loc, type,
                                           rewriter.getStringAttr("_pargs_v_"));
    auto regf = rewriter.create<sv::RegOp>(loc, i1ty,
                                           rewriter.getStringAttr("_pargs_f"));

    state.usedSynthesisMacro = true;
    rewriter.create<sv::IfDefOp>(
        loc, "SYNTHESIS",
        [&]() {
          auto cstFalse = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
          auto cstZ = rewriter.create<sv::ConstantZOp>(loc, type);
          auto assignZ = rewriter.create<sv::AssignOp>(loc, regv, cstZ);
          circt::sv::setSVAttributes(
              assignZ,
              sv::SVAttributeAttr::get(
                  rewriter.getContext(),
                  "This dummy assignment exists to avoid undriven lint "
                  "warnings (e.g., Verilator UNDRIVEN).",
                  /*emitAsComment=*/true));
          rewriter.create<sv::AssignOp>(loc, regf, cstFalse);
        },
        [&]() {
          rewriter.create<sv::InitialOp>(loc, [&] {
            auto zero32 = rewriter.create<hw::ConstantOp>(loc, APInt(32, 0));
            auto tmpResultType = rewriter.getIntegerType(32);
            auto str =
                rewriter.create<sv::ConstantStrOp>(loc, op.getFormatString());
            auto call = rewriter.create<sv::SystemFunctionOp>(
                loc, tmpResultType, "value$plusargs",
                ArrayRef<Value>{str, regv});
            auto test = rewriter.create<comb::ICmpOp>(
                loc, comb::ICmpPredicate::ne, call, zero32, true);
            rewriter.create<sv::BPAssignOp>(loc, regf, test);
          });
        });

    auto readf = rewriter.create<sv::ReadInOutOp>(loc, regf);
    auto readv = rewriter.create<sv::ReadInOutOp>(loc, regv);
    rewriter.replaceOp(op, {readf, readv});
    return success();
  }
};

template <typename FromOp, typename ToOp>
class SimulatorStopLowering : public SimConversionPattern<FromOp> {
public:
  using SimConversionPattern<FromOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(FromOp op, typename FromOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Value clockCast = rewriter.create<seq::FromClockOp>(loc, adaptor.getClk());

    this->state.usedSynthesisMacro = true;
    rewriter.create<sv::IfDefOp>(
        loc, "SYNTHESIS", [&] {},
        [&] {
          rewriter.create<sv::AlwaysOp>(
              loc, sv::EventControl::AtPosEdge, clockCast, [&] {
                rewriter.create<sv::IfOp>(loc, adaptor.getCond(),
                                          [&] { rewriter.create<ToOp>(loc); });
              });
        });

    rewriter.eraseOp(op);

    return success();
  }
};

namespace {
struct SimToSVPass : public LowerSimToSVBase<SimToSVPass> {
  void runOnOperation() override {
    auto circuit = getOperation();
    MLIRContext *context = &getContext();

    SimConversionState state;
    auto lowerModule = [&](hw::HWModuleOp module) {
      ConversionTarget target(*context);
      target.addIllegalDialect<SimDialect>();
      target.addLegalDialect<sv::SVDialect>();
      target.addLegalDialect<hw::HWDialect>();
      target.addLegalDialect<seq::SeqDialect>();
      target.addLegalDialect<comb::CombDialect>();

      RewritePatternSet patterns(context);
      patterns.add<PlusArgsTestLowering>(context, state);
      patterns.add<PlusArgsValueLowering>(context, state);
      patterns.add<SimulatorStopLowering<sim::FinishOp, sv::FinishOp>>(context,
                                                                       state);
      patterns.add<SimulatorStopLowering<sim::FatalOp, sv::FatalOp>>(context,
                                                                     state);
      return applyPartialConversion(module, target, std::move(patterns));
    };

    if (failed(mlir::failableParallelForEach(
            context, circuit.getOps<hw::HWModuleOp>(), lowerModule)))
      return signalPassFailure();

    if (state.usedSynthesisMacro) {
      Operation *op = circuit.lookupSymbol("SYNTHESIS");
      if (op) {
        if (!isa<sv::MacroDeclOp>(op)) {
          op->emitOpError("should be a macro declaration");
          return signalPassFailure();
        }
      } else {
        auto builder = ImplicitLocOpBuilder::atBlockBegin(
            UnknownLoc::get(context), circuit.getBody());
        builder.create<sv::MacroDeclOp>("SYNTHESIS");
      }
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> circt::createLowerSimToSVPass() {
  return std::make_unique<SimToSVPass>();
}
