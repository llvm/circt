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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "lower-sim-to-sv"

namespace circt {
#define GEN_PASS_DEF_LOWERSIMTOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace sim;

namespace {

struct SimConversionState {
  hw::HWModuleOp module;
  bool usedSynthesisMacro = false;
  SetVector<StringAttr> dpiCallees;
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
    auto str = sv::ConstantStrOp::create(rewriter, loc, op.getFormatString());
    auto reg = sv::RegOp::create(rewriter, loc, resultType,
                                 rewriter.getStringAttr("_pargs"));
    sv::InitialOp::create(rewriter, loc, [&] {
      auto call = sv::SystemFunctionOp::create(
          rewriter, loc, resultType, "test$plusargs", ArrayRef<Value>{str});
      sv::BPAssignOp::create(rewriter, loc, reg, call);
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

    auto wirev = sv::WireOp::create(rewriter, loc, type,
                                    rewriter.getStringAttr("_pargs_v"));
    auto wiref = sv::WireOp::create(rewriter, loc, i1ty,
                                    rewriter.getStringAttr("_pargs_f"));

    state.usedSynthesisMacro = true;
    sv::IfDefOp::create(
        rewriter, loc, "SYNTHESIS",
        [&]() {
          auto cstFalse = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
          auto cstZ = sv::ConstantZOp::create(rewriter, loc, type);
          auto assignZ = sv::AssignOp::create(rewriter, loc, wirev, cstZ);
          circt::sv::setSVAttributes(
              assignZ,
              sv::SVAttributeAttr::get(
                  rewriter.getContext(),
                  "This dummy assignment exists to avoid undriven lint "
                  "warnings (e.g., Verilator UNDRIVEN).",
                  /*emitAsComment=*/true));
          sv::AssignOp::create(rewriter, loc, wiref, cstFalse);
        },
        [&]() {
          auto i32ty = rewriter.getIntegerType(32);
          auto regf = sv::RegOp::create(rewriter, loc, i32ty,
                                        rewriter.getStringAttr("_found"));
          auto regv = sv::RegOp::create(rewriter, loc, type,
                                        rewriter.getStringAttr("_value"));
          sv::InitialOp::create(rewriter, loc, [&] {
            auto str =
                sv::ConstantStrOp::create(rewriter, loc, op.getFormatString());
            auto call = sv::SystemFunctionOp::create(
                rewriter, loc, i32ty, "value$plusargs",
                ArrayRef<Value>{str, regv});
            sv::BPAssignOp::create(rewriter, loc, regf, call);
          });
          Value readRegF = sv::ReadInOutOp::create(rewriter, loc, regf);
          Value readRegV = sv::ReadInOutOp::create(rewriter, loc, regv);
          auto cstTrue = hw::ConstantOp::create(rewriter, loc, i32ty, 1);
          // Squash any X coming from the regf to 0.
          auto cmp = comb::ICmpOp::create(
              rewriter, loc, comb::ICmpPredicate::ceq, readRegF, cstTrue);
          sv::AssignOp::create(rewriter, loc, wiref, cmp);
          sv::AssignOp::create(rewriter, loc, wirev, readRegV);
        });

    Value readf = sv::ReadInOutOp::create(rewriter, loc, wiref);
    Value readv = sv::ReadInOutOp::create(rewriter, loc, wirev);

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

    Value clockCast = seq::FromClockOp::create(rewriter, loc, adaptor.getClk());

    this->state.usedSynthesisMacro = true;
    sv::IfDefOp::create(
        rewriter, loc, "SYNTHESIS", [&] {},
        [&] {
          sv::AlwaysOp::create(
              rewriter, loc, sv::EventControl::AtPosEdge, clockCast, [&] {
                sv::IfOp::create(rewriter, loc, adaptor.getCond(),
                                 [&] { ToOp::create(rewriter, loc); });
              });
        });

    rewriter.eraseOp(op);

    return success();
  }
};

class DPICallLowering : public SimConversionPattern<DPICallOp> {
public:
  using SimConversionPattern<DPICallOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(DPICallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    // Record the callee.
    state.dpiCallees.insert(op.getCalleeAttr().getAttr());

    bool isClockedCall = !!op.getClock();
    bool hasEnable = !!op.getEnable();

    SmallVector<sv::RegOp> temporaries;
    SmallVector<Value> reads;
    for (auto [type, result] :
         llvm::zip(op.getResultTypes(), op.getResults())) {
      temporaries.push_back(sv::RegOp::create(rewriter, op.getLoc(), type));
      reads.push_back(
          sv::ReadInOutOp::create(rewriter, op.getLoc(), temporaries.back()));
    }

    auto emitCall = [&]() {
      auto call = sv::FuncCallProceduralOp::create(
          rewriter, op.getLoc(), op.getResultTypes(), op.getCalleeAttr(),
          adaptor.getInputs());
      for (auto [lhs, rhs] : llvm::zip(temporaries, call.getResults())) {
        if (isClockedCall)
          sv::PAssignOp::create(rewriter, op.getLoc(), lhs, rhs);
        else
          sv::BPAssignOp::create(rewriter, op.getLoc(), lhs, rhs);
      }
    };
    if (isClockedCall) {
      Value clockCast =
          seq::FromClockOp::create(rewriter, loc, adaptor.getClock());
      sv::AlwaysOp::create(
          rewriter, loc,
          ArrayRef<sv::EventControl>{sv::EventControl::AtPosEdge},
          ArrayRef<Value>{clockCast}, [&]() {
            if (!hasEnable)
              return emitCall();
            sv::IfOp::create(rewriter, op.getLoc(), adaptor.getEnable(),
                             emitCall);
          });
    } else {
      // Unclocked call is lowered into always_comb.
      // TODO: If there is a return value and no output argument, use an
      // unclocked call op.
      sv::AlwaysCombOp::create(rewriter, loc, [&]() {
        if (!hasEnable)
          return emitCall();
        auto assignXToResults = [&] {
          for (auto lhs : temporaries) {
            auto xValue = sv::ConstantXOp::create(
                rewriter, op.getLoc(), lhs.getType().getElementType());
            sv::BPAssignOp::create(rewriter, op.getLoc(), lhs, xValue);
          }
        };
        sv::IfOp::create(rewriter, op.getLoc(), adaptor.getEnable(), emitCall,
                         assignXToResults);
      });
    }

    rewriter.replaceOp(op, reads);
    return success();
  }
};

// A helper struct to lower DPI function/call.
struct LowerDPIFunc {
  llvm::DenseMap<StringAttr, StringAttr> symbolToFragment;
  circt::Namespace nameSpace;
  LowerDPIFunc(mlir::ModuleOp module) { nameSpace.add(module); }
  void lower(sim::DPIFuncOp func);
  void addFragments(hw::HWModuleOp module,
                    ArrayRef<StringAttr> dpiCallees) const;
};

void LowerDPIFunc::lower(sim::DPIFuncOp func) {
  ImplicitLocOpBuilder builder(func.getLoc(), func);
  ArrayAttr inputLocsAttr, outputLocsAttr;
  if (func.getArgumentLocs()) {
    SmallVector<Attribute> inputLocs, outputLocs;
    for (auto [port, loc] :
         llvm::zip(func.getModuleType().getPorts(),
                   func.getArgumentLocsAttr().getAsRange<LocationAttr>())) {
      (port.dir == hw::ModulePort::Output ? outputLocs : inputLocs)
          .push_back(loc);
    }
    inputLocsAttr = builder.getArrayAttr(inputLocs);
    outputLocsAttr = builder.getArrayAttr(outputLocs);
  }

  auto svFuncDecl =
      sv::FuncOp::create(builder, func.getSymNameAttr(), func.getModuleType(),
                         func.getPerArgumentAttrsAttr(), inputLocsAttr,
                         outputLocsAttr, func.getVerilogNameAttr());
  // DPI function is a declaration so it must be a private function.
  svFuncDecl.setPrivate();
  auto name = builder.getStringAttr(nameSpace.newName(
      func.getSymNameAttr().getValue(), "dpi_import_fragument"));

  // Add include guards to avoid duplicate declarations. See Issue 7458.
  auto macroDecl = sv::MacroDeclOp::create(
      builder, nameSpace.newName("__CIRCT_DPI_IMPORT",
                                 func.getSymNameAttr().getValue().upper()));
  emit::FragmentOp::create(builder, name, [&]() {
    sv::IfDefOp::create(
        builder, macroDecl.getSymNameAttr(), []() {},
        [&]() {
          sv::FuncDPIImportOp::create(builder, func.getSymNameAttr(),
                                      StringAttr());
          sv::MacroDefOp::create(builder, macroDecl.getSymNameAttr(), "");
        });
  });

  symbolToFragment.insert({func.getSymNameAttr(), name});
  func.erase();
}

void LowerDPIFunc::addFragments(hw::HWModuleOp module,
                                ArrayRef<StringAttr> dpiCallees) const {
  llvm::SetVector<Attribute> fragments;
  // Add existing emit fragments.
  if (auto exstingFragments =
          module->getAttrOfType<ArrayAttr>(emit::getFragmentsAttrName()))
    for (auto fragment : exstingFragments.getAsRange<FlatSymbolRefAttr>())
      fragments.insert(fragment);
  for (auto callee : dpiCallees) {
    auto attr = symbolToFragment.at(callee);
    fragments.insert(FlatSymbolRefAttr::get(attr));
  }
  if (!fragments.empty())
    module->setAttr(
        emit::getFragmentsAttrName(),
        ArrayAttr::get(module.getContext(), fragments.takeVector()));
}

namespace {
struct SimToSVPass : public circt::impl::LowerSimToSVBase<SimToSVPass> {
  void runOnOperation() override {
    auto circuit = getOperation();
    MLIRContext *context = &getContext();
    LowerDPIFunc lowerDPIFunc(circuit);

    // Lower DPI functions.
    for (auto func :
         llvm::make_early_inc_range(circuit.getOps<sim::DPIFuncOp>()))
      lowerDPIFunc.lower(func);

    std::atomic<bool> usedSynthesisMacro = false;
    auto lowerModule = [&](hw::HWModuleOp module) {
      SimConversionState state;
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
      patterns.add<DPICallLowering>(context, state);
      auto result = applyPartialConversion(module, target, std::move(patterns));

      if (failed(result))
        return result;

      // Set the emit fragment.
      lowerDPIFunc.addFragments(module, state.dpiCallees.takeVector());

      if (state.usedSynthesisMacro)
        usedSynthesisMacro = true;
      return result;
    };

    if (failed(mlir::failableParallelForEach(
            context, circuit.getOps<hw::HWModuleOp>(), lowerModule)))
      return signalPassFailure();

    if (usedSynthesisMacro) {
      Operation *op = circuit.lookupSymbol("SYNTHESIS");
      if (op) {
        if (!isa<sv::MacroDeclOp>(op)) {
          op->emitOpError("should be a macro declaration");
          return signalPassFailure();
        }
      } else {
        auto builder = ImplicitLocOpBuilder::atBlockBegin(
            UnknownLoc::get(context), circuit.getBody());
        sv::MacroDeclOp::create(builder, "SYNTHESIS");
      }
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> circt::createLowerSimToSVPass() {
  return std::make_unique<SimToSVPass>();
}
