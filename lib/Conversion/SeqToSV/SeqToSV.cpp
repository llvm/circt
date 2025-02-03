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
#include "FirMemLowering.h"
#include "FirRegLowering.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Naming.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Mutex.h"

#define DEBUG_TYPE "lower-seq-to-sv"

using namespace circt;
using namespace seq;
using hw::HWModuleOp;
using llvm::MapVector;

namespace circt {
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
} // namespace circt

namespace {
struct ModuleLoweringState {
  ModuleLoweringState(HWModuleOp module)
      : immutableValueLowering(module), module(module) {}

  struct ImmutableValueLowering {
    ImmutableValueLowering(hw::HWModuleOp module) : module(module) {}

    // Lower initial ops.
    LogicalResult lower();
    LogicalResult lower(seq::InitialOp initialOp);

    Value
    lookupImmutableValue(mlir::TypedValue<seq::ImmutableType> immut) const {
      return mapping.lookup(immut);
    }

    sv::InitialOp getSVInitial() const { return svInitialOp; }

  private:
    sv::InitialOp svInitialOp = {};
    // A mapping from a dummy immutable value to the actual initial value
    // defined in SV initial op.
    MapVector<mlir::TypedValue<seq::ImmutableType>, Value> mapping;

    hw::HWModuleOp module;
  } immutableValueLowering;

  struct FragmentInfo {
    bool needsRegFragment = false;
    bool needsMemFragment = false;
  } fragment;

  HWModuleOp module;
};

LogicalResult ModuleLoweringState::ImmutableValueLowering::lower() {
  auto result = mergeInitialOps(module.getBodyBlock());
  if (failed(result))
    return failure();

  auto initialOp = *result;
  if (!initialOp)
    return success();

  return lower(initialOp);
}

LogicalResult
ModuleLoweringState::ImmutableValueLowering::lower(seq::InitialOp initialOp) {
  OpBuilder builder = OpBuilder::atBlockBegin(module.getBodyBlock());
  if (!svInitialOp)
    svInitialOp = builder.create<sv::InitialOp>(initialOp->getLoc());
  // Initial ops are merged to single one and must not have operands.
  assert(initialOp.getNumOperands() == 0 &&
         "initial op should have no operands");

  auto loc = initialOp.getLoc();
  llvm::SmallVector<Value> results;

  auto yieldOp = cast<seq::YieldOp>(initialOp.getBodyBlock()->getTerminator());

  for (auto [result, operand] :
       llvm::zip(initialOp.getResults(), yieldOp->getOperands())) {
    auto placeholder =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                loc, ArrayRef<Type>{result.getType()}, ArrayRef<Value>{})
            ->getResult(0);
    result.replaceAllUsesWith(placeholder);
    mapping.insert(
        {cast<mlir::TypedValue<seq ::ImmutableType>>(placeholder), operand});
  }

  svInitialOp.getBodyBlock()->getOperations().splice(
      svInitialOp.end(), initialOp.getBodyBlock()->getOperations());

  assert(initialOp->use_empty());
  initialOp.erase();
  yieldOp->erase();
  return success();
}

/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
template <typename OpTy>
class CompRegLower : public OpConversionPattern<OpTy> {
public:
  CompRegLower(
      TypeConverter &typeConverter, MLIRContext *context, bool lowerToAlwaysFF,
      const MapVector<StringAttr, ModuleLoweringState> &moduleLoweringStates)
      : OpConversionPattern<OpTy>(typeConverter, context),
        lowerToAlwaysFF(lowerToAlwaysFF),
        moduleLoweringStates(moduleLoweringStates) {}

  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy reg, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto regTy =
        ConversionPattern::getTypeConverter()->convertType(reg.getType());
    auto svReg = rewriter.create<sv::RegOp>(loc, regTy, reg.getNameAttr(),
                                            reg.getInnerSymAttr());

    svReg->setDialectAttrs(reg->getDialectAttrs());

    circt::sv::setSVAttributes(svReg, circt::sv::getSVAttributes(reg));

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);

    auto assignValue = [&] {
      createAssign(rewriter, reg.getLoc(), svReg, reg);
    };
    auto assignReset = [&] {
      rewriter.create<sv::PAssignOp>(loc, svReg, adaptor.getResetValue());
    };

    // Registers written in an `always_ff` process may not have any assignments
    // outside of that process.
    // For some tools this also prohibits inititalization.
    bool mayLowerToAlwaysFF = lowerToAlwaysFF && !reg.getInitialValue();

    if (adaptor.getReset() && adaptor.getResetValue()) {
      if (mayLowerToAlwaysFF) {
        rewriter.create<sv::AlwaysFFOp>(
            loc, sv::EventControl::AtPosEdge, adaptor.getClk(),
            sv::ResetType::SyncReset, sv::EventControl::AtPosEdge,
            adaptor.getReset(), assignValue, assignReset);
      } else {
        rewriter.create<sv::AlwaysOp>(
            loc, sv::EventControl::AtPosEdge, adaptor.getClk(), [&] {
              rewriter.create<sv::IfOp>(loc, adaptor.getReset(), assignReset,
                                        assignValue);
            });
      }
    } else {
      if (mayLowerToAlwaysFF) {
        rewriter.create<sv::AlwaysFFOp>(loc, sv::EventControl::AtPosEdge,
                                        adaptor.getClk(), assignValue);
      } else {
        rewriter.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge,
                                      adaptor.getClk(), assignValue);
      }
    }

    // Lower initial values.
    if (auto init = reg.getInitialValue()) {
      auto module = reg->template getParentOfType<hw::HWModuleOp>();
      const auto &initial =
          moduleLoweringStates.find(module.getModuleNameAttr())
              ->second.immutableValueLowering;

      Value initialValue = initial.lookupImmutableValue(init);

      if (auto op = initialValue.getDefiningOp();
          op && op->hasTrait<mlir::OpTrait::ConstantLike>()) {
        auto clonedConstant = rewriter.clone(*op);
        rewriter.moveOpBefore(clonedConstant, svReg);
        svReg.getInitMutable().assign(clonedConstant->getResult(0));
      } else {
        OpBuilder::InsertionGuard guard(rewriter);
        auto in = initial.getSVInitial();
        rewriter.setInsertionPointToEnd(in.getBodyBlock());
        rewriter.create<sv::BPAssignOp>(reg->getLoc(), svReg, initialValue);
      }
    }

    rewriter.replaceOp(reg, regVal);
    return success();
  }

  // Helper to create an assignment based on the register type.
  void createAssign(ConversionPatternRewriter &rewriter, Location loc,
                    sv::RegOp svReg, OpAdaptor reg) const;

private:
  bool lowerToAlwaysFF;
  const MapVector<StringAttr, ModuleLoweringState> &moduleLoweringStates;
};

/// Create the assign.
template <>
void CompRegLower<CompRegOp>::createAssign(ConversionPatternRewriter &rewriter,
                                           Location loc, sv::RegOp svReg,
                                           OpAdaptor reg) const {
  rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput());
}
/// Create the assign inside of an if block.
template <>
void CompRegLower<CompRegClockEnabledOp>::createAssign(
    ConversionPatternRewriter &rewriter, Location loc, sv::RegOp svReg,
    OpAdaptor reg) const {
  rewriter.create<sv::IfOp>(loc, reg.getClockEnable(), [&]() {
    rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput());
  });
}

/// Lower FromImmutable to `sv.reg` and `sv.initial`.
class FromImmutableLowering : public OpConversionPattern<FromImmutableOp> {
public:
  FromImmutableLowering(
      TypeConverter &typeConverter, MLIRContext *context,
      const MapVector<StringAttr, ModuleLoweringState> &moduleLoweringStates)
      : OpConversionPattern<FromImmutableOp>(typeConverter, context),
        moduleLoweringStates(moduleLoweringStates) {}

  using OpAdaptor = typename OpConversionPattern<FromImmutableOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(FromImmutableOp fromImmutableOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = fromImmutableOp.getLoc();

    auto regTy = ConversionPattern::getTypeConverter()->convertType(
        fromImmutableOp.getType());
    auto svReg = rewriter.create<sv::RegOp>(loc, regTy);

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);

    // Lower initial values.
    auto module = fromImmutableOp->template getParentOfType<hw::HWModuleOp>();
    const auto &initial = moduleLoweringStates.find(module.getModuleNameAttr())
                              ->second.immutableValueLowering;

    Value initialValue =
        initial.lookupImmutableValue(fromImmutableOp.getInput());

    OpBuilder::InsertionGuard guard(rewriter);
    auto in = initial.getSVInitial();
    rewriter.setInsertionPointToEnd(in.getBodyBlock());
    rewriter.create<sv::BPAssignOp>(fromImmutableOp->getLoc(), svReg,
                                    initialValue);

    rewriter.replaceOp(fromImmutableOp, regVal);
    return success();
  }

private:
  const MapVector<StringAttr, ModuleLoweringState> &moduleLoweringStates;
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
    rewriter.replaceOpWithNewOp<comb::AndOp>(
        clockGate, clk, rewriter.create<sv::ReadInOutOp>(loc, enableLatch));
    return success();
  }
};

// Lower seq.clock_inv to a regular inverter.
//
class ClockInverterLowering : public OpConversionPattern<ClockInverterOp> {
public:
  using OpConversionPattern<ClockInverterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClockInverterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value clk = adaptor.getInput();

    StringAttr name = op->getAttrOfType<StringAttr>("sv.namehint");
    Value one = rewriter.create<hw::ConstantOp>(loc, APInt(1, 1));
    auto newOp = rewriter.replaceOpWithNewOp<comb::XorOp>(op, clk, one);
    if (name)
      rewriter.modifyOpInPlace(newOp,
                               [&] { newOp->setAttr("sv.namehint", name); });
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
                                             adaptor.getFalseClock());
    return success();
  }
};

/// Map `seq.clock` to `i1`.
struct SeqToSVTypeConverter : public TypeConverter {
  SeqToSVTypeConverter() {
    addConversion([&](Type type) { return type; });
    addConversion([&](seq::ImmutableType type) { return type.getInnerType(); });
    addConversion([&](seq::ClockType type) {
      return IntegerType::get(type.getContext(), 1);
    });
    addConversion([&](hw::StructType structTy) {
      bool changed = false;

      SmallVector<hw::StructType::FieldInfo> newFields;
      for (auto field : structTy.getElements()) {
        auto &newField = newFields.emplace_back();
        newField.name = field.name;
        newField.type = convertType(field.type);
        if (field.type != newField.type)
          changed = true;
      }

      if (!changed)
        return structTy;

      return hw::StructType::get(structTy.getContext(), newFields);
    });
    addConversion([&](hw::ArrayType arrayTy) {
      auto elementTy = arrayTy.getElementType();
      auto newElementTy = convertType(elementTy);
      if (elementTy != newElementTy)
        return hw::ArrayType::get(newElementTy, arrayTy.getNumElements());
      return arrayTy;
    });

    addTargetMaterialization([&](mlir::OpBuilder &builder,
                                 mlir::Type resultType, mlir::ValueRange inputs,
                                 mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();
      return builder
          .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0])
          ->getResult(0);
    });

    addSourceMaterialization([&](mlir::OpBuilder &builder,
                                 mlir::Type resultType, mlir::ValueRange inputs,
                                 mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();
      return builder
          .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0])
          ->getResult(0);
    });
  }
};

/// Eliminate no-op clock casts.
template <typename T>
class ClockCastLowering : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // If the cast had a better name than its input, propagate it.
    if (Operation *inputOp = adaptor.getInput().getDefiningOp())
      if (!isa<mlir::UnrealizedConversionCastOp>(inputOp))
        if (auto name = chooseName(op, inputOp))
          rewriter.modifyOpInPlace(
              inputOp, [&] { inputOp->setAttr("sv.namehint", name); });

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

// Lower seq.const_clock to `hw.constant`
//
class ClockConstLowering : public OpConversionPattern<ConstClockOp> {
public:
  using OpConversionPattern<ConstClockOp>::OpConversionPattern;
  using OpConversionPattern<ConstClockOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ConstClockOp clockConst, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        clockConst, APInt(1, clockConst.getValue() == ClockConst::High));
    return success();
  }
};

class AggregateConstantPattern
    : public OpConversionPattern<hw::AggregateConstantOp> {
public:
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;
  using OpConversionPattern<hw::AggregateConstantOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp aggregateConstant, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto newType = typeConverter->convertType(aggregateConstant.getType());
    auto newAttr = aggregateConstant.getFieldsAttr().replace(
        [](seq::ClockConstAttr clockConst) {
          return mlir::IntegerAttr::get(
              mlir::IntegerType::get(clockConst.getContext(), 1),
              APInt(1, clockConst.getValue() == ClockConst::High));
        });
    rewriter.replaceOpWithNewOp<hw::AggregateConstantOp>(
        aggregateConstant, newType, cast<ArrayAttr>(newAttr));
    return success();
  }
};

/// Lower `seq.clock_div` to a behavioural clock divider
///
class ClockDividerLowering : public OpConversionPattern<ClockDividerOp> {
public:
  using OpConversionPattern<ClockDividerOp>::OpConversionPattern;
  using OpConversionPattern<ClockDividerOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ClockDividerOp clockDiv, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = clockDiv.getLoc();

    Value one;
    if (clockDiv.getPow2()) {
      one = rewriter.create<hw::ConstantOp>(loc, APInt(1, 1));
    }

    Value output = clockDiv.getInput();

    SmallVector<Value> regs;
    for (unsigned i = 0; i < clockDiv.getPow2(); ++i) {
      Value reg = rewriter.create<sv::RegOp>(
          loc, rewriter.getI1Type(),
          rewriter.getStringAttr("clock_out_" + std::to_string(i)));
      regs.push_back(reg);

      rewriter.create<sv::AlwaysOp>(
          loc, sv::EventControl::AtPosEdge, output, [&] {
            Value outputVal = rewriter.create<sv::ReadInOutOp>(loc, reg);
            Value inverted = rewriter.create<comb::XorOp>(loc, outputVal, one);
            rewriter.create<sv::BPAssignOp>(loc, reg, inverted);
          });

      output = rewriter.create<sv::ReadInOutOp>(loc, reg);
    }

    if (!regs.empty()) {
      Value zero = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
      rewriter.create<sv::InitialOp>(loc, [&] {
        for (Value reg : regs) {
          rewriter.create<sv::BPAssignOp>(loc, reg, zero);
        }
      });
    }

    rewriter.replaceOp(clockDiv, output);
    return success();
  }
};

} // namespace

// NOLINTBEGIN(misc-no-recursion)
static bool isLegalType(Type ty) {
  if (hw::type_isa<ClockType>(ty))
    return false;

  if (auto arrayTy = hw::type_dyn_cast<hw::ArrayType>(ty))
    return isLegalType(arrayTy.getElementType());

  if (auto structTy = hw::type_dyn_cast<hw::StructType>(ty)) {
    for (auto field : structTy.getElements())
      if (!isLegalType(field.type))
        return false;
    return true;
  }

  return true;
}
// NOLINTEND(misc-no-recursion)

static bool isLegalOp(Operation *op) {
  if (auto module = dyn_cast<hw::HWModuleLike>(op)) {
    for (auto port : module.getHWModuleType().getPorts())
      if (!isLegalType(port.type))
        return false;
    return true;
  }

  if (auto hwAggregateConstantOp = dyn_cast<hw::AggregateConstantOp>(op)) {
    bool foundClockAttr = false;
    hwAggregateConstantOp.getFieldsAttr().walk(
        [&](seq::ClockConstAttr attr) { foundClockAttr = true; });
    if (foundClockAttr)
      return false;
  }

  bool allOperandsLowered = llvm::all_of(
      op->getOperands(), [](auto op) { return isLegalType(op.getType()); });
  bool allResultsLowered = llvm::all_of(op->getResults(), [](auto result) {
    return isLegalType(result.getType());
  });
  return allOperandsLowered && allResultsLowered;
}

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
  std::atomic<bool> needsRegRandomization = false;
  std::atomic<bool> needsMemRandomization = false;

  MapVector<StringAttr, ModuleLoweringState> moduleLoweringStates;
  for (auto module : circuit.getOps<HWModuleOp>())
    moduleLoweringStates.try_emplace(module.getModuleNameAttr(),
                                     ModuleLoweringState(module));

  auto result = mlir::failableParallelForEach(
      &getContext(), moduleLoweringStates, [&](auto &moduleAndState) {
        auto &state = moduleAndState.second;
        auto module = state.module;
        SeqToSVTypeConverter typeConverter;
        FirRegLowering regLowering(typeConverter, module,
                                   disableRegRandomization,
                                   emitSeparateAlwaysBlocks);
        regLowering.lower();
        if (regLowering.needsRegRandomization()) {
          if (!disableRegRandomization) {
            state.fragment.needsRegFragment = true;
          }
          needsRegRandomization = true;
        }
        numSubaccessRestored += regLowering.numSubaccessRestored;

        if (auto *it = memsByModule.find(module); it != memsByModule.end()) {
          memLowering.lowerMemoriesInModule(module, it->second);
          if (!disableMemRandomization) {
            state.fragment.needsMemFragment = true;
          }
          needsMemRandomization = true;
        }
        return state.immutableValueLowering.lower();
      });

  if (failed(result))
    return signalPassFailure();

  auto randomInitFragmentName =
      FlatSymbolRefAttr::get(context, "RANDOM_INIT_FRAGMENT");
  auto randomInitRegFragmentName =
      FlatSymbolRefAttr::get(context, "RANDOM_INIT_REG_FRAGMENT");
  auto randomInitMemFragmentName =
      FlatSymbolRefAttr::get(context, "RANDOM_INIT_MEM_FRAGMENT");

  for (auto &[_, state] : moduleLoweringStates) {
    const auto &info = state.fragment;
    if (!info.needsRegFragment && !info.needsMemFragment) {
      // If neither is emitted, just skip it.
      continue;
    }

    SmallVector<Attribute> fragmentAttrs;
    auto module = state.module;
    if (auto others =
            module->getAttrOfType<ArrayAttr>(emit::getFragmentsAttrName()))
      fragmentAttrs = llvm::to_vector(others);

    if (info.needsRegFragment)
      fragmentAttrs.push_back(randomInitRegFragmentName);
    if (info.needsMemFragment)
      fragmentAttrs.push_back(randomInitMemFragmentName);
    fragmentAttrs.push_back(randomInitFragmentName);

    module->setAttr(emit::getFragmentsAttrName(),
                    ArrayAttr::get(context, fragmentAttrs));
  }

  // Mark all ops which can have clock types as illegal.
  SeqToSVTypeConverter typeConverter;
  ConversionTarget target(*context);
  target.addIllegalDialect<SeqDialect>();
  target.markUnknownOpDynamicallyLegal(isLegalOp);

  RewritePatternSet patterns(context);
  patterns.add<CompRegLower<CompRegOp>>(typeConverter, context, lowerToAlwaysFF,
                                        moduleLoweringStates);
  patterns.add<CompRegLower<CompRegClockEnabledOp>>(
      typeConverter, context, lowerToAlwaysFF, moduleLoweringStates);
  patterns.add<FromImmutableLowering>(typeConverter, context,
                                      moduleLoweringStates);
  patterns.add<ClockCastLowering<seq::FromClockOp>>(typeConverter, context);
  patterns.add<ClockCastLowering<seq::ToClockOp>>(typeConverter, context);
  patterns.add<ClockGateLowering>(typeConverter, context);
  patterns.add<ClockInverterLowering>(typeConverter, context);
  patterns.add<ClockMuxLowering>(typeConverter, context);
  patterns.add<ClockDividerLowering>(typeConverter, context);
  patterns.add<ClockConstLowering>(typeConverter, context);
  patterns.add<TypeConversionPattern>(typeConverter, context);
  patterns.add<AggregateConstantPattern>(typeConverter, context);

  if (failed(applyPartialConversion(circuit, target, std::move(patterns))))
    signalPassFailure();

  auto loc = UnknownLoc::get(context);
  auto b = ImplicitLocOpBuilder::atBlockBegin(loc, circuit.getBody());
  if (needsRegRandomization || needsMemRandomization) {
    b.create<sv::MacroDeclOp>("ENABLE_INITIAL_REG_");
    b.create<sv::MacroDeclOp>("ENABLE_INITIAL_MEM_");
    if (needsRegRandomization) {
      b.create<sv::MacroDeclOp>("FIRRTL_BEFORE_INITIAL");
      b.create<sv::MacroDeclOp>("FIRRTL_AFTER_INITIAL");
    }
    if (needsMemRandomization)
      b.create<sv::MacroDeclOp>("RANDOMIZE_MEM_INIT");
    b.create<sv::MacroDeclOp>("RANDOMIZE_REG_INIT");
    b.create<sv::MacroDeclOp>("RANDOMIZE");
    b.create<sv::MacroDeclOp>("RANDOMIZE_DELAY");
    b.create<sv::MacroDeclOp>("RANDOM");
    b.create<sv::MacroDeclOp>("INIT_RANDOM");
    b.create<sv::MacroDeclOp>("INIT_RANDOM_PROLOG_");
  }

  bool hasRegRandomization = needsRegRandomization && !disableRegRandomization;
  bool hasMemRandomization = needsMemRandomization && !disableMemRandomization;
  if (!hasRegRandomization && !hasMemRandomization)
    return;

  // Build macros for FIRRTL-style register and memory initialization.
  // Insert them at the start of the module, after any other verbatims.
  for (Operation &op : *circuit.getBody()) {
    if (!isa<sv::VerbatimOp, sv::IfDefOp>(&op)) {
      b.setInsertionPoint(&op);
      break;
    }
  }

  // Create SYNTHESIS/VERILATOR macros if other passes have not done so already.
  {
    StringSet<> symbols;
    for (auto sym : circuit.getOps<sv::MacroDeclOp>())
      symbols.insert(sym.getName());
    if (!symbols.count("SYNTHESIS"))
      b.create<sv::MacroDeclOp>("SYNTHESIS");
    if (!symbols.count("VERILATOR"))
      b.create<sv::MacroDeclOp>("VERILATOR");
  }

  // TODO: We could have an operation for macros and uses of them, and
  // even turn them into symbols so we can DCE unused macro definitions.
  auto emitGuardedDefine = [&](StringRef guard, StringRef defName,
                               StringRef defineTrue = "",
                               StringRef defineFalse = StringRef()) {
    if (!defineFalse.data()) {
      assert(defineTrue.data() && "didn't define anything");
      b.create<sv::IfDefOp>(
          guard, [&]() { b.create<sv::MacroDefOp>(defName, defineTrue); });
    } else {
      b.create<sv::IfDefOp>(
          guard,
          [&]() {
            if (defineTrue.data())
              b.create<sv::MacroDefOp>(defName, defineTrue);
          },
          [&]() { b.create<sv::MacroDefOp>(defName, defineFalse); });
    }
  };

  // Helper function to emit #ifndef guard.
  auto emitGuard = [&](const char *guard, llvm::function_ref<void(void)> body) {
    b.create<sv::IfDefOp>(
        guard, []() {}, body);
  };

  b.create<emit::FragmentOp>(randomInitFragmentName.getAttr(), [&] {
    b.create<sv::VerbatimOp>(
        "// Standard header to adapt well known macros for "
        "register randomization.");

    b.create<sv::VerbatimOp>(
        "\n// RANDOM may be set to an expression that produces a 32-bit "
        "random unsigned value.");
    emitGuardedDefine("RANDOM", "RANDOM", StringRef(), "$random");

    b.create<sv::VerbatimOp>(
        "\n// Users can define INIT_RANDOM as general code that gets "
        "injected "
        "into the\n// initializer block for modules with registers.");
    emitGuardedDefine("INIT_RANDOM", "INIT_RANDOM", StringRef(), "");

    b.create<sv::VerbatimOp>(
        "\n// If using random initialization, you can also define "
        "RANDOMIZE_DELAY to\n// customize the delay used, otherwise 0.002 "
        "is used.");
    emitGuardedDefine("RANDOMIZE_DELAY", "RANDOMIZE_DELAY", StringRef(),
                      "0.002");

    b.create<sv::VerbatimOp>(
        "\n// Define INIT_RANDOM_PROLOG_ for use in our modules below.");
    emitGuard("INIT_RANDOM_PROLOG_", [&]() {
      b.create<sv::IfDefOp>(
          "RANDOMIZE",
          [&]() {
            emitGuardedDefine("VERILATOR", "INIT_RANDOM_PROLOG_",
                              "`INIT_RANDOM",
                              "`INIT_RANDOM #`RANDOMIZE_DELAY begin end");
          },
          [&]() { b.create<sv::MacroDefOp>("INIT_RANDOM_PROLOG_", ""); });
    });
  });

  if (hasMemRandomization) {
    b.create<emit::FragmentOp>(randomInitMemFragmentName.getAttr(), [&] {
      b.create<sv::VerbatimOp>("\n// Include rmemory initializers in init "
                               "blocks unless synthesis is set");
      emitGuard("RANDOMIZE", [&]() {
        emitGuardedDefine("RANDOMIZE_MEM_INIT", "RANDOMIZE");
      });
      emitGuard("SYNTHESIS", [&] {
        emitGuardedDefine("ENABLE_INITIAL_MEM_", "ENABLE_INITIAL_MEM_",
                          StringRef(), "");
      });
      b.create<sv::VerbatimOp>("");
    });
  }

  if (hasRegRandomization) {
    b.create<emit::FragmentOp>(randomInitRegFragmentName.getAttr(), [&] {
      b.create<sv::VerbatimOp>("\n// Include register initializers in init "
                               "blocks unless synthesis is set");
      emitGuard("RANDOMIZE", [&]() {
        emitGuardedDefine("RANDOMIZE_REG_INIT", "RANDOMIZE");
      });
      emitGuard("SYNTHESIS", [&] {
        emitGuardedDefine("ENABLE_INITIAL_REG_", "ENABLE_INITIAL_REG_",
                          StringRef(), "");
      });
      b.create<sv::VerbatimOp>("");
    });
  }
}

std::unique_ptr<Pass>
circt::createLowerSeqToSVPass(const LowerSeqToSVOptions &options) {
  return std::make_unique<SeqToSVPass>(options);
}
