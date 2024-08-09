//===- MooreToCore.cpp - Moore To Core Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Moore to Core Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTMOORETOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace moore;

using comb::ICmpPredicate;

namespace {

/// Returns the passed value if the integer width is already correct.
/// Zero-extends if it is too narrow.
/// Truncates if the integer is too wide and the truncated part is zero, if it
/// is not zero it returns the max value integer of target-width.
static Value adjustIntegerWidth(OpBuilder &builder, Value value,
                                uint32_t targetWidth, Location loc) {
  uint32_t intWidth = value.getType().getIntOrFloatBitWidth();
  if (intWidth == targetWidth)
    return value;

  if (intWidth < targetWidth) {
    Value zeroExt = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerType(targetWidth - intWidth), 0);
    return builder.create<comb::ConcatOp>(loc, ValueRange{zeroExt, value});
  }

  Value hi = builder.create<comb::ExtractOp>(loc, value, targetWidth,
                                             intWidth - targetWidth);
  Value zero = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(intWidth - targetWidth), 0);
  Value isZero = builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq, hi,
                                              zero, false);
  Value lo = builder.create<comb::ExtractOp>(loc, value, 0, targetWidth);
  Value max = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(targetWidth), -1);
  return builder.create<comb::MuxOp>(loc, isZero, lo, max, false);
}

/// Get the ModulePortInfo from a SVModuleOp.
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> inputs, outputs;
  inputs.reserve(moduleTy.getNumInputs());
  outputs.reserve(moduleTy.getNumOutputs());

  for (auto port : moduleTy.getPorts())
    if (port.dir == hw::ModulePort::Direction::Output) {
      outputs.push_back(
          hw::PortInfo({{port.name, port.type, port.dir}, resultNum++, {}}));
    } else {
      // FIXME: Once we support net<...>, ref<...> type to represent type of
      // special port like inout or ref port which is not a input or output
      // port. It can change to generate corresponding types for direction of
      // port or do specified operation to it. Now inout and ref port is treated
      // as input port.
      inputs.push_back(
          hw::PortInfo({{port.name, port.type, port.dir}, inputNum++, {}}));
    }

  return hw::ModulePortInfo(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// Structural Conversion
//===----------------------------------------------------------------------===//

struct SVModuleOpConversion : public OpConversionPattern<SVModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SVModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputOp = op.getOutputOp();
    rewriter.setInsertionPoint(op);

    // Create the hw.module to replace moore.module
    auto hwModuleOp =
        rewriter.create<hw::HWModuleOp>(op.getLoc(), op.getSymNameAttr(),
                                        getModulePortInfo(*typeConverter, op));
    // Make hw.module have the same visibility as the moore.module.
    // The entry/top level module is public, otherwise is private.
    SymbolTable::setSymbolVisibility(hwModuleOp,
                                     SymbolTable::getSymbolVisibility(op));
    rewriter.eraseBlock(hwModuleOp.getBodyBlock());
    rewriter.inlineRegionBefore(op.getBodyRegion(), hwModuleOp.getBodyRegion(),
                                hwModuleOp.getBodyRegion().end());

    // Rewrite the hw.output op
    rewriter.setInsertionPointToEnd(hwModuleOp.getBodyBlock());
    rewriter.replaceOpWithNewOp<hw::OutputOp>(outputOp, outputOp.getOperands());

    // Erase the original op
    rewriter.eraseOp(op);
    return success();
  }
};

struct InstanceOpConversion : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = op.getInstanceNameAttr();
    auto moduleName = op.getModuleNameAttr();

    // Create the new hw instanceOp to replace the original one.
    rewriter.setInsertionPoint(op);
    auto instOp = rewriter.create<hw::InstanceOp>(
        op.getLoc(), op.getResultTypes(), instName, moduleName, op.getInputs(),
        op.getInputNamesAttr(), op.getOutputNamesAttr(),
        /*Parameter*/ rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr);

    // Replace uses chain and erase the original op.
    op.replaceAllUsesWith(instOp.getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ProcedureOpConversion : public OpConversionPattern<ProcedureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProcedureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto procOp = rewriter.create<llhd::ProcessOp>(loc);

    // TODO: properly handle the procedure kind attribute
    if (op.getKind() != ProcedureKind::Always)
      return rewriter.notifyMatchFailure(loc, "not yet supported");

    // Collect all event ops in the procedure.
    SmallVector<EventOp> events(op.getOps<EventOp>());

    auto *entry = rewriter.createBlock(&procOp.getBody());
    auto *wait = rewriter.createBlock(&procOp.getBody());
    auto *check = rewriter.createBlock(&procOp.getBody());

    // We need to add an empty entry block because it is not allowed in MLIR to
    // branch back to the entry block. Instead we put the logic in the second
    // block and branch to that.
    rewriter.setInsertionPointToStart(entry);
    rewriter.create<cf::BranchOp>(loc, wait);

    // The block in which we can sample the past and where the wait terminator
    // resides.
    rewriter.setInsertionPointToStart(wait);

    auto getSignal = [&](Value input) -> Value {
      // If the read op input is defined outside and before the procedure
      // operation, we can get the remapped value directly.
      Value signal = rewriter.getRemappedValue(input);

      // Otherwise, it hasn't been converted yet, so we take the old one and
      // insert a cast.
      if (!signal) {
        Type convertedType = typeConverter->convertType(input.getType());
        assert(convertedType &&
               "if the input has not been converted yet, it should have a "
               "moore type and a valid type conversion");
        signal =
            rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedType, input)
                ->getResult(0);
      }

      return signal;
    };

    // All signals to observe in the `llhd.wait` operation.
    SmallVector<Value> toObserve;
    DenseSet<Value> alreadyObserved;
    // If there are no event operations in the procedure, it's a combinational
    // one. Thus we need to collect all signals used.
    if (events.empty()) {
      op->walk([&](Operation *operation) {
        for (auto &operand : operation->getOpOperands()) {
          Value value = getSignal(operand.get());
          auto memOp = dyn_cast<MemoryEffectOpInterface>(operation);
          if (!memOp)
            return;

          // The process is only sensitive to values that are read.
          if (isa<RefType>(operand.get().getType()) &&
              memOp.getEffectOnValue<MemoryEffects::Read>(operand.get())
                  .has_value()) {
            if (!alreadyObserved.contains(value))
              toObserve.push_back(value);

            alreadyObserved.insert(value);
          }
        }
      });
    }

    // Forall edge triggered events, probe the old value
    SmallVector<Value> oldValues(events.size(), Value());
    for (auto [i, event] : llvm::enumerate(events)) {
      auto readOp = event.getInput().getDefiningOp<ReadOp>();
      if (!readOp)
        return failure();

      Value signal = getSignal(readOp.getInput());
      toObserve.push_back(signal);

      // Non-edge triggered events only need the value in the present
      if (event.getEdge() != Edge::None)
        oldValues[i] = rewriter.create<llhd::PrbOp>(loc, signal);
    }

    rewriter.create<llhd::WaitOp>(loc, toObserve, Value(), ValueRange{}, check);
    rewriter.setInsertionPointToStart(check);

    if (events.empty()) {
      rewriter.create<cf::BranchOp>(loc, &op.getBody().front());
    } else {
      SmallVector<Value> disjuncts;
      for (auto [i, signal, event] : llvm::enumerate(toObserve, events)) {
        if (event.getEdge() == Edge::None)
          disjuncts.push_back(rewriter.create<llhd::PrbOp>(loc, signal));

        if (event.getEdge() == Edge::PosEdge ||
            event.getEdge() == Edge::BothEdges) {
          Value currVal = rewriter.create<llhd::PrbOp>(loc, signal);
          Value trueVal = rewriter.create<hw::ConstantOp>(loc, APInt(1, 1));
          Value notOldVal =
              rewriter.create<comb::XorOp>(loc, oldValues[i], trueVal);
          Value posedge = rewriter.create<comb::AndOp>(loc, notOldVal, currVal);
          disjuncts.push_back(posedge);
        }
        if (event.getEdge() == Edge::NegEdge ||
            event.getEdge() == Edge::BothEdges) {
          Value currVal = rewriter.create<llhd::PrbOp>(loc, signal);
          Value trueVal = rewriter.create<hw::ConstantOp>(loc, APInt(1, 1));
          Value notCurrVal =
              rewriter.create<comb::XorOp>(loc, currVal, trueVal);
          Value posedge =
              rewriter.create<comb::AndOp>(loc, oldValues[i], notCurrVal);
          disjuncts.push_back(posedge);
        }
      }

      Value isValid = rewriter.create<comb::OrOp>(loc, disjuncts, false);
      rewriter.create<cf::CondBranchOp>(loc, isValid, &op.getBody().front(),
                                        wait);
    }

    for (auto event : events)
      rewriter.eraseOp(event);

    rewriter.inlineRegionBefore(op.getBody(), procOp.getBody(),
                                procOp.getBody().end());

    for (auto returnOp : procOp.getOps<ReturnOp>()) {
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<cf::BranchOp>(loc, wait);
      rewriter.eraseOp(returnOp);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Declaration Conversion
//===----------------------------------------------------------------------===//

struct VariableOpConversion : public OpConversionPattern<VariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value init = adaptor.getInitial();
    // TODO: Unsupport x/z, so the initial value is 0.
    if (!init && cast<RefType>(op.getResult().getType()).getDomain() ==
                     Domain::FourValued)
      return failure();

    if (!init) {
      Type elementType = cast<hw::InOutType>(resultType).getElementType();
      int64_t width = hw::getBitWidth(elementType);
      if (width == -1)
        return failure();

      Value constZero = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));
      init = rewriter.createOrFold<hw::BitcastOp>(loc, elementType, constZero);
    }

    rewriter.replaceOpWithNewOp<llhd::SigOp>(op, resultType, op.getNameAttr(),
                                             init);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // FIXME: Discard unknown bits and map them to 0 for now.
    auto value = op.getValue().toAPInt(false);
    auto type = rewriter.getIntegerType(value.getBitWidth());
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, type, rewriter.getIntegerAttr(type, value));
    return success();
  }
};

struct NamedConstantOpConv : public OpConversionPattern<NamedConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NamedConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = typeConverter->convertType(op.getResult().getType());
    SmallString<32> symStr;
    switch (op.getKind()) {
    case NamedConst::Parameter:
      symStr = "parameter";
      break;
    case NamedConst::LocalParameter:
      symStr = "localparameter";
      break;
    case NamedConst::SpecParameter:
      symStr = "specparameter";
      break;
    }
    auto symAttr =
        rewriter.getStringAttr(symStr + Twine("_") + adaptor.getName());
    rewriter.replaceOpWithNewOp<hw::WireOp>(op, resultType, adaptor.getValue(),
                                            op.getNameAttr(),
                                            hw::InnerSymAttr::get(symAttr));
    return success();
  }
};

struct ConcatOpConversion : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getValues());
    return success();
  }
};

struct ReplicateOpConversion : public OpConversionPattern<ReplicateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ReplicateOp>(op, resultType,
                                                   adaptor.getValue());
    return success();
  }
};

struct ExtractOpConversion : public OpConversionPattern<ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(
        op, resultType, adaptor.getInput(), adaptor.getLowBit());
    return success();
  }
};

struct DynExtractOpConversion : public OpConversionPattern<DynExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Type inputType = adaptor.getInput().getType();

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      Value amount = adjustIntegerWidth(rewriter, adaptor.getLowBit(),
                                        intType.getWidth(), op->getLoc());
      Value value = rewriter.create<comb::ShrUOp>(op->getLoc(),
                                                  adaptor.getInput(), amount);

      rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, value, 0);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getNumElements());
      Value idx = adjustIntegerWidth(rewriter, adaptor.getLowBit(), idxWidth,
                                     op->getLoc());
      rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, adaptor.getInput(), idx);
      return success();
    }

    return failure();
  }
};

struct StructExtractOpConversion : public OpConversionPattern<StructExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::StructExtractOp>(
        op, adaptor.getInput(), adaptor.getFieldNameAttr());
    return success();
  }
};

struct ReduceAndOpConversion : public OpConversionPattern<ReduceAndOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value max = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::eq,
                                              adaptor.getInput(), max);
    return success();
  }
};

struct ReduceOrOpConversion : public OpConversionPattern<ReduceOrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value zero = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                              adaptor.getInput(), zero);
    return success();
  }
};

struct ReduceXorOpConversion : public OpConversionPattern<ReduceXorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, adaptor.getInput());
    return success();
  }
};

struct BoolCastOpConversion : public OpConversionPattern<BoolCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BoolCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    if (isa_and_nonnull<IntegerType>(resultType)) {
      Value zero = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                adaptor.getInput(), zero);
      return success();
    }
    return failure();
  }
};

struct NotOpConversion : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value max = rewriter.create<hw::ConstantOp>(op.getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getInput(), max);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value zero = rewriter.create<hw::ConstantOp>(op.getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::SubOp>(op, zero, adaptor.getInput());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), false);
    return success();
  }
};

template <typename SourceOp, ICmpPredicate pred>
struct ICmpOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, resultType, pred, adapter.getLhs(), adapter.getRhs());
    return success();
  }
};

struct ConversionOpConversion : public OpConversionPattern<ConversionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConversionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    int64_t inputBw = hw::getBitWidth(adaptor.getInput().getType());
    int64_t resultBw = hw::getBitWidth(resultType);
    if (inputBw == -1 || resultBw == -1)
      return failure();

    Value input = rewriter.createOrFold<hw::BitcastOp>(
        loc, rewriter.getIntegerType(inputBw), adaptor.getInput());
    Value amount = adjustIntegerWidth(rewriter, input, resultBw, loc);

    Value result =
        rewriter.createOrFold<hw::BitcastOp>(loc, resultType, amount);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Statement Conversion
//===----------------------------------------------------------------------===//

struct HWOutputOpConversion : public OpConversionPattern<hw::OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, adaptor.getOperands());
    return success();
  }
};

struct HWInstanceOpConversion : public OpConversionPattern<hw::InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, convResTypes, op.getInstanceName(), op.getModuleName(),
        adaptor.getOperands(), op.getArgNames(),
        op.getResultNames(), /*Parameter*/
        rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct CondBranchOpConversion : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

struct BranchOpConversion : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), convResTypes, adaptor.getOperands());
    return success();
  }
};

struct UnrealizedConversionCastConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    // Drop the cast if the operand and result types agree after type
    // conversion.
    if (convResTypes == adaptor.getOperands().getTypes()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convResTypes, adaptor.getOperands());
    return success();
  }
};

struct ShlOpConversion : public OpConversionPattern<ShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, adaptor.getValue(),
                                             amount, false);
    return success();
  }
};

struct ShrOpConversion : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrUOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct AShrOpConversion : public OpConversionPattern<AShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrSOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct ReadOpConversion : public OpConversionPattern<ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<llhd::PrbOp>(op, adaptor.getInput());
    return success();
  }
};

template <typename OpTy, unsigned DeltaTime, unsigned EpsilonTime>
struct AssignOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: When we support delay control in Moore dialect, we need to update
    // this conversion.
    auto timeAttr = llhd::TimeAttr::get(
        op->getContext(), 0U, llvm::StringRef("ns"), DeltaTime, EpsilonTime);
    auto time = rewriter.create<llhd::ConstantTimeOp>(op->getLoc(), timeAttr);
    rewriter.replaceOpWithNewOp<llhd::DrvOp>(op, adaptor.getDst(),
                                             adaptor.getSrc(), time, Value{});
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static bool isMooreType(Type type) { return isa<UnpackedType>(type); }

static bool hasMooreType(TypeRange types) {
  return llvm::any_of(types, isMooreType);
}

static bool hasMooreType(ValueRange values) {
  return hasMooreType(values.getTypes());
}

template <typename Op>
static void addGenericLegality(ConversionTarget &target) {
  target.addDynamicallyLegalOp<Op>([](Op op) {
    return !hasMooreType(op->getOperands()) && !hasMooreType(op->getResults());
  });
}

static void populateLegality(ConversionTarget &target) {
  target.addIllegalDialect<MooreDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<llhd::LLHDDialect>();
  target.addLegalDialect<comb::CombDialect>();

  addGenericLegality<cf::CondBranchOp>(target);
  addGenericLegality<cf::BranchOp>(target);
  addGenericLegality<func::CallOp>(target);
  addGenericLegality<func::ReturnOp>(target);
  addGenericLegality<UnrealizedConversionCastOp>(target);

  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
      return hasMooreType(block.getArguments());
    });
    auto resultsConverted = !hasMooreType(op.getResultTypes());
    return argsConverted && resultsConverted;
  });

  target.addDynamicallyLegalOp<hw::HWModuleOp>([](hw::HWModuleOp op) {
    return !hasMooreType(op.getInputTypes()) &&
           !hasMooreType(op.getOutputTypes()) &&
           !hasMooreType(op.getBody().getArgumentTypes());
  });

  target.addDynamicallyLegalOp<hw::InstanceOp>([](hw::InstanceOp op) {
    return !hasMooreType(op.getInputs()) && !hasMooreType(op.getResults());
  });

  target.addDynamicallyLegalOp<hw::OutputOp>(
      [](hw::OutputOp op) { return !hasMooreType(op.getOutputs()); });
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](IntType type) {
    return IntegerType::get(type.getContext(), type.getWidth());
  });

  typeConverter.addConversion([&](ArrayType type) {
    return hw::ArrayType::get(typeConverter.convertType(type.getElementType()),
                              type.getSize());
  });

  typeConverter.addConversion([&](StructType type) {
    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto field : type.getMembers()) {
      hw::StructType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      info.name = field.name;
      fields.push_back(info);
    }
    return hw::StructType::get(type.getContext(), fields);
  });

  typeConverter.addConversion([&](RefType type) -> std::optional<Type> {
    auto innerType = typeConverter.convertType(type.getNestedType());
    if (innerType)
      return hw::InOutType::get(innerType);
    return {};
  });

  // Valid target types.
  typeConverter.addConversion([](IntegerType type) { return type; });

  typeConverter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  typeConverter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
  // Patterns of declaration operations.
    VariableOpConversion,

    // Patterns of miscellaneous operations.
    ConstantOpConv, ConcatOpConversion, ReplicateOpConversion,
    ExtractOpConversion, DynExtractOpConversion, ConversionOpConversion,
    ReadOpConversion, NamedConstantOpConv, StructExtractOpConversion,

    // Patterns of unary operations.
    ReduceAndOpConversion, ReduceOrOpConversion, ReduceXorOpConversion,
    BoolCastOpConversion, NotOpConversion, NegOpConversion,

    // Patterns of binary operations.
    BinaryOpConversion<AddOp, comb::AddOp>,
    BinaryOpConversion<SubOp, comb::SubOp>,
    BinaryOpConversion<MulOp, comb::MulOp>,
    BinaryOpConversion<DivUOp, comb::DivUOp>,
    BinaryOpConversion<DivSOp, comb::DivSOp>,
    BinaryOpConversion<ModUOp, comb::ModUOp>,
    BinaryOpConversion<ModSOp, comb::ModSOp>,
    BinaryOpConversion<AndOp, comb::AndOp>,
    BinaryOpConversion<OrOp, comb::OrOp>,
    BinaryOpConversion<XorOp, comb::XorOp>,

    // Patterns of relational operations.
    ICmpOpConversion<UltOp, ICmpPredicate::ult>,
    ICmpOpConversion<SltOp, ICmpPredicate::slt>,
    ICmpOpConversion<UleOp, ICmpPredicate::ule>,
    ICmpOpConversion<SleOp, ICmpPredicate::sle>,
    ICmpOpConversion<UgtOp, ICmpPredicate::ugt>,
    ICmpOpConversion<SgtOp, ICmpPredicate::sgt>,
    ICmpOpConversion<UgeOp, ICmpPredicate::uge>,
    ICmpOpConversion<SgeOp, ICmpPredicate::sge>,
    ICmpOpConversion<EqOp, ICmpPredicate::eq>,
    ICmpOpConversion<NeOp, ICmpPredicate::ne>,
    ICmpOpConversion<CaseEqOp, ICmpPredicate::ceq>,
    ICmpOpConversion<CaseNeOp, ICmpPredicate::cne>,
    ICmpOpConversion<WildcardEqOp, ICmpPredicate::weq>,
    ICmpOpConversion<WildcardNeOp, ICmpPredicate::wne>,
    
    // Patterns of structural operations.
    SVModuleOpConversion, InstanceOpConversion, ProcedureOpConversion,

    // Patterns of shifting operations.
    ShrOpConversion, ShlOpConversion, AShrOpConversion,

    // Patterns of assignment operations.
    AssignOpConversion<ContinuousAssignOp, 0, 1>,
    AssignOpConversion<BlockingAssignOp, 0, 1>,
    AssignOpConversion<NonBlockingAssignOp, 1, 0>,

    // Patterns of branch operations.
    CondBranchOpConversion, BranchOpConversion,

    // Patterns of other operations outside Moore dialect.
    HWOutputOpConversion, HWInstanceOpConversion, ReturnOpConversion,
    CallOpConversion, UnrealizedConversionCastConversion
  >(typeConverter, context);
  // clang-format on
  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);

  hw::populateHWModuleLikeTypeConversionPattern(
      hw::HWModuleOp::getOperationName(), patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Moore to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MooreToCorePass
    : public circt::impl::ConvertMooreToCoreBase<MooreToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertMooreToCorePass() {
  return std::make_unique<MooreToCorePass>();
}

/// This is the main entrypoint for the Moore to Core conversion pass.
void MooreToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  populateLegality(target);
  populateTypeConversion(typeConverter);
  populateOpConversion(patterns, typeConverter);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
