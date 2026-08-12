//===- FlattenStructs.cpp - Flatten all structs to integers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_FLATTENSTRUCTS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

namespace {

struct FlattenStructsPass
    : public arc::impl::FlattenStructsBase<FlattenStructsPass> {
  using FlattenStructsBase::FlattenStructsBase;
  void runOnOperation() override;
};

// Struct flattening layout: elements are ordered from MSB to LSB matching
// declaration order. The first element (`elements[0]`) occupies the most
// significant bits, while the last element occupies the least significant
// bits. This matches `comb::ConcatOp` operand ordering where the first operand
// forms the MSB slice.

static Value ensureIntegerType(OpBuilder &builder, Location loc, Value val,
                               int64_t expectedWidth) {
  if (auto intTy = dyn_cast<IntegerType>(val.getType())) {
    if (intTy.getWidth() == expectedWidth)
      return val;
  }
  return hw::BitcastOp::create(builder, loc,
                               builder.getIntegerType(expectedWidth), val)
      .getResult();
}

static Value castFromInteger(OpBuilder &builder, Location loc, Value intVal,
                             Type targetType) {
  if (intVal.getType() == targetType)
    return intVal;
  return hw::BitcastOp::create(builder, loc, targetType, intVal).getResult();
}

static Attribute convertAggregateAttr(Attribute attr, Type originalType,
                                      Type convertedType,
                                      const TypeConverter &converter) {
  if (originalType == convertedType)
    return attr;
  if (auto structType = dyn_cast<hw::StructType>(originalType)) {
    if (auto intType = dyn_cast<IntegerType>(convertedType)) {
      if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
        APInt intVal;
        if (succeeded(hw::aggregateAttrToAPInt(structType, arrayAttr, intVal)))
          return IntegerAttr::get(intType, intVal);
      }
    }
  }
  if (auto arrayType = dyn_cast<hw::ArrayType>(originalType)) {
    if (auto convArrayType = dyn_cast<hw::ArrayType>(convertedType)) {
      if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
        SmallVector<Attribute> elements;
        for (Attribute elemAttr : arrayAttr) {
          elements.push_back(
              convertAggregateAttr(elemAttr, arrayType.getElementType(),
                                   convArrayType.getElementType(), converter));
        }
        return ArrayAttr::get(attr.getContext(), elements);
      }
    }
  }
  return attr;
}

// Converts aggregate struct constants to integer constants representing their
// combined bit patterns, or converts inner struct attributes of aggregate
// constants.
struct HWAggregateConstantOpConversion
    : public OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type origType = op.getType();
    Type resultType = getTypeConverter()->convertType(origType);
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert constant result type");

    if (auto structType = dyn_cast<hw::StructType>(origType)) {
      if (isa<IntegerType>(resultType)) {
        APInt intVal;
        if (failed(
                hw::aggregateAttrToAPInt(structType, op.getFields(), intVal)))
          return rewriter.notifyMatchFailure(
              op, "failed to evaluate aggregate attribute to integer");
        rewriter.replaceOpWithNewOp<hw::ConstantOp>(
            op, resultType, rewriter.getIntegerAttr(resultType, intVal));
        return success();
      }
    }

    Attribute newFields = convertAggregateAttr(op.getFields(), origType,
                                               resultType, *getTypeConverter());
    if (newFields == op.getFields() && origType == resultType)
      return rewriter.notifyMatchFailure(op, "no fields changed");

    rewriter.replaceOpWithNewOp<hw::AggregateConstantOp>(
        op, resultType, cast<ArrayAttr>(newFields));
    return success();
  }
};

// Converts `hw.bitcast` between types whose flattened forms change.
struct HWBitcastOpConversion : public OpConversionPattern<hw::BitcastOp> {
  using OpConversionPattern<hw::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type inType = adaptor.getInput().getType();
    Type outType = getTypeConverter()->convertType(op.getType());
    if (!outType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert bitcast result type");

    if (inType == outType) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, outType, adaptor.getInput());
    return success();
  }
};

// Fallback pattern to update generic operations when operand, result, or region
// block argument types change due to struct flattening.
struct GenericOpConversion : public ConversionPattern {
  GenericOpConversion(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, mlir::Pattern::MatchAnyOpTypeTag{},
                          /*benefit=*/0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter &converter = *getTypeConverter();

    SmallVector<Value> flatOperands;
    for (auto range : operands)
      flatOperands.insert(flatOperands.end(), range.begin(), range.end());

    SmallVector<Type> resultTypes;
    if (failed(converter.convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert operation results");

    bool hasIllegalRegionArg = false;
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        if (!converter.isLegal(block.getArgumentTypes())) {
          hasIllegalRegionArg = true;
          break;
        }
      }
      if (hasIllegalRegionArg)
        break;
    }

    if (llvm::equal(flatOperands, op->getOperands()) &&
        resultTypes == op->getResultTypes() && !hasIllegalRegionArg)
      return rewriter.notifyMatchFailure(op,
                                         "operation types are already legal");

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(flatOperands);
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();

    Operation *newOp = rewriter.create(state);
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
      rewriter.inlineRegionBefore(op->getRegion(i), newOp->getRegion(i),
                                  newOp->getRegion(i).end());
      if (failed(rewriter.convertRegionTypes(&newOp->getRegion(i), converter)))
        return rewriter.notifyMatchFailure(
            op, "failed to convert region argument types");
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void FlattenStructsPass::runOnOperation() {
  TypeConverter converter;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());

  converter.addConversion([](Type type) -> Type { return type; });

  converter.addConversion([](hw::StructType type) -> std::optional<Type> {
    int64_t width = hw::getBitWidth(type);
    if (width <= 0)
      return std::nullopt;
    return IntegerType::get(type.getContext(), width);
  });

  converter.addConversion(
      [&converter](hw::ArrayType type) -> std::optional<Type> {
        Type elemType = converter.convertType(type.getElementType());
        if (!elemType)
          return std::nullopt;
        return hw::ArrayType::get(elemType, type.getNumElements());
      });

  converter.addConversion(
      [&converter](hw::UnionType type) -> std::optional<Type> {
        SmallVector<hw::UnionType::FieldInfo> fields;
        for (const auto &field : type.getElements()) {
          Type fieldType = converter.convertType(field.type);
          if (!fieldType)
            return std::nullopt;
          fields.push_back({field.name, fieldType, field.offset});
        }
        return hw::UnionType::get(type.getContext(), fields);
      });

  converter.addConversion(
      [&converter](hw::TypeAliasType type) -> std::optional<Type> {
        return converter.convertType(type.getCanonicalType());
      });

  converter.addConversion(
      [&converter](arc::StateType type) -> std::optional<Type> {
        Type inner = converter.convertType(type.getType());
        if (!inner)
          return std::nullopt;
        return arc::StateType::get(inner);
      });

  converter.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1)
      return nullptr;
    if (isa<IntegerType>(resultType) &&
        isa<hw::StructType>(inputs[0].getType()))
      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          .getResult();
    return nullptr;
  });

  converter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1)
      return nullptr;
    if (isa<hw::StructType>(resultType) &&
        isa<IntegerType>(inputs[0].getType()))
      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          .getResult();
    return nullptr;
  });

  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return converter.isLegal(op); });

  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp func) {
        FunctionType fty = func.getFunctionType();
        return converter.isLegal(fty.getInputs()) &&
               converter.isLegal(fty.getResults());
      });

  target.addDynamicallyLegalOp<hw::HWModuleOp>([&](hw::HWModuleOp op) {
    for (hw::ModulePort port : op.getHWModuleType().getPorts())
      if (!converter.isLegal(port.type))
        return false;
    return true;
  });

  target.addDynamicallyLegalOp<hw::HWModuleExternOp>(
      [&](hw::HWModuleExternOp op) {
        for (hw::ModulePort port : op.getHWModuleType().getPorts())
          if (!converter.isLegal(port.type))
            return false;
        return true;
      });

  hw::populateHWModuleLikeTypeConversionPattern("hw.module", patterns,
                                                converter);
  hw::populateHWModuleLikeTypeConversionPattern("hw.module.extern", patterns,
                                                converter);
  populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, converter);

  hw::populateHWStructToCombConversionPatterns(patterns, converter);

  patterns.add<HWAggregateConstantOpConversion, HWBitcastOpConversion,
               GenericOpConversion>(converter, &getContext());

  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns),
                                 config)))
    return signalPassFailure();
}
