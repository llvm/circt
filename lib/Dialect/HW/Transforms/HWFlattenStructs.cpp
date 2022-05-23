//===- HWFlattenStructs.cpp - Struct type flattening pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass breaks down `hw.struct` types into their scalar elements, and
// expands module and instance ports accordingly.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Expansion utilities
//===----------------------------------------------------------------------===//

// Expand names by appending element names to a structure's basename.
static void expandNames(ArrayRef<Attribute> names, TypeRange types,
                        SmallVector<Attribute> &expandedNames,
                        ConversionPatternRewriter &rewriter) {
  assert(names.size() == types.size());

  for (auto item : llvm::enumerate(names)) {
    Attribute name = item.value();
    Type type = types[item.index()];
    if (auto structTy = type.dyn_cast<StructType>()) {
      for (auto element : structTy.getElements()) {
        StringRef prefix = name.dyn_cast<StringAttr>().getValue();
        StringRef elementName = element.name.getValue();
        expandedNames.push_back(
            rewriter.getStringAttr(prefix + Twine('_') + elementName));
      }
      continue;
    }
    expandedNames.push_back(name);
  }
}

// Expand operand lists by looking through unrealized casts.
static void expandOperands(ValueRange oldOperands, ValueRange newOperands,
                           SmallVectorImpl<Value> &expandedOperands,
                           int index = -1) {
  assert(oldOperands.size() == newOperands.size());

  for (auto item : llvm::enumerate(oldOperands)) {
    Value oldOperand = item.value();
    Value newOperand = newOperands[item.index()];
    if (auto structTy = oldOperand.getType().dyn_cast<StructType>()) {
      if (auto cast = newOperand.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (index < 0)
          expandedOperands.append(cast.getInputs().begin(),
                                  cast.getInputs().end());
        else
          expandedOperands.push_back(cast.getInputs()[index]);
        continue;
      }
      // else: single-element struct, was lowered automatically
    }
    expandedOperands.push_back(newOperand);
  }
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
struct UpdateModulePortNames : public OpConversionPattern<HWModuleOp> {
  using OpConversionPattern<HWModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HWModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FlattenStructCreate : public OpConversionPattern<StructCreateOp> {
  using OpConversionPattern<StructCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FlattenStructExtract : public OpConversionPattern<StructExtractOp> {
  using OpConversionPattern<StructExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FlattenOutput : public OpConversionPattern<OutputOp> {
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FlattenInstance : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern<InstanceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CleanupUnrealizedCast
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // anonymous namespace

LogicalResult UpdateModulePortNames::matchAndRewrite(
    HWModuleOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<Attribute> newArgNames, newResultNames;
  expandNames(op.argNames().getValue(), op.getFunctionType().getInputs(),
              newArgNames, rewriter);
  expandNames(op.resultNames().getValue(), op.getFunctionType().getResults(),
              newResultNames, rewriter);

  // The module has already been copied once by the signature conversion, so
  // just modify it in place here.
  rewriter.updateRootInPlace(op, [&]() {
    op.argNamesAttr(rewriter.getArrayAttr(newArgNames));
    op.resultNamesAttr(rewriter.getArrayAttr(newResultNames));
  });

  return success();
}

LogicalResult FlattenStructCreate::matchAndRewrite(
    StructCreateOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Just insert the cast as a placeholder -- the users will look through it to
  // access the struct elements.
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, op.getType(), adaptor.getOperands());
  return success();
}

LogicalResult FlattenStructExtract::matchAndRewrite(
    StructExtractOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // The only new operand should be an UnrealizedCastOp.
  ValueRange operands = adaptor.getOperands();
  if (auto cast = dyn_cast_or_null<UnrealizedConversionCastOp>(
          operands.front().getDefiningOp())) {
    // Figure out which of the cast's operands we need to forward.
    auto structTy = op.input().getType().cast<StructType>();
    for (auto item : llvm::enumerate(structTy.getElements()))
      if (item.value().name == op.fieldAttr()) {
        rewriter.replaceOp(op, cast.getInputs()[item.index()]);
        return success();
      }
    // Verifier of StructExtractOp has to ensure that the field name is indeed
    // present.
    llvm_unreachable("Field name attribute of hw::StructExtractOp invalid");
    return failure();
  }

  // single-element structs seem to lower be lowered automatically.
  rewriter.replaceOp(op, operands.front());
  return success();
}

LogicalResult
FlattenOutput::matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  SmallVector<Value> expandedOperands;
  expandOperands(op.operands(), adaptor.operands(), expandedOperands);
  rewriter.updateRootInPlace(op, [&]() { op->setOperands(expandedOperands); });
  return success();
}

LogicalResult
FlattenInstance::matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  // Determine new port names.
  SmallVector<Attribute> newArgNames, newResultNames;
  expandNames(op.argNames().getValue(), op.getOperandTypes(), newArgNames,
              rewriter);
  expandNames(op.resultNames().getValue(), op.getResultTypes(), newResultNames,
              rewriter);

  // Look through incoming unrealized casts.
  SmallVector<Value> newInputs;
  expandOperands(op.inputs(), adaptor.inputs(), newInputs);

  // Convert result types.
  SmallVector<Type> newTypes;
  if (failed(typeConverter->convertTypes(op.getResultTypes(), newTypes)))
    return failure();

  // Make new instance, with the converted types and port names.
  auto newInstance = rewriter.create<InstanceOp>(
      op.getLoc(), newTypes, op.instanceNameAttr(), op.moduleNameAttr(),
      newInputs, rewriter.getArrayAttr(newArgNames),
      rewriter.getArrayAttr(newResultNames), op.parametersAttr(),
      op.instanceNameAttr());

  // Insert unrealized casts to wrap results that originated from a struct
  // expansion.
  unsigned resultIndex = 0;
  SmallVector<Value> newResults;
  for (auto type : op.getResultTypes()) {
    if (auto structTy = type.dyn_cast<StructType>()) {
      unsigned nElements = structTy.getElements().size();
      ResultRange castInputs =
          newInstance.getResults().slice(resultIndex, nElements);
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), structTy, castInputs);
      newResults.push_back(cast.getResult(0));
      resultIndex += nElements;
      continue;
    }
    newResults.push_back(newInstance.getResult(resultIndex));
    ++resultIndex;
  }

  // Finally replace the old instance with the mix of new instance's results
  // and the casts.
  rewriter.replaceOp(op, newResults);
  return success();
}

LogicalResult CleanupUnrealizedCast::matchAndRewrite(
    UnrealizedConversionCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Catch casts that were introduced by the automatic lowering of
  // single-element structs.
  if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
    rewriter.replaceOp(op, adaptor.getInputs());
    return success();
  }

  // All casts should be dead at the end of the conversion.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct HWFlattenStructsPass
    : public HWFlattenStructsBase<HWFlattenStructsPass> {
  void runOnOperation() override;
};
} // anonymous namespace

static bool isStructType(Type ty) { return ty.isa<StructType>(); }

void HWFlattenStructsPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto *ctx = &getContext();

  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<HWModuleOp>([](HWModuleOp op) {
    auto funTy = op.getFunctionType();
    return llvm::none_of(funTy.getInputs(), isStructType) &&
           llvm::none_of(funTy.getResults(), isStructType);
  });
  target.addDynamicallyLegalDialect<HWDialect>([](Operation *op) {
    return llvm::none_of(op->getOperandTypes(), isStructType) &&
           llvm::none_of(op->getResultTypes(), isStructType);
  });

  // TODO: support nested structs!
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Optional<Type> { return type; });
  typeConverter.addConversion(
      [](StructType type, SmallVectorImpl<Type> &results) {
        type.getInnerTypes(results);
        return success();
      });

  RewritePatternSet patterns(ctx);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<HWModuleOp>(
      patterns, typeConverter);

  // TODO: support the remaining hw operations!
  patterns.add<UpdateModulePortNames, FlattenStructCreate, FlattenStructExtract,
               FlattenOutput, CleanupUnrealizedCast>(ctx);
  patterns.add<FlattenInstance>(typeConverter, ctx);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> circt::hw::createHWFlattenStructsPass() {
  return std::make_unique<HWFlattenStructsPass>();
}
