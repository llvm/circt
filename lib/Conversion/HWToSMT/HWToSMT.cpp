//===- HWToSMT.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTHWTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a hw::ConstantOp operation to smt::BVConstantOp
struct HWConstantOpConversion : OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getValue().getBitWidth() < 1)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "0-bit constants not supported");
    rewriter.replaceOpWithNewOp<mlir::smt::BVConstantOp>(op,
                                                         adaptor.getValue());
    return success();
  }
};

/// Lower a hw::HWModuleOp operation to func::FuncOp.
struct HWModuleOpConversion : OpConversionPattern<HWModuleOp> {
  using OpConversionPattern<HWModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HWModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcTy = op.getModuleType().getFuncType();
    SmallVector<Type> inputTypes, resultTypes;
    if (failed(typeConverter->convertTypes(funcTy.getInputs(), inputTypes)))
      return failure();
    if (failed(typeConverter->convertTypes(funcTy.getResults(), resultTypes)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter)))
      return failure();
    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), adaptor.getSymNameAttr(),
        rewriter.getFunctionType(inputTypes, resultTypes));
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower a hw::OutputOp operation to func::ReturnOp.
struct OutputOpConversion : OpConversionPattern<OutputOp> {
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOutputs());
    return success();
  }
};

/// Lower a hw::InstanceOp operation to func::CallOp.
struct InstanceOpConversion : OpConversionPattern<InstanceOp> {
  using OpConversionPattern<InstanceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, adaptor.getModuleNameAttr(), resultTypes, adaptor.getInputs());
    return success();
  }
};

/// Lower a hw::ArrayCreateOp operation to smt::DeclareFun and an
/// smt::ArrayStoreOp for each operand.
struct ArrayCreateOpConversion : OpConversionPattern<ArrayCreateOp> {
  using OpConversionPattern<ArrayCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type arrTy = typeConverter->convertType(op.getType());
    if (!arrTy)
      return rewriter.notifyMatchFailure(op.getLoc(), "unsupported array type");

    unsigned width = adaptor.getInputs().size();

    Value arr = rewriter.create<mlir::smt::DeclareFunOp>(loc, arrTy);
    for (auto [i, el] : llvm::enumerate(adaptor.getInputs())) {
      Value idx = rewriter.create<mlir::smt::BVConstantOp>(
          loc, width - i - 1, llvm::Log2_64_Ceil(width));
      arr = rewriter.create<mlir::smt::ArrayStoreOp>(loc, arr, idx, el);
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};

/// Lower a hw::ArrayGetOp operation to smt::ArraySelectOp
struct ArrayGetOpConversion : OpConversionPattern<ArrayGetOp> {
  using OpConversionPattern<ArrayGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numElements =
        cast<hw::ArrayType>(op.getInput().getType()).getNumElements();

    Type type = typeConverter->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unsupported array element type");

    Value oobVal = rewriter.create<mlir::smt::DeclareFunOp>(loc, type);
    Value numElementsVal = rewriter.create<mlir::smt::BVConstantOp>(
        loc, numElements - 1, llvm::Log2_64_Ceil(numElements));
    Value inBounds =
        rewriter.create<mlir::smt::BVCmpOp>(loc, mlir::smt::BVCmpPredicate::ule,
                                            adaptor.getIndex(), numElementsVal);
    Value indexed = rewriter.create<mlir::smt::ArraySelectOp>(
        loc, adaptor.getInput(), adaptor.getIndex());
    rewriter.replaceOpWithNewOp<mlir::smt::IteOp>(op, inBounds, indexed,
                                                  oobVal);
    return success();
  }
};

/// Lower a hw::ArrayInjectOp operation to smt::ArrayStoreOp.
struct ArrayInjectOpConversion : OpConversionPattern<ArrayInjectOp> {
  using OpConversionPattern<ArrayInjectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numElements =
        cast<hw::ArrayType>(op.getInput().getType()).getNumElements();

    Type arrType = typeConverter->convertType(op.getType());
    if (!arrType)
      return rewriter.notifyMatchFailure(op.getLoc(), "unsupported array type");

    // Check if the index is within bounds
    Value numElementsVal = rewriter.create<mlir::smt::BVConstantOp>(
        loc, numElements - 1, llvm::Log2_64_Ceil(numElements));
    Value inBounds =
        rewriter.create<mlir::smt::BVCmpOp>(loc, mlir::smt::BVCmpPredicate::ule,
                                            adaptor.getIndex(), numElementsVal);

    // Store the element at the given index
    Value stored = rewriter.create<mlir::smt::ArrayStoreOp>(
        loc, adaptor.getInput(), adaptor.getIndex(), adaptor.getElement());

    // Return the original array if out of bounds, otherwise return the new
    // array
    rewriter.replaceOpWithNewOp<mlir::smt::IteOp>(op, inBounds, stored,
                                                  adaptor.getInput());
    return success();
  }
};

/// Remove redundant (seq::FromClock and seq::ToClock) ops.
template <typename OpTy>
struct ReplaceWithInput : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert HW to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertHWToSMTPass
    : public impl::ConvertHWToSMTBase<ConvertHWToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateHWToSMTTypeConverter(TypeConverter &converter) {
  // The semantics of the builtin integer at the CIRCT core level is currently
  // not very well defined. It is used for two-valued, four-valued, and possible
  // other multi-valued logic. Here, we interpret it as two-valued for now.
  // From a formal perspective, CIRCT would ideally define its own types for
  // two-valued, four-valued, nine-valued (etc.) logic each. In MLIR upstream
  // the integer type also carries poison information (which we don't have in
  // CIRCT?).
  converter.addConversion([](IntegerType type) -> std::optional<Type> {
    if (type.getWidth() <= 0)
      return std::nullopt;
    return mlir::smt::BitVectorType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([](seq::ClockType type) -> std::optional<Type> {
    return mlir::smt::BitVectorType::get(type.getContext(), 1);
  });
  converter.addConversion([&](ArrayType type) -> std::optional<Type> {
    auto rangeType = converter.convertType(type.getElementType());
    if (!rangeType)
      return {};
    auto domainType = mlir::smt::BitVectorType::get(
        type.getContext(), llvm::Log2_64_Ceil(type.getNumElements()));
    return mlir::smt::ArrayType::get(type.getContext(), domainType, rangeType);
  });

  // Default target materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        ->getResult(0);
  });

  // Convert a 'smt.bool'-typed value to a 'smt.bv<N>'-typed value
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, mlir::smt::BitVectorType resultType,
          ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1)
          return Value();

        if (!isa<mlir::smt::BoolType>(inputs[0].getType()))
          return Value();

        unsigned width = resultType.getWidth();
        Value constZero =
            builder.create<mlir::smt::BVConstantOp>(loc, 0, width);
        Value constOne = builder.create<mlir::smt::BVConstantOp>(loc, 1, width);
        return builder.create<mlir::smt::IteOp>(loc, inputs[0], constOne,
                                                constZero);
      });

  // Convert an unrealized conversion cast from 'smt.bool' to i1
  // into a direct conversion from 'smt.bool' to 'smt.bv<1>'.
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, mlir::smt::BitVectorType resultType,
          ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1 || resultType.getWidth() != 1)
          return Value();

        auto intType = dyn_cast<IntegerType>(inputs[0].getType());
        if (!intType || intType.getWidth() != 1)
          return Value();

        auto castOp =
            inputs[0].getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          return Value();

        if (!isa<mlir::smt::BoolType>(castOp.getInputs()[0].getType()))
          return Value();

        Value constZero = builder.create<mlir::smt::BVConstantOp>(loc, 0, 1);
        Value constOne = builder.create<mlir::smt::BVConstantOp>(loc, 1, 1);
        return builder.create<mlir::smt::IteOp>(loc, castOp.getInputs()[0],
                                                constOne, constZero);
      });

  // Convert a 'smt.bv<1>'-typed value to a 'smt.bool'-typed value
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, mlir::smt::BoolType resultType, ValueRange inputs,
          Location loc) -> Value {
        if (inputs.size() != 1)
          return Value();

        auto bvType = dyn_cast<mlir::smt::BitVectorType>(inputs[0].getType());
        if (!bvType || bvType.getWidth() != 1)
          return Value();

        Value constOne = builder.create<mlir::smt::BVConstantOp>(loc, 1, 1);
        return builder.create<mlir::smt::EqOp>(loc, inputs[0], constOne);
      });

  // Default source materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        ->getResult(0);
  });
}

void circt::populateHWToSMTConversionPatterns(TypeConverter &converter,
                                              RewritePatternSet &patterns) {
  patterns.add<HWConstantOpConversion, HWModuleOpConversion, OutputOpConversion,
               InstanceOpConversion, ReplaceWithInput<seq::ToClockOp>,
               ReplaceWithInput<seq::FromClockOp>, ArrayCreateOpConversion,
               ArrayGetOpConversion, ArrayInjectOpConversion>(
      converter, patterns.getContext());
}

void ConvertHWToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalOp<seq::FromClockOp>();
  target.addIllegalOp<seq::ToClockOp>();
  target.addLegalDialect<mlir::smt::SMTDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);
  populateHWToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();

  // Sort the functions topologically because 'hw.module' has a graph region
  // while 'func.func' is a regular SSACFG region. Real combinational cycles or
  // pseudo cycles through module instances are not supported yet.
  for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
    // Skip functions that are definitely not the result of lowering from
    // 'hw.module'
    if (func.getBody().getBlocks().size() != 1)
      continue;

    mlir::sortTopologically(&func.getBody().front());
  }
}
