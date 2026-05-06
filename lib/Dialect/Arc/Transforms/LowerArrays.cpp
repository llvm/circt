//===- LowerArrays.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <utility>

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "arc-lower-arrays"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERARRAYS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;
using ::llvm::enumerate;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerArraysPass : public arc::impl::LowerArraysBase<LowerArraysPass> {
  LowerArraysPass() = default;
  LowerArraysPass(const LowerArraysPass &pass) : LowerArraysPass() {}

  void runOnOperation() override;
};

Value AsIndex(Value value, OpBuilder &builder) {
  Location loc = builder.getUnknownLoc();
  if (Operation *parent = value.getDefiningOp()) {
    loc = parent->getLoc();
  }
  return arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                      value);
}

Value CloneArrayRef(Value value, OpBuilder &builder, Location loc) {
  Value newAlloc = ArrayRefAllocOp::create(builder, loc, value.getType(), {});
  return ArrayRefCopyOp::create(builder, loc, newAlloc, value);
}

struct ConvertFunc : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter &converter = *getTypeConverter();
    TypeConverter::SignatureConversion conversion(op.getNumArguments());

    SmallVector<Type> newArgTypes;
    SmallVector<Type> newResultTypes;
    assert(op.getBody().getBlocks().size() == 1);

    // Any array-typed results become parameters.
    Operation *ret = op.getBody().front().getTerminator();
    for (Value result : ret->getOperands()) {
      if (isa<ArrayType>(result.getType())) {
        Type newType = converter.convertType(result.getType());
        conversion.addInputs(newType);
        newArgTypes.push_back(newType);
      }
    }

    if (failed(converter.convertTypes(op.getArgumentTypes(), newArgTypes)) ||
        failed(converter.convertTypes(op.getResultTypes(), newResultTypes))) {
      return failure();
    }

    if (failed(converter.convertSignatureArgs(op.getArgumentTypes(),
                                              conversion)) ||
        failed(rewriter.convertRegionTypes(&op.getBody(), converter,
                                           &conversion))) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&] {
      op.setType(FunctionType::get(getContext(), newArgTypes, newResultTypes));
    });

    return success();
  }
};

struct ConvertReturn : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    int sretIndex = 0;
    SmallVector<Value> newOperands;
    for (Value operand : adaptor.getOperands()) {
      if (isa<ArrayRefType>(operand.getType())) {
        Value arg = func.getArgument(sretIndex++);
        Value copy =
            ArrayRefCopyOp::create(rewriter, op.getLoc(), arg, operand);
        newOperands.push_back(copy);
      } else {
        newOperands.push_back(operand);
      }
    }
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(newOperands); });
    return success();
  }
};

struct ConvertCall : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newOperands;
    for (Type resultType : op.getResultTypes()) {
      auto arrayType = dyn_cast<ArrayType>(resultType);
      if (!arrayType)
        continue;
      auto arrayRefType = ArrayRefType::get(arrayType.getElementType(),
                                            arrayType.getNumElements());
      Value alloc =
          ArrayRefAllocOp::create(rewriter, op.getLoc(), arrayRefType, {});
      newOperands.push_back(alloc);
    }
    for (Value operand : adaptor.getOperands()) {
      newOperands.push_back(operand);
    }

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    }

    auto newCall = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                        resultTypes, newOperands);
    newCall->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    rewriter.replaceOp(op, newCall);
    return success();
  }
};

struct ConvertAggregateConstant
    : public OpConversionPattern<AggregateConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    Value newOp = ArrayRefAllocOp::create(rewriter, op.getLoc(), newType,
                                          op.getFieldsAttr());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArrayGet : public OpConversionPattern<ArrayGetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value index = AsIndex(op.getIndex(), rewriter);
    Type resultType = getTypeConverter()->convertType(op.getType());
    Value newOp = ArrayRefGetOp::create(rewriter, op.getLoc(), resultType,
                                        adaptor.getInput(), index);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArrayInject : public OpConversionPattern<ArrayInjectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value index = AsIndex(op.getIndex(), rewriter);
    Value dest = CloneArrayRef(adaptor.getInput(), rewriter, op.getLoc());
    Value newOp =
        ArrayRefInjectOp::create(rewriter, op.getLoc(), dest.getType(), dest,
                                 index, adaptor.getElement());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArraySlice : public OpConversionPattern<ArraySliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Because we're converting from value semantics and ArrayRefSliceOp is
    // a subview-like operator, we must clone the input array first.
    Value newInput = CloneArrayRef(adaptor.getInput(), rewriter, op.getLoc());
    Value index = AsIndex(op.getLowIndex(), rewriter);
    Type destType = getTypeConverter()->convertType(op.getType());
    Value newOp = ArrayRefSliceOp::create(rewriter, op.getLoc(), destType,
                                          newInput, index);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArrayConcat : public OpConversionPattern<ArrayConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type destType = getTypeConverter()->convertType(op.getType());
    Value dest = ArrayRefAllocOp::create(rewriter, op.getLoc(), destType, {});

    // ArrayConcatOp's operands are ordered from most significant to least
    // significant.
    int offset = cast<ArrayRefType>(destType).getNumElements();
    for (Value operand : adaptor.getOperands()) {
      offset -= cast<ArrayRefType>(operand.getType()).getNumElements();
      Value index =
          arith::ConstantIndexOp::create(rewriter, op.getLoc(), offset);
      Value destSlice = ArrayRefSliceOp::create(rewriter, op.getLoc(),
                                                operand.getType(), dest, index);
      ArrayRefCopyOp::create(rewriter, op.getLoc(), destSlice, operand);
    }
    assert(offset == 0);
    rewriter.replaceOp(op, dest);
    return success();
  }
};

struct ConvertArrayCreate : public OpConversionPattern<ArrayCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    Value alloc = ArrayRefAllocOp::create(rewriter, op.getLoc(), newType, {});
    Value create = ArrayRefCreateOp::create(rewriter, op.getLoc(), newType,
                                            alloc, adaptor.getInputs());
    rewriter.replaceOp(op, create);
    return success();
  }
};

struct ConvertStorageGet : public OpConversionPattern<StorageGetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StorageGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = convertOpResultTypes(op, adaptor.getOperands(),
                                       *getTypeConverter(), rewriter);
    if (failed(result))
      return failure();

    rewriter.replaceOp(op, *result);
    return success();
  }
};

struct ConvertMux : public OpConversionPattern<comb::MuxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    Value newOp = arith::SelectOp::create(
        rewriter, op.getLoc(), newType, adaptor.getCond(),
        adaptor.getTrueValue(), adaptor.getFalseValue());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Identifies a return of an ArrayRef that is defined by an ArrayRefAllocOp,
// and replaces the alloc with the sret buffer. Also removes the copy.
struct OptimizeReturnOfAlloc : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    bool changed = false;

    // Iterate over all pairs of (result, sret-buffer). The sret buffers are
    // always the initial function arguments, and every !arc.arrayref<T>
    // result has one sret buffer associated with it.
    auto args = funcOp.getArguments();
    auto results =
        llvm::make_filter_range(op->getOpOperands(), [](OpOperand &operand) {
          return isa<ArrayRefType>(operand.get().getType());
        });

    for (auto [arg, result] : llvm::zip(args, results)) {
      Value resultValue = result.get();
      auto copy = resultValue.getDefiningOp<ArrayRefCopyOp>();
      if (!copy || copy.getDestInput() != arg)
        continue;
      ArrayRefAllocOp alloc = getUltimatelyDefiningAlloc(copy.getSource());
      if (!alloc || alloc.getInit())
        continue;
      result.set(copy.getSource());
      rewriter.replaceAllUsesWith(alloc, arg);
      rewriter.eraseOp(alloc);
      rewriter.eraseOp(copy);
      changed = true;
    }
    return success(changed);
  }

  ArrayRefAllocOp getUltimatelyDefiningAlloc(Value value) const {
    if (!isa<ArrayRefType>(value.getType()))
      return nullptr;
    while (value) {
      Operation *op = value.getDefiningOp();
      if (!op)
        return nullptr;
      if (auto alloc = dyn_cast<ArrayRefAllocOp>(op))
        return alloc;

      value = TypeSwitch<Operation *, Value>(op)
                  .Case<ArrayRefCopyOp>(
                      [&](auto copy) { return copy.getDestInput(); })
                  .Case<ArrayRefInjectOp>(
                      [&](auto inject) { return inject.getInput(); })
                  .Case<func::CallOp>([&](auto call) {
                    OpResult result = cast<OpResult>(value);
                    return call.getOperand(result.getResultNumber());
                  })
                  .Default([&](Operation *) { return nullptr; });
    }
    return nullptr;
  }
};

template <typename Op>
struct ConvertTrivially : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = convertOpResultTypes(op, adaptor.getOperands(),
                                       *this->getTypeConverter(), rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

using ConvertAllocState = ConvertTrivially<arc::AllocStateOp>;
using ConvertStateRead = ConvertTrivially<arc::StateReadOp>;
using ConvertStateWrite = ConvertTrivially<arc::StateWriteOp>;
using ConvertRootInput = ConvertTrivially<arc::RootInputOp>;
using ConvertRootOutput = ConvertTrivially<arc::RootOutputOp>;
using ConvertUnrealizedConversionCast =
    ConvertTrivially<UnrealizedConversionCastOp>;

} // namespace

void LowerArraysPass::runOnOperation() {
  TypeConverter converter;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());

  converter.addConversion([](Type type) { return type; });
  converter.addConversion([](ArrayType type) {
    return ArrayRefType::get(type.getElementType(), type.getNumElements());
  });
  converter.addConversion([&converter](StateType type) {
    return StateType::get(converter.convertType(type.getType()));
  });

  target.addIllegalOp<ArrayCreateOp, ArrayConcatOp, ArrayGetOp, ArrayInjectOp,
                      ArraySliceOp>();
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
    FunctionType fty = func.getFunctionType();
    return converter.isLegal(fty.getInputs()) &&
           converter.isLegal(fty.getResults());
  });

  patterns.add<ConvertFunc, ConvertReturn, ConvertCall,
               ConvertAggregateConstant, ConvertArrayGet, ConvertArrayInject,
               ConvertArraySlice, ConvertArrayConcat, ConvertArrayCreate,
               ConvertStorageGet, ConvertMux, ConvertAllocState,
               ConvertStateRead, ConvertStateWrite, ConvertRootInput,
               ConvertRootOutput, ConvertUnrealizedConversionCast>(
      converter, &getContext());

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }

  // Apply some cleanup patterns to optimize away ArrayRefAllocOps.
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<OptimizeReturnOfAlloc>(&getContext());
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(cleanupPatterns)))) {
    return signalPassFailure();
  }
}
