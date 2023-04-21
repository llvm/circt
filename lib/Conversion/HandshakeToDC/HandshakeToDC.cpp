//===- HandshakeToDC.cpp - Translate Handshake into DC --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to DC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HandshakeToDC.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::dc;

namespace {

using ConvertedOps = DenseSet<Operation *>;

class DCTypeConverter : public TypeConverter {
public:
  DCTypeConverter() {
    addConversion([](Type type) -> Type {
      if (type.isa<NoneType>())
        return dc::TokenType::get(type.getContext());
      else
        return dc::ValueType::get(type.getContext(), type);
    });

    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;

          // Materialize !dc.value<> -> !dc.token
          if (resultType.isa<dc::TokenType>() &&
              inputs.front().getType().cast<dc::ValueType>())
            return builder.create<dc::UnpackOp>(loc, inputs.front()).getToken();

          // Materialize !dc.token -> !dc.value<>
          auto vt = resultType.dyn_cast<dc::ValueType>();
          if (vt && vt.getInnerTypes().empty())
            return builder.create<dc::PackOp>(loc, vt, inputs.front(),
                                              ValueRange{});

          return inputs[0];
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;

          // Materialize !dc.value<> -> !dc.token
          if (resultType.isa<dc::TokenType>() &&
              inputs.front().getType().cast<dc::ValueType>())
            return builder.create<dc::UnpackOp>(loc, inputs.front()).getToken();

          // Materialize !dc.token -> !dc.value<>
          auto vt = resultType.dyn_cast<dc::ValueType>();
          if (vt && vt.getInnerTypes().empty())
            return builder.create<dc::PackOp>(loc, vt, inputs.front(),
                                              ValueRange{});

          return inputs[0];
        });
  }
};

struct DCTuple {
  DCTuple() {}
  DCTuple(Value token, ValueRange data) : token(token), data(data) {}
  DCTuple(dc::UnpackOp unpack)
      : token(unpack.getToken()), data(unpack.getOutputs()) {}
  Value token;
  ValueRange data;
};

// Unpack a !dc.value<...> into a DCTuple.
static DCTuple unpack(OpBuilder &b, Value v) {
  if (v.getType().isa<dc::ValueType>())
    return DCTuple(b.create<dc::UnpackOp>(v.getLoc(), v));
  else {
    assert(v.getType().isa<dc::TokenType>() && "Expected a dc::TokenType");
    return DCTuple(v, ValueRange{});
  }
}

template <typename OpTy>
class DCOpConversionPattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  DCOpConversionPattern(MLIRContext *context, TypeConverter &typeConverter,
                        ConvertedOps *convertedOps)
      : OpConversionPattern<OpTy>(typeConverter, context),
        convertedOps(convertedOps) {}
  mutable ConvertedOps *convertedOps;
};

class CondBranchConversionPattern
    : public DCOpConversionPattern<handshake::ConditionalBranchOp> {
public:
  using DCOpConversionPattern<
      handshake::ConditionalBranchOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::ConditionalBranchOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::ConditionalBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto condition = unpack(rewriter, adaptor.getConditionOperand());
    auto data = unpack(rewriter, adaptor.getDataOperand());

    // Join the token of the condition and the input.
    auto join = rewriter.create<dc::JoinOp>(
        op.getLoc(), ValueRange{condition.token, data.token});

    // Pack that together with the condition data.
    auto packedCondition = rewriter.create<dc::PackOp>(
        op.getLoc(), join, ValueRange{condition.data});

    // Branch on the input data and the joined control input.
    auto branch = rewriter.create<dc::BranchOp>(op.getLoc(), packedCondition);

    // Pack the branch output tokens with the input data, and replace the uses.
    llvm::SmallVector<Value, 4> packed;
    packed.push_back(rewriter.create<dc::PackOp>(
        op.getLoc(), branch.getTrueToken(), ValueRange{data.data}));
    packed.push_back(rewriter.create<dc::PackOp>(
        op.getLoc(), branch.getFalseToken(), ValueRange{data.data}));

    rewriter.replaceOp(op, packed);
    return success();
  }
};

class ForkOpConversionPattern
    : public DCOpConversionPattern<handshake::ForkOp> {
public:
  using DCOpConversionPattern<handshake::ForkOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::ForkOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::ForkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = unpack(rewriter, adaptor.getOperand());
    auto forkOut = rewriter.create<dc::ForkOp>(op.getLoc(), input.token,
                                               op.getNumResults());

    // Pack the fork result tokens with the input data, and replace the uses.
    llvm::SmallVector<Value, 4> packed;
    for (auto res : forkOut.getResults())
      packed.push_back(rewriter.create<dc::PackOp>(op.getLoc(), res,
                                                   ValueRange{input.data}));

    rewriter.replaceOp(op, packed);
    return success();
  }
};

class JoinOpConversion : public DCOpConversionPattern<handshake::JoinOp> {
public:
  using DCOpConversionPattern<handshake::JoinOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::JoinOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value, 4> inputTokens;
    for (auto input : adaptor.getData())
      inputTokens.push_back(unpack(rewriter, input).token);

    rewriter.replaceOpWithNewOp<dc::JoinOp>(op, inputTokens);
    return success();
  }
};

class SyncOpConversion : public DCOpConversionPattern<handshake::SyncOp> {
public:
  using DCOpConversionPattern<handshake::SyncOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::SyncOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::SyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value, 4> inputTokens;
    for (auto input : adaptor.getOperands())
      inputTokens.push_back(unpack(rewriter, input).token);

    auto syncToken = rewriter.create<dc::JoinOp>(op.getLoc(), inputTokens);

    // Wrap all outputs with the synchronization token
    llvm::SmallVector<Value, 4> wrappedInputs;
    for (auto input : adaptor.getOperands())
      wrappedInputs.push_back(rewriter.create<dc::PackOp>(
          op.getLoc(), syncToken, ValueRange{input}));

    rewriter.replaceOp(op, wrappedInputs);

    return success();
  }
};

class ConstantOpConversion
    : public DCOpConversionPattern<handshake::ConstantOp> {
public:
  using DCOpConversionPattern<handshake::ConstantOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::ConstantOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Wrap the constant with a token.
    auto token = rewriter.create<dc::SourceOp>(op.getLoc());
    auto cst =
        rewriter.create<arith::ConstantOp>(op.getLoc(), adaptor.getValue());
    convertedOps->insert(cst);
    rewriter.replaceOpWithNewOp<dc::PackOp>(op, token,
                                            llvm::SmallVector<Value>{cst});
    return success();
  }
};

struct UnitRateConversionPattern : public ConversionPattern {
public:
  UnitRateConversionPattern(MLIRContext *context, TypeConverter &converter,
                            ConvertedOps *joinedOps)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context),
        joinedOps(joinedOps) {}
  using ConversionPattern::ConversionPattern;

  // Generic pattern which replaces an operation by one of the same type, but
  // with the in- and outputs synchronized through join semantics.
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return op->emitOpError("expected single result for pattern to apply");

    llvm::SmallVector<Value, 4> inputData;
    llvm::SmallVector<Value, 4> inputTokens;
    for (auto input : operands) {
      auto dct = unpack(rewriter, input);
      inputData.append(dct.data.begin(), dct.data.end());
      inputTokens.push_back(dct.token);
    }

    // Join the tokens of the inputs.
    auto join = rewriter.create<dc::JoinOp>(op->getLoc(), inputTokens);

    // Patchwork to fix bad IR design in Handshake.
    auto opName = op->getName();
    if (opName.getStringRef() == "handshake.select") {
      opName = OperationName("arith.select", getContext());
    } else if (opName.getStringRef() == "handshake.constant") {
      opName = OperationName("arith.constant", getContext());
    }

    // Re-create the operation using the unpacked input data.
    OperationState state(op->getLoc(), opName, inputData, op->getResultTypes(),
                         op->getAttrs(), op->getSuccessors());

    Operation *newOp = rewriter.create(state);
    joinedOps->insert(newOp);

    // Pack the result token with the output data, and replace the use.
    rewriter.replaceOp(op,
                       rewriter
                           .create<dc::PackOp>(op->getLoc(), join.getResult(),
                                               newOp->getResults())
                           ->getResults());

    return success();
  }

  mutable ConvertedOps *joinedOps;
};

class SinkOpConversionPattern
    : public DCOpConversionPattern<handshake::SinkOp> {
public:
  using DCOpConversionPattern<handshake::SinkOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::SinkOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::SinkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = unpack(rewriter, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<dc::SinkOp>(op, input.token);
    return success();
  }
};

class SourceOpConversionPattern
    : public DCOpConversionPattern<handshake::SourceOp> {
public:
  using DCOpConversionPattern<handshake::SourceOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::SourceOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<dc::SourceOp>(op);
    return success();
  }
};

class BufferOpConversion : public DCOpConversionPattern<handshake::BufferOp> {
public:
  using DCOpConversionPattern<handshake::BufferOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::BufferOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::BufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.getI32IntegerAttr(1);
    rewriter.replaceOpWithNewOp<dc::BufferOp>(
        op, adaptor.getOperand(), static_cast<size_t>(op.getNumSlots()));
    return success();
  }
};

class ReturnOpConversion : public DCOpConversionPattern<handshake::ReturnOp> {
public:
  using DCOpConversionPattern<handshake::ReturnOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::ReturnOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<dc::ReturnOp>(op, adaptor.getOpOperands());
    return success();
  }
};

class MuxOpConversionPattern : public DCOpConversionPattern<handshake::MuxOp> {
public:
  using DCOpConversionPattern<handshake::MuxOp>::DCOpConversionPattern;
  using OpAdaptor = typename DCOpConversionPattern<handshake::MuxOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(handshake::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto select = unpack(rewriter, adaptor.getSelectOperand());
    auto selectData = select.data.front();
    bool isIndexType = selectData.getType().isa<IndexType>();

    bool withData = !op.getResult().getType().isa<NoneType>();

    llvm::SmallVector<DCTuple> inputs;
    for (auto input : adaptor.getDataOperands())
      inputs.push_back(unpack(rewriter, input));

    Value dataMux;
    // Convert the data-side mux to a sequence of arith.select operations.
    // The data and control muxes are assumed one-hot and the base-case is set
    // as the first input.
    if (withData)
      dataMux = inputs[0].data.front();

    llvm::SmallVector<Value> controlMuxInputs = {inputs.front().token};
    for (auto [i, input] :
         llvm::enumerate(llvm::make_range(inputs.begin() + 1, inputs.end()))) {
      controlMuxInputs.push_back(input.token);
      if (withData) {
        Value cmpIndex;
        Value inputData = input.data.front();
        if (isIndexType) {
          cmpIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
        } else {
          size_t width = selectData.getType().cast<IntegerType>().getWidth();
          cmpIndex =
              rewriter.create<arith::ConstantIntOp>(op.getLoc(), i, width);
        }
        auto inputSelected = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::eq, selectData, cmpIndex);
        dataMux = rewriter.create<arith::SelectOp>(op.getLoc(), inputSelected,
                                                   inputData, dataMux);

        // Legalize the newly created operations.
        convertedOps->insert(cmpIndex.getDefiningOp());
        convertedOps->insert(dataMux.getDefiningOp());
        convertedOps->insert(inputSelected);
      }
    }

    // Convert the control-side into a dc.merge operation.
    auto controlMux = rewriter.create<dc::MergeOp>(
        op.getLoc(), adaptor.getSelectOperand(), controlMuxInputs);

    // finally, pack the dc.token-side muxing with the data-side mux.
    rewriter.replaceOpWithNewOp<dc::PackOp>(
        op, controlMux, withData ? ValueRange{dataMux} : ValueRange{});
    return success();
  }
};

class FuncOpConversion : public DCOpConversionPattern<handshake::FuncOp> {
public:
  using DCOpConversionPattern<handshake::FuncOp>::DCOpConversionPattern;
  using OpAdaptor =
      typename DCOpConversionPattern<handshake::FuncOp>::OpAdaptor;

  // Replaces a handshake.func with a func.func, converting the argument and
  // result types using the provided type converter.
  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the function signature.
    auto funcType = op.getFunctionType();
    llvm::SmallVector<Type, 4> argTypes, resTypes;
    if (failed(typeConverter->convertTypes(funcType.getInputs(), argTypes)) ||
        failed(typeConverter->convertTypes(funcType.getResults(), resTypes)))
      return failure();

    auto newFuncType =
        FunctionType::get(funcType.getContext(), argTypes, resTypes);

    // Create the new function.
    auto newFuncOp =
        rewriter.create<dc::FuncOp>(op.getLoc(), op.getName(), newFuncType);

    // Convert the function body.
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    DCTypeConverter::SignatureConversion result(funcType.getNumInputs());
    (void)typeConverter->convertSignatureArgs(
        newFuncOp.getBody().getArgumentTypes(), result);
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // Replace the old function.
    rewriter.replaceOp(op, newFuncOp.getOperation()->getResults());
    return success();
  }
};

class HandshakeToDCPass : public HandshakeToDCBase<HandshakeToDCPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Maintain the set of operations which has been converted either through
    // unit rate conversion, or as part of other conversions.
    ConvertedOps convertedOps;

    ConversionTarget target(getContext());
    target.addIllegalDialect<handshake::HandshakeDialect>();
    target.addLegalDialect<dc::DCDialect, func::FuncDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    // The various patterns will insert new operations into the module to
    // facilitate the conversion - however, these operations must be
    // distinguishable from already converted operations (which may be of the
    // same type as the newly inserted operations). To do this, we mark all
    // operations which have been converted as legal, and all other operations
    // as illegal.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return convertedOps.contains(op); });

    DCTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());

    // Add handshake conversion patterns.
    // Note: merge/control merge are not supported - these are non-deterministic
    // operators and we do not care for them.
    patterns.add<FuncOpConversion, BufferOpConversion,
                 CondBranchConversionPattern, SinkOpConversionPattern,
                 SourceOpConversionPattern, MuxOpConversionPattern,
                 ReturnOpConversion, ForkOpConversionPattern, JoinOpConversion,
                 ConstantOpConversion, SyncOpConversion>(
        &getContext(), typeConverter, &convertedOps);

    // ALL other single-result operations are converted via the
    // UnitRateConversionPattern.
    patterns.add<UnitRateConversionPattern>(&getContext(), typeConverter,
                                            &convertedOps);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createHandshakeToDCPass() {
  return std::make_unique<HandshakeToDCPass>();
}
