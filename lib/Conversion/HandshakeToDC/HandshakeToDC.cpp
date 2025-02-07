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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace circt {
#define GEN_PASS_DEF_HANDSHAKETODC
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace handshake;
using namespace dc;
using namespace hw;
using namespace handshaketodc;

namespace {

struct DCTuple {
  DCTuple() = default;
  DCTuple(Value token, Value data) : token(token), data(data) {}
  DCTuple(dc::UnpackOp unpack)
      : token(unpack.getToken()), data(unpack.getOutput()) {}
  Value token;
  Value data;
};

// Unpack a !dc.value<...> into a DCTuple.
static DCTuple unpack(OpBuilder &b, Value v) {
  if (isa<dc::ValueType>(v.getType()))
    return DCTuple(b.create<dc::UnpackOp>(v.getLoc(), v));
  assert(isa<dc::TokenType>(v.getType()) && "Expected a dc::TokenType");
  return DCTuple(v, {});
}

static Value pack(OpBuilder &b, Value token, Value data = {}) {
  if (!data)
    return token;
  return b.create<dc::PackOp>(token.getLoc(), token, data);
}

// NOLINTNEXTLINE(misc-no-recursion)
static StructType tupleToStruct(TupleType tuple) {
  auto *ctx = tuple.getContext();
  mlir::SmallVector<hw::StructType::FieldInfo, 8> hwfields;
  for (auto [i, innerType] : llvm::enumerate(tuple)) {
    Type convertedInnerType = innerType;
    if (auto tupleInnerType = dyn_cast<TupleType>(innerType))
      convertedInnerType = tupleToStruct(tupleInnerType);
    hwfields.push_back(
        {StringAttr::get(ctx, "field" + Twine(i)), convertedInnerType});
  }

  return hw::StructType::get(ctx, hwfields);
}

class DCTypeConverter : public TypeConverter {
public:
  DCTypeConverter() {
    addConversion([](Type type) -> Type {
      if (isa<NoneType>(type))
        return dc::TokenType::get(type.getContext());

      // For pragmatic reasons, we use a struct type to represent tuples in the
      // DC lowering; upstream MLIR doesn't have builtin type-modifying ops,
      // so the next best thing is our "local" struct type in CIRCT.
      if (auto tupleType = dyn_cast<TupleType>(type))
        return dc::ValueType::get(type.getContext(), tupleToStruct(tupleType));
      return dc::ValueType::get(type.getContext(), type);
    });
    addConversion([](ValueType type) { return type; });
    addConversion([](TokenType type) { return type; });

    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      // Materialize !dc.value<> -> !dc.token
      if (isa<dc::TokenType>(resultType) &&
          isa<dc::ValueType>(inputs.front().getType()))
        return unpack(builder, inputs.front()).token;

      // Materialize !dc.token -> !dc.value<>
      auto vt = dyn_cast<dc::ValueType>(resultType);
      if (vt && !vt.getInnerType())
        return pack(builder, inputs.front());

      return builder
          .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
          ->getResult(0);
    });

    addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      // Materialize !dc.value<> -> !dc.token
      if (isa<dc::TokenType>(resultType) &&
          isa<dc::ValueType>(inputs.front().getType()))
        return unpack(builder, inputs.front()).token;

      // Materialize !dc.token -> !dc.value<>
      auto vt = dyn_cast<dc::ValueType>(resultType);
      if (vt && !vt.getInnerType())
        return pack(builder, inputs.front());

      return builder
          .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
          ->getResult(0);
    });
  }
};

template <typename OpTy>
class DCOpConversionPattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

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
  using OpAdaptor = typename handshake::ConditionalBranchOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ConditionalBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto condition = unpack(rewriter, adaptor.getConditionOperand());
    auto data = unpack(rewriter, adaptor.getDataOperand());

    // Join the token of the condition and the input.
    auto join = rewriter.create<dc::JoinOp>(
        op.getLoc(), ValueRange{condition.token, data.token});

    // Pack that together with the condition data.
    auto packedCondition = pack(rewriter, join, condition.data);

    // Branch on the input data and the joined control input.
    auto branch = rewriter.create<dc::BranchOp>(op.getLoc(), packedCondition);

    // Pack the branch output tokens with the input data, and replace the uses.
    llvm::SmallVector<Value, 4> packed;
    packed.push_back(pack(rewriter, branch.getTrueToken(), data.data));
    packed.push_back(pack(rewriter, branch.getFalseToken(), data.data));

    rewriter.replaceOp(op, packed);
    return success();
  }
};

class ForkOpConversionPattern
    : public DCOpConversionPattern<handshake::ForkOp> {
public:
  using DCOpConversionPattern<handshake::ForkOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::ForkOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ForkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = unpack(rewriter, adaptor.getOperand());
    auto forkOut = rewriter.create<dc::ForkOp>(op.getLoc(), input.token,
                                               op.getNumResults());

    // Pack the fork result tokens with the input data, and replace the uses.
    llvm::SmallVector<Value, 4> packed;
    for (auto res : forkOut.getResults())
      packed.push_back(pack(rewriter, res, input.data));

    rewriter.replaceOp(op, packed);
    return success();
  }
};

class JoinOpConversion : public DCOpConversionPattern<handshake::JoinOp> {
public:
  using DCOpConversionPattern<handshake::JoinOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::JoinOp::Adaptor;

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

class MergeOpConversion : public DCOpConversionPattern<handshake::MergeOp> {
public:
  using DCOpConversionPattern<handshake::MergeOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::MergeOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::MergeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumOperands() > 2)
      return rewriter.notifyMatchFailure(op, "only two inputs supported");

    SmallVector<Value, 4> tokens, data;

    for (auto input : adaptor.getDataOperands()) {
      auto up = unpack(rewriter, input);
      tokens.push_back(up.token);
      if (up.data)
        data.push_back(up.data);
    }

    // Control side
    Value selectedIndex = rewriter.create<dc::MergeOp>(op.getLoc(), tokens);
    auto selectedIndexUnpacked = unpack(rewriter, selectedIndex);
    Value mergeOutput;

    if (!data.empty()) {
      // Data-merge; mux the selected input.
      auto dataMux = rewriter.create<arith::SelectOp>(
          op.getLoc(), selectedIndexUnpacked.data, data[0], data[1]);
      convertedOps->insert(dataMux);

      // Pack the data mux with the control token.
      mergeOutput = pack(rewriter, selectedIndexUnpacked.token, dataMux);
    } else {
      // Control-only merge; throw away the index value of the dc.merge
      // operation and only forward the dc.token.
      mergeOutput = selectedIndexUnpacked.token;
    }

    rewriter.replaceOp(op, mergeOutput);
    return success();
  }
};

class PackOpConversion : public DCOpConversionPattern<handshake::PackOp> {
public:
  using DCOpConversionPattern<handshake::PackOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::PackOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Like the join conversion, but also emits a dc.pack_tuple operation to
    // handle the data side of the operation (since there's no upstream support
    // for doing so, sigh...)
    llvm::SmallVector<Value, 4> inputTokens, inputData;
    for (auto input : adaptor.getOperands()) {
      DCTuple dct = unpack(rewriter, input);
      inputTokens.push_back(dct.token);
      if (dct.data)
        inputData.push_back(dct.data);
    }

    auto join = rewriter.create<dc::JoinOp>(op.getLoc(), inputTokens);
    StructType structType =
        tupleToStruct(cast<TupleType>(op.getResult().getType()));
    auto packedData =
        rewriter.create<hw::StructCreateOp>(op.getLoc(), structType, inputData);
    convertedOps->insert(packedData);
    rewriter.replaceOp(op, pack(rewriter, join, packedData));
    return success();
  }
};

class UnpackOpConversion : public DCOpConversionPattern<handshake::UnpackOp> {
public:
  using DCOpConversionPattern<handshake::UnpackOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::UnpackOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Unpack the !dc.value<tuple<...>> into the !dc.token and tuple<...>
    // values.
    DCTuple unpackedInput = unpack(rewriter, adaptor.getInput());
    auto unpackedData =
        rewriter.create<hw::StructExplodeOp>(op.getLoc(), unpackedInput.data);
    convertedOps->insert(unpackedData);
    // Re-pack each of the tuple elements with the token.
    llvm::SmallVector<Value, 4> repackedInputs;
    for (auto outputData : unpackedData.getResults())
      repackedInputs.push_back(pack(rewriter, unpackedInput.token, outputData));

    rewriter.replaceOp(op, repackedInputs);
    return success();
  }
};

class ControlMergeOpConversion
    : public DCOpConversionPattern<handshake::ControlMergeOp> {
public:
  using DCOpConversionPattern<handshake::ControlMergeOp>::DCOpConversionPattern;

  using OpAdaptor = typename handshake::ControlMergeOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ControlMergeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDataOperands().size() != 2)
      return op.emitOpError("expected two data operands");

    llvm::SmallVector<Value> tokens, data;
    for (auto input : adaptor.getDataOperands()) {
      auto up = unpack(rewriter, input);
      tokens.push_back(up.token);
      if (up.data)
        data.push_back(up.data);
    }

    bool isIndexType = isa<IndexType>(op.getIndex().getType());

    // control-side
    Value selectedIndex = rewriter.create<dc::MergeOp>(op.getLoc(), tokens);
    auto mergeOpUnpacked = unpack(rewriter, selectedIndex);
    auto selValue = mergeOpUnpacked.data;

    Value dataSide = selectedIndex;
    if (!data.empty()) {
      // Data side mux using the selected input.
      auto dataMux = rewriter.create<arith::SelectOp>(op.getLoc(), selValue,
                                                      data[0], data[1]);
      convertedOps->insert(dataMux);
      // Pack the data mux with the control token.
      auto packed = pack(rewriter, mergeOpUnpacked.token, dataMux);

      dataSide = packed;
    }

    // if the original op used `index` as the select operand type, we need to
    // index-cast the unpacked select operand
    if (isIndexType) {
      selValue = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), selValue);
      convertedOps->insert(selValue.getDefiningOp());
      selectedIndex = pack(rewriter, mergeOpUnpacked.token, selValue);
    } else {
      // The cmerge had a specific type defined for the index type. dc.merge
      // provides an i1 operand for the selected index, so we need to cast it.
      selValue = rewriter.create<arith::ExtUIOp>(
          op.getLoc(), op.getIndex().getType(), selValue);
      convertedOps->insert(selValue.getDefiningOp());
      selectedIndex = pack(rewriter, mergeOpUnpacked.token, selValue);
    }

    rewriter.replaceOp(op, {dataSide, selectedIndex});
    return success();
  }
};

class SyncOpConversion : public DCOpConversionPattern<handshake::SyncOp> {
public:
  using DCOpConversionPattern<handshake::SyncOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::SyncOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::SyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value, 4> inputTokens;
    llvm::SmallVector<Value, 4> inputData;
    for (auto input : adaptor.getOperands()) {
      auto unpacked = unpack(rewriter, input);
      inputTokens.push_back(unpacked.token);
      inputData.push_back(unpacked.data);
    }

    auto syncToken = rewriter.create<dc::JoinOp>(op.getLoc(), inputTokens);

    // Wrap all outputs with the synchronization token
    llvm::SmallVector<Value, 4> wrappedInputs;
    for (auto inputData : inputData)
      wrappedInputs.push_back(pack(rewriter, syncToken, inputData));

    rewriter.replaceOp(op, wrappedInputs);

    return success();
  }
};

class ConstantOpConversion
    : public DCOpConversionPattern<handshake::ConstantOp> {
public:
  using DCOpConversionPattern<handshake::ConstantOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::ConstantOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Wrap the constant with a token.
    auto token = rewriter.create<dc::SourceOp>(op.getLoc());
    auto cst =
        rewriter.create<arith::ConstantOp>(op.getLoc(), adaptor.getValue());
    convertedOps->insert(cst);
    rewriter.replaceOp(op, pack(rewriter, token, cst));
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
    llvm::SmallVector<Value> inputData;

    Value outToken;
    if (operands.empty()) {
      if (!op->hasTrait<OpTrait::ConstantLike>())
        return op->emitOpError(
            "no-operand operation which isn't constant-like. Too dangerous "
            "to assume semantics - won't convert");

      // Constant-like operation; assume the token can be represented as a
      // constant `dc.source`.
      outToken = rewriter.create<dc::SourceOp>(op->getLoc());
    } else {
      llvm::SmallVector<Value> inputTokens;
      for (auto input : operands) {
        auto dct = unpack(rewriter, input);
        inputData.push_back(dct.data);
        inputTokens.push_back(dct.token);
      }
      // Join the tokens of the inputs.
      assert(!inputTokens.empty() && "Expected at least one input token");
      outToken = rewriter.create<dc::JoinOp>(op->getLoc(), inputTokens);
    }

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

    // Pack the result token with the output data, and replace the uses.
    llvm::SmallVector<Value> results;
    for (auto result : newOp->getResults())
      results.push_back(pack(rewriter, outToken, result));

    rewriter.replaceOp(op, results);

    return success();
  }

  mutable ConvertedOps *joinedOps;
};

class SinkOpConversionPattern
    : public DCOpConversionPattern<handshake::SinkOp> {
public:
  using DCOpConversionPattern<handshake::SinkOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::SinkOp::Adaptor;

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
  using OpAdaptor = typename handshake::SourceOp::Adaptor;

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
  using OpAdaptor = typename handshake::BufferOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::BufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.getI32IntegerAttr(1);
    rewriter.replaceOpWithNewOp<dc::BufferOp>(
        op, adaptor.getOperand(), static_cast<size_t>(op.getNumSlots()),
        op.getInitValuesAttr());
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<handshake::ReturnOp> {
public:
  using OpConversionPattern<handshake::ReturnOp>::OpConversionPattern;
  using OpAdaptor = typename handshake::ReturnOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Locate existing output op, Append operands to output op, and move to
    // the end of the block.
    auto hwModule = op->getParentOfType<hw::HWModuleOp>();
    auto outputOp = *hwModule.getBodyBlock()->getOps<hw::OutputOp>().begin();
    outputOp->setOperands(adaptor.getOperands());
    outputOp->moveAfter(&hwModule.getBodyBlock()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

class MuxOpConversionPattern : public DCOpConversionPattern<handshake::MuxOp> {
public:
  using DCOpConversionPattern<handshake::MuxOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::MuxOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto select = unpack(rewriter, adaptor.getSelectOperand());
    auto selectData = select.data;
    auto selectToken = select.token;
    bool isIndexType = isa<IndexType>(selectData.getType());

    bool withData = !isa<NoneType>(op.getResult().getType());

    llvm::SmallVector<DCTuple> inputs;
    for (auto input : adaptor.getDataOperands())
      inputs.push_back(unpack(rewriter, input));

    Value dataMux;
    Value controlMux = inputs.front().token;
    // Convert the data-side mux to a sequence of arith.select operations.
    // The data and control muxes are assumed one-hot and the base-case is set
    // as the first input.
    if (withData)
      dataMux = inputs[0].data;

    llvm::SmallVector<Value> controlMuxInputs = {inputs.front().token};
    for (auto [i, input] :
         llvm::enumerate(llvm::make_range(inputs.begin() + 1, inputs.end()))) {
      if (!withData)
        continue;

      Value cmpIndex;
      Value inputData = input.data;
      Value inputControl = input.token;
      if (isIndexType) {
        cmpIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
      } else {
        size_t width = cast<IntegerType>(selectData.getType()).getWidth();
        cmpIndex = rewriter.create<arith::ConstantIntOp>(op.getLoc(), i, width);
      }
      auto inputSelected = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, selectData, cmpIndex);
      dataMux = rewriter.create<arith::SelectOp>(op.getLoc(), inputSelected,
                                                 inputData, dataMux);

      // Legalize the newly created operations.
      convertedOps->insert(cmpIndex.getDefiningOp());
      convertedOps->insert(dataMux.getDefiningOp());
      convertedOps->insert(inputSelected);

      // And similarly for the control mux, by muxing the input token with a
      // select value that has it's control from the original select token +
      // the inputSelected value.
      auto inputSelectedControl = pack(rewriter, selectToken, inputSelected);
      controlMux = rewriter.create<dc::SelectOp>(
          op.getLoc(), inputSelectedControl, inputControl, controlMux);
      convertedOps->insert(controlMux.getDefiningOp());
    }

    // finally, pack the control and data side muxes into the output value.
    rewriter.replaceOp(
        op, pack(rewriter, controlMux, withData ? dataMux : Value{}));
    return success();
  }
};

static hw::ModulePortInfo getModulePortInfoHS(const TypeConverter &tc,
                                              handshake::FuncOp funcOp) {
  SmallVector<hw::PortInfo> inputs, outputs;
  auto ft = funcOp.getFunctionType();
  funcOp.resolveArgAndResNames();

  // Add all inputs of funcOp.
  for (auto [index, type] : llvm::enumerate(ft.getInputs()))
    inputs.push_back({{funcOp.getArgName(index), tc.convertType(type),
                       hw::ModulePort::Direction::Input},
                      index,
                      {}});

  // Add all outputs of funcOp.
  for (auto [index, type] : llvm::enumerate(ft.getResults()))
    outputs.push_back({{funcOp.getResName(index), tc.convertType(type),
                        hw::ModulePort::Direction::Output},
                       index,
                       {}});

  return hw::ModulePortInfo{inputs, outputs};
}

class FuncOpConversion : public DCOpConversionPattern<handshake::FuncOp> {
public:
  using DCOpConversionPattern<handshake::FuncOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::FuncOp::Adaptor;

  // Replaces a handshake.func with a hw.module, converting the argument and
  // result types using the provided type converter.
  // @mortbopet: Not a fan of converting to hw here seeing as we don't
  // necessarily have hardware semantics here. But, DC doesn't define a
  // function operation, and there is no "func.graph_func" or any other
  // generic function operation which is a graph region...
  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModulePortInfo ports = getModulePortInfoHS(*getTypeConverter(), op);

    if (op.isExternal()) {
      auto mod = rewriter.create<hw::HWModuleExternOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
      convertedOps->insert(mod);
    } else {
      auto hwModule = rewriter.create<hw::HWModuleOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);

      auto &region = op->getRegions().front();

      Region &moduleRegion = hwModule->getRegions().front();
      rewriter.mergeBlocks(&region.getBlocks().front(), hwModule.getBodyBlock(),
                           hwModule.getBodyBlock()->getArguments());
      TypeConverter::SignatureConversion result(moduleRegion.getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          TypeRange(moduleRegion.getArgumentTypes()), result);
      rewriter.applySignatureConversion(hwModule.getBodyBlock(), result);
      convertedOps->insert(hwModule);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower the ESIInstanceOp to `hw.instance` with `dc.from_esi` and `dc.to_esi`
/// to convert the args/results.
class ESIInstanceConversionPattern
    : public OpConversionPattern<handshake::ESIInstanceOp> {
public:
  ESIInstanceConversionPattern(MLIRContext *context,
                               const HWSymbolCache &symCache)
      : OpConversionPattern(context), symCache(symCache) {}

  LogicalResult
  matchAndRewrite(ESIInstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> operands;
    for (size_t i = ESIInstanceOp::NumFixedOperands, e = op.getNumOperands();
         i < e; ++i)
      operands.push_back(
          rewriter.create<dc::FromESIOp>(loc, adaptor.getOperands()[i]));
    operands.push_back(adaptor.getClk());
    operands.push_back(adaptor.getRst());
    // Locate the lowered module so the instance builder can get all the
    // metadata.
    Operation *targetModule = symCache.getDefinition(op.getModuleAttr());
    // And replace the op with an instance of the target module.
    auto inst = rewriter.create<hw::InstanceOp>(loc, targetModule,
                                                op.getInstNameAttr(), operands);
    SmallVector<Value> esiResults(
        llvm::map_range(inst.getResults(), [&](Value v) {
          return rewriter.create<dc::ToESIOp>(loc, v);
        }));
    rewriter.replaceOp(op, esiResults);
    return success();
  }

private:
  const HWSymbolCache &symCache;
};

/// Add DC clock and reset ports to the module.
void addClkRst(hw::HWModuleOp mod, StringRef clkName, StringRef rstName) {
  auto *ctx = mod.getContext();

  size_t numInputs = mod.getNumInputPorts();
  mod.insertInput(numInputs, clkName, seq::ClockType::get(ctx));
  mod.setPortAttrs(
      numInputs,
      DictionaryAttr::get(ctx, {NamedAttribute(StringAttr::get(ctx, "dc.clock"),
                                               UnitAttr::get(ctx))}));
  mod.insertInput(numInputs + 1, rstName, IntegerType::get(ctx, 1));
  mod.setPortAttrs(
      numInputs + 1,
      DictionaryAttr::get(ctx, {NamedAttribute(StringAttr::get(ctx, "dc.reset"),
                                               UnitAttr::get(ctx))}));

  // We must initialize any port attributes that are not set otherwise the
  // verifier will fail.
  for (size_t portNum = 0, e = mod.getNumPorts(); portNum < e; ++portNum) {
    auto attrs = dyn_cast_or_null<DictionaryAttr>(mod.getPortAttrs(portNum));
    if (attrs)
      continue;
    mod.setPortAttrs(portNum, DictionaryAttr::get(ctx, {}));
  }
}

class HandshakeToDCPass
    : public circt::impl::HandshakeToDCBase<HandshakeToDCPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    auto patternBuilder = [&](TypeConverter &typeConverter,
                              handshaketodc::ConvertedOps &convertedOps,
                              RewritePatternSet &patterns) {
      patterns.add<FuncOpConversion>(mod.getContext(), typeConverter,
                                     &convertedOps);
      patterns.add<ReturnOpConversion>(typeConverter, mod.getContext());
    };

    LogicalResult res =
        runHandshakeToDC(mod, circt::HandshakeToDCOptions{clkName, rstName},
                         patternBuilder, nullptr);
    if (failed(res))
      signalPassFailure();
  }
};
} // namespace

LogicalResult circt::handshaketodc::runHandshakeToDC(
    mlir::Operation *op, circt::HandshakeToDCOptions options,
    llvm::function_ref<void(TypeConverter &typeConverter,
                            handshaketodc::ConvertedOps &convertedOps,
                            RewritePatternSet &patterns)>
        patternBuilder,
    llvm::function_ref<void(mlir::ConversionTarget &)> configureTarget) {
  // Maintain the set of operations which has been converted either through
  // unit rate conversion, or as part of other conversions.
  // Rationale:
  // This is needed for all of the arith ops that get created as part of the
  // handshake ops (e.g. arith.select for handshake.mux). There's a bit of a
  // dilemma here seeing as all operations need to be converted/touched in a
  // handshake.func - which is done so by UnitRateConversionPattern (when no
  // other pattern applies). However, we obviously don't want to run said
  // pattern on these newly created ops since they do not have handshake
  // semantics.
  handshaketodc::ConvertedOps convertedOps;
  mlir::MLIRContext *ctx = op->getContext();
  ConversionTarget target(*ctx);
  target.addIllegalDialect<handshake::HandshakeDialect>();
  target.addLegalDialect<dc::DCDialect>();
  target.addLegalOp<mlir::ModuleOp, handshake::ESIInstanceOp, hw::HWModuleOp,
                    hw::OutputOp>();

  // And any user-specified target adjustments
  if (configureTarget)
    configureTarget(target);

  // The various patterns will insert new operations into the module to
  // facilitate the conversion - however, these operations must be
  // distinguishable from already converted operations (which may be of the
  // same type as the newly inserted operations). To do this, we mark all
  // operations which have been converted as legal, and all other operations
  // as illegal.
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return convertedOps.contains(op) ||
           // Allow any ops which weren't in a `handshake.func` to pass through.
           !convertedOps.contains(op->getParentOfType<hw::HWModuleOp>());
  });

  DCTypeConverter typeConverter;
  RewritePatternSet patterns(ctx);

  // Add handshake conversion patterns.
  // Note: merge/control merge are not supported - these are non-deterministic
  // operators and we do not care for them.
  patterns
      .add<BufferOpConversion, CondBranchConversionPattern,
           SinkOpConversionPattern, SourceOpConversionPattern,
           MuxOpConversionPattern, ForkOpConversionPattern, JoinOpConversion,
           PackOpConversion, UnpackOpConversion, MergeOpConversion,
           ControlMergeOpConversion, ConstantOpConversion, SyncOpConversion>(
          ctx, typeConverter, &convertedOps);

  // ALL other single-result operations are converted via the
  // UnitRateConversionPattern.
  patterns.add<UnitRateConversionPattern>(ctx, typeConverter, &convertedOps);

  // Build any user-specified patterns
  patternBuilder(typeConverter, convertedOps, patterns);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  // Add clock and reset ports to each converted module.
  for (auto &op : convertedOps)
    if (auto mod = dyn_cast<hw::HWModuleOp>(op); mod)
      addClkRst(mod, options.clkName, options.rstName);

  // Run conversions which need see everything.
  HWSymbolCache symbolCache;
  symbolCache.addDefinitions(op);
  symbolCache.freeze();
  ConversionTarget globalLoweringTarget(*ctx);
  globalLoweringTarget.addIllegalDialect<handshake::HandshakeDialect>();
  globalLoweringTarget.addLegalDialect<dc::DCDialect, hw::HWDialect>();
  RewritePatternSet globalPatterns(ctx);
  globalPatterns.add<ESIInstanceConversionPattern>(ctx, symbolCache);
  if (failed(applyPartialConversion(op, globalLoweringTarget,
                                    std::move(globalPatterns))))
    return op->emitOpError() << "error during conversion";

  return success();
}
