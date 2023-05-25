//===- DCToHW.cpp - Translate DC into HW ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main DC to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DCToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ConversionPatterns.h"
#include "circt/Support/ValueMapper.h"
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
using namespace circt::dc;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

// NOLINTNEXTLINE(misc-no-recursion)
static Type tupleToStruct(TupleType tuple) {
  auto *ctx = tuple.getContext();
  mlir::SmallVector<hw::StructType::FieldInfo, 8> hwfields;
  for (auto [i, innerType] : llvm::enumerate(tuple)) {
    Type convertedInnerType = innerType;
    if (auto tupleInnerType = innerType.dyn_cast<TupleType>())
      convertedInnerType = tupleToStruct(tupleInnerType);
    hwfields.push_back({StringAttr::get(ctx, "field" + std::to_string(i)),
                        convertedInnerType});
  }

  return hw::StructType::get(ctx, hwfields);
}

// Converts the range of 'types' into a `hw`-dialect type. The range will be
// converted to a `hw.struct` type.
static Type toHWType(Type t);
static Type toHWType(TypeRange types) {
  if (types.size() == 1)
    return toHWType(types.front());
  return toHWType(mlir::TupleType::get(types[0].getContext(), types));
}

// Converts any type 't' into a `hw`-compatible type.
// tuple -> hw.struct
// none -> i0
// (tuple[...] | hw.struct)[...] -> (tuple | hw.struct)[toHwType(...)]
static Type toHWType(Type t) {
  return TypeSwitch<Type, Type>(t)
      .Case<TupleType>(
          [&](TupleType tt) { return toHWType(tupleToStruct(tt)); })
      .Case<hw::StructType>([&](auto st) {
        llvm::SmallVector<hw::StructType::FieldInfo> structFields(
            st.getElements());
        for (auto &field : structFields)
          field.type = toHWType(field.type);
        return hw::StructType::get(st.getContext(), structFields);
      })
      .Case<NoneType>(
          [&](NoneType nt) { return IntegerType::get(nt.getContext(), 0); })
      .Default([&](Type t) { return t; });
}

static Type toESIHWType(Type t) {
  auto *ctx = t.getContext();
  Type outType =
      llvm::TypeSwitch<Type, Type>(t)
          .Case<ValueType>([&](auto vt) {
            return esi::ChannelType::get(ctx, toHWType(vt.getInnerTypes()));
          })
          .Case<TokenType>([&](auto tt) {
            return esi::ChannelType::get(ctx,
                                         IntegerType::get(tt.getContext(), 0));
          })
          .Default([](auto t) { return toHWType(t); });

  return outType;
}

namespace {

// Shared state used by various functions; captured in a struct to reduce the
// number of arguments that we have to pass around.
struct DCLoweringState {
  ModuleOp parentModule;
  NameUniquer nameUniquer;
};

// A type converter is needed to perform the in-flight materialization of "raw"
// (non-ESI channel) types to their ESI channel correspondents. This comes into
// effect when backedges exist in the input IR.
class ESITypeConverter : public TypeConverter {
public:
  ESITypeConverter() {
    addConversion([](Type type) -> Type { return toESIHWType(type); });
    addConversion([](esi::ChannelType t) -> Type { return t; });

    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW Sub-module Related Functions
//===----------------------------------------------------------------------===//

namespace {

// Input handshakes contain a resolved valid and (optional )data signal, and
// a to-be-assigned ready signal.
struct InputHandshake {
  Value channel;
  Value valid;
  std::shared_ptr<Backedge> ready;
  Value data;
};

// Output handshakes contain a resolved ready, and to-be-assigned valid and
// (optional) data signals.
struct OutputHandshake {
  Value channel;
  std::shared_ptr<Backedge> valid;
  Value ready;
  std::shared_ptr<Backedge> data;
};

// Directly connect an input handshake to an output handshake
static void connect(InputHandshake &input, OutputHandshake &output) {
  output.valid->setValue(input.valid);
  input.ready->setValue(output.ready);
}

/// A helper struct that acts like a wire. Can be used to interact with the
/// RTLBuilder when multiple built components should be connected.
struct HandshakeWire {
  HandshakeWire(BackedgeBuilder &bb, Type dataType) {
    MLIRContext *ctx = dataType.getContext();
    auto i1Type = IntegerType::get(ctx, 1);
    valid = std::make_shared<Backedge>(bb.get(i1Type));
    ready = std::make_shared<Backedge>(bb.get(i1Type));
    data = std::make_shared<Backedge>(bb.get(dataType));
  }

  // Functions that allow to treat a wire like an input or output port.
  // **Careful**: Such a port will not be updated when backedges are resolved.
  InputHandshake getAsInput() {
    InputHandshake ih;
    ih.valid = *valid;
    ih.ready = ready;
    ih.data = *data;
    ih.channel = nullptr;
    return ih;
  }
  OutputHandshake getAsOutput() {
    OutputHandshake oh;
    oh.valid = valid;
    oh.ready = *ready;
    oh.data = data;
    oh.channel = nullptr;
    return oh;
  }

  std::shared_ptr<Backedge> valid;
  std::shared_ptr<Backedge> ready;
  std::shared_ptr<Backedge> data;
};

template <typename T, typename TInner>
llvm::SmallVector<T> extractValues(llvm::SmallVector<TInner> &container,
                                   llvm::function_ref<T(TInner &)> extractor) {
  llvm::SmallVector<T> result;
  llvm::transform(container, std::back_inserter(result), extractor);
  return result;
}
struct UnwrappedIO {
  llvm::SmallVector<InputHandshake> inputs;
  llvm::SmallVector<OutputHandshake> outputs;

  llvm::SmallVector<Value> getInputValids() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getInputReadys() {
    return extractValues<std::shared_ptr<Backedge>, InputHandshake>(
        inputs, [](auto &hs) { return hs.ready; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getOutputValids() {
    return extractValues<std::shared_ptr<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<Value> getInputDatas() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.data; });
  }
  llvm::SmallVector<Value> getOutputReadys() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.ready; });
  }

  llvm::SmallVector<Value> getOutputChannels() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.channel; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getOutputDatas() {
    return extractValues<std::shared_ptr<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.data; });
  }
};

// A class containing a bunch of syntactic sugar to reduce builder function
// verbosity.
// @todo: should be moved to support.
struct RTLBuilder {
  RTLBuilder(Location loc, OpBuilder &builder, Value clk = Value(),
             Value rst = Value())
      : b(builder), loc(loc), clk(clk), rst(rst) {}

  Value constant(const APInt &apv, std::optional<StringRef> name = {}) {
    // Cannot use zero-width APInt's in DenseMap's, see
    // https://github.com/llvm/llvm-project/issues/58013
    bool isZeroWidth = apv.getBitWidth() == 0;
    if (!isZeroWidth) {
      auto it = constants.find(apv);
      if (it != constants.end())
        return it->second;
    }

    auto cval = b.create<hw::ConstantOp>(loc, apv);
    if (!isZeroWidth)
      constants[apv] = cval;
    return cval;
  }

  Value constant(unsigned width, int64_t value,
                 std::optional<StringRef> name = {}) {
    return constant(APInt(width, value));
  }
  std::pair<Value, Value> wrap(Value data, Value valid,
                               std::optional<StringRef> name = {}) {
    auto wrapOp = b.create<esi::WrapValidReadyOp>(loc, data, valid);
    return {wrapOp.getResult(0), wrapOp.getResult(1)};
  }
  std::pair<Value, Value> unwrap(Value channel, Value ready,
                                 std::optional<StringRef> name = {}) {
    auto unwrapOp = b.create<esi::UnwrapValidReadyOp>(loc, channel, ready);
    return {unwrapOp.getResult(0), unwrapOp.getResult(1)};
  }

  // Various syntactic sugar functions.
  Value reg(StringRef name, Value in, Value rstValue, Value clk = Value(),
            Value rst = Value()) {
    Value resolvedClk = clk ? clk : this->clk;
    Value resolvedRst = rst ? rst : this->rst;
    assert(resolvedClk &&
           "No global clock provided to this RTLBuilder - a clock "
           "signal must be provided to the reg(...) function.");
    assert(resolvedRst &&
           "No global reset provided to this RTLBuilder - a reset "
           "signal must be provided to the reg(...) function.");

    return b.create<seq::CompRegOp>(loc, in.getType(), in, resolvedClk, name,
                                    resolvedRst, rstValue, StringAttr());
  }

  Value cmp(Value lhs, Value rhs, comb::ICmpPredicate predicate,
            std::optional<StringRef> name = {}) {
    return b.create<comb::ICmpOp>(loc, predicate, lhs, rhs);
  }

  Value buildNamedOp(llvm::function_ref<Value()> f,
                     std::optional<StringRef> name) {
    Value v = f();
    StringAttr nameAttr;
    Operation *op = v.getDefiningOp();
    if (name.has_value()) {
      op->setAttr("sv.namehint", b.getStringAttr(*name));
      nameAttr = b.getStringAttr(*name);
    }
    return v;
  }

  // Bitwise 'and'.
  Value bAnd(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::AndOp>(loc, values, false); }, name);
  }

  Value bOr(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::OrOp>(loc, values, false); }, name);
  }

  // Bitwise 'not'.
  Value bNot(Value value, std::optional<StringRef> name = {}) {
    auto allOnes = constant(value.getType().getIntOrFloatBitWidth(), -1);
    std::string inferedName;
    if (!name) {
      // Try to create a name from the input value.
      if (auto valueName =
              value.getDefiningOp()->getAttrOfType<StringAttr>("sv.namehint")) {
        inferedName = ("not_" + valueName.getValue()).str();
        name = inferedName;
      }
    }

    return buildNamedOp(
        [&]() { return b.create<comb::XorOp>(loc, value, allOnes); }, name);
  }

  Value shl(Value value, Value shift, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::ShlOp>(loc, value, shift); }, name);
  }

  Value concat(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp([&]() { return b.create<comb::ConcatOp>(loc, values); },
                        name);
  }

  // Packs a list of values into a hw.struct.
  Value pack(ValueRange values, Type structType = Type(),
             std::optional<StringRef> name = {}) {
    if (!structType)
      structType = toHWType(values.getTypes());

    return buildNamedOp(
        [&]() { return b.create<hw::StructCreateOp>(loc, structType, values); },
        name);
  }

  // Unpacks a hw.struct into a list of values.
  ValueRange unpack(Value value) {
    auto structType = value.getType().cast<hw::StructType>();
    llvm::SmallVector<Type> innerTypes;
    structType.getInnerTypes(innerTypes);
    return b.create<hw::StructExplodeOp>(loc, innerTypes, value).getResults();
  }

  llvm::SmallVector<Value> toBits(Value v, std::optional<StringRef> name = {}) {
    llvm::SmallVector<Value> bits;
    for (unsigned i = 0, e = v.getType().getIntOrFloatBitWidth(); i != e; ++i)
      bits.push_back(b.create<comb::ExtractOp>(loc, v, i, /*bitWidth=*/1));
    return bits;
  }

  // OR-reduction of the bits in 'v'.
  Value rOr(Value v, std::optional<StringRef> name = {}) {
    return buildNamedOp([&]() { return bOr(toBits(v)); }, name);
  }

  // Extract bits v[hi:lo] (inclusive).
  Value extract(Value v, unsigned lo, unsigned hi,
                std::optional<StringRef> name = {}) {
    unsigned width = hi - lo + 1;
    return buildNamedOp(
        [&]() { return b.create<comb::ExtractOp>(loc, v, lo, width); }, name);
  }

  // Truncates 'value' to its lower 'width' bits.
  Value truncate(Value value, unsigned width,
                 std::optional<StringRef> name = {}) {
    return extract(value, 0, width - 1, name);
  }

  Value zext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {}) {
    unsigned inWidth = value.getType().getIntOrFloatBitWidth();
    assert(inWidth <= outWidth && "zext: input width must be <- output width.");
    if (inWidth == outWidth)
      return value;
    auto c0 = constant(outWidth - inWidth, 0);
    return concat({c0, value}, name);
  }

  Value sext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {}) {
    return comb::createOrFoldSExt(loc, value, b.getIntegerType(outWidth), b);
  }

  // Extracts a single bit v[bit].
  Value bit(Value v, unsigned index, std::optional<StringRef> name = {}) {
    return extract(v, index, index, name);
  }

  // Creates a hw.array of the given values.
  Value arrayCreate(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayCreateOp>(loc, values); }, name);
  }

  // Extract the 'index'th value from the input array.
  Value arrayGet(Value array, Value index, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayGetOp>(loc, array, index); }, name);
  }

  // Muxes a range of values.
  // The select signal is expected to be a decimal value which selects starting
  // from the lowest index of value.
  Value mux(Value index, ValueRange values,
            std::optional<StringRef> name = {}) {
    if (values.size() == 2)
      return b.create<comb::MuxOp>(loc, index, values[1], values[0]);

    return arrayGet(arrayCreate(values), index, name);
  }

  // Muxes a range of values. The select signal is expected to be a 1-hot
  // encoded value.
  Value ohMux(Value index, ValueRange inputs) {
    // Confirm the select input can be a one-hot encoding for the inputs.
    unsigned numInputs = inputs.size();
    assert(numInputs == index.getType().getIntOrFloatBitWidth() &&
           "one-hot select can't mux inputs");

    // Start the mux tree with zero value.
    // Todo: clean up when handshake supports i0.
    auto dataType = inputs[0].getType();
    unsigned width =
        dataType.isa<NoneType>() ? 0 : dataType.getIntOrFloatBitWidth();
    Value muxValue = constant(width, 0);

    // Iteratively chain together muxes from the high bit to the low bit.
    for (size_t i = numInputs - 1; i != 0; --i) {
      Value input = inputs[i];
      Value selectBit = bit(index, i);
      muxValue = mux(selectBit, {muxValue, input});
    }

    return muxValue;
  }

  OpBuilder &b;
  Location loc;
  Value clk, rst;
  DenseMap<APInt, Value> constants;
};

/// Creates a Value that has an assigned zero value. For structs, this
/// corresponds to assigning zero to each element recursively.
static Value createZeroDataConst(RTLBuilder &s, Location loc, Type type) {
  return TypeSwitch<Type, Value>(type)
      .Case<NoneType>([&](NoneType) { return s.constant(0, 0); })
      .Case<IntType, IntegerType>([&](auto type) {
        return s.constant(type.getIntOrFloatBitWidth(), 0);
      })
      .Case<hw::StructType>([&](auto structType) {
        SmallVector<Value> zeroValues;
        for (auto field : structType.getElements())
          zeroValues.push_back(createZeroDataConst(s, loc, field.type));
        return s.b.create<hw::StructCreateOp>(loc, structType, zeroValues);
      })
      .Default([&](Type) -> Value {
        emitError(loc) << "unsupported type for zero value: " << type;
        assert(false);
        return {};
      });
}

static bool isZeroWidthType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.getWidth() == 0;
  if (type.isa<NoneType>())
    return true;

  return false;
}

static UnwrappedIO unwrapIO(Location loc, ValueRange operands,
                            TypeRange results,
                            ConversionPatternRewriter &rewriter,
                            BackedgeBuilder &bb) {
  RTLBuilder rtlb(loc, rewriter);
  UnwrappedIO unwrapped;
  for (auto in : operands) {
    assert(isa<esi::ChannelType>(in.getType()));
    InputHandshake hs;
    auto ready = std::make_shared<Backedge>(bb.get(rtlb.b.getI1Type()));
    auto [data, valid] = rtlb.unwrap(in, *ready);
    hs.valid = valid;
    hs.ready = ready;
    hs.data = data;
    hs.channel = in;
    unwrapped.inputs.push_back(hs);
  }
  for (auto outputType : results) {
    outputType = toESIHWType(outputType);
    esi::ChannelType channelType = cast<esi::ChannelType>(outputType);
    OutputHandshake hs;
    Type innerType = channelType.getInner();
    Value data;
    if (isZeroWidthType(innerType)) {
      // Feed the ESI wrap with an i0 constant.
      data =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getIntegerType(0), 0);
    } else {
      // Create a backedge for the unresolved data.
      auto dataBackedge = std::make_shared<Backedge>(bb.get(innerType));
      hs.data = dataBackedge;
      data = *dataBackedge;
    }
    auto valid = std::make_shared<Backedge>(bb.get(rewriter.getI1Type()));
    auto [dataCh, ready] = rtlb.wrap(data, *valid);
    hs.valid = valid;
    hs.ready = ready;
    hs.channel = dataCh;
    unwrapped.outputs.push_back(hs);
  }
  return unwrapped;
}

static UnwrappedIO unwrapIO(Operation *op, ValueRange operands,
                            ConversionPatternRewriter &rewriter,
                            BackedgeBuilder &bb) {
  return unwrapIO(op->getLoc(), operands, op->getResultTypes(), rewriter, bb);
}

// Locate the clock and reset values from the parent operation based on
// attributes assigned to the arguments.
static FailureOr<std::pair<Value, Value>> getClockAndReset(Operation *op) {
  auto *parent = op->getParentOp();
  mlir::FunctionOpInterface parentFuncOp =
      dyn_cast<mlir::FunctionOpInterface>(parent);
  if (!parent)
    return parent->emitOpError(
        "parent op does not implement FunctionOpInterface");

  SmallVector<DictionaryAttr> argAttrs;
  parentFuncOp.getAllArgAttrs(argAttrs);

  std::optional<size_t> clockIdx, resetIdx;

  for (auto [idx, attrs] : llvm::enumerate(argAttrs)) {
    if (attrs.get("dc.clock")) {
      if (clockIdx)
        return parent->emitOpError(
            "multiple arguments contains a 'dc.clock' attribute");
      clockIdx = idx;
    }

    if (attrs.get("dc.reset")) {
      if (resetIdx)
        return parent->emitOpError(
            "multiple arguments contains a 'dc.reset' attribute");
      resetIdx = idx;
    }
  }

  if (!clockIdx)
    return parent->emitOpError("no argument contains a 'dc.clock' attribute");

  if (!resetIdx)
    return parent->emitOpError("no argument contains a 'dc.reset' attribute");

  return std::pair<Value, Value>{parentFuncOp.getArgument(*clockIdx),
                                 parentFuncOp.getArgument(*resetIdx)};
}

class ForkConversionPattern : public OpConversionPattern<ForkOp> {
public:
  using OpConversionPattern<ForkOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ForkOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto crRes = getClockAndReset(op);
    if (failed(crRes))
      return failure();
    auto [clock, reset] = *crRes;
    auto rtlb = RTLBuilder(op.getLoc(), rewriter, clock, reset);
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);

    auto &input = io.inputs[0];

    auto c0I1 = rtlb.constant(1, 0);
    llvm::SmallVector<Value> doneWires;
    for (auto [i, output] : llvm::enumerate(io.outputs)) {
      auto doneBE = bb.get(rtlb.b.getI1Type());
      auto emitted = rtlb.bAnd({doneBE, rtlb.bNot(*input.ready)});
      auto emittedReg = rtlb.reg("emitted_" + std::to_string(i), emitted, c0I1);
      auto outValid = rtlb.bAnd({rtlb.bNot(emittedReg), input.valid});
      output.valid->setValue(outValid);
      auto validReady = rtlb.bAnd({output.ready, outValid});
      auto done =
          rtlb.bOr({validReady, emittedReg}, "done" + std::to_string(i));
      doneBE.setValue(done);
      doneWires.push_back(done);
    }
    input.ready->setValue(rtlb.bAnd(doneWires, "allDone"));

    rewriter.replaceOp(op, io.getOutputChannels());
    return success();
  }
};

class JoinConversionPattern : public OpConversionPattern<JoinOp> {
public:
  using OpConversionPattern<JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto &output = io.outputs[0];

    Value allValid = rtlb.bAnd(io.getInputValids());
    output.valid->setValue(allValid);

    auto validAndReady = rtlb.bAnd({output.ready, allValid});
    for (auto &input : io.inputs)
      input.ready->setValue(validAndReady);

    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class SelectConversionPattern : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SelectOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);

    // Extract select signal from the unwrapped IO.
    auto select = io.inputs[0];
    io.inputs.erase(io.inputs.begin());
    buildMuxLogic(rtlb, io, select);

    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }

  // Builds mux logic for the given inputs and outputs.
  // Note: it is assumed that the caller has removed the 'select' signal from
  // the 'unwrapped' inputs and provide it as a separate argument.
  void buildMuxLogic(RTLBuilder &s, UnwrappedIO &unwrapped,
                     InputHandshake &select) const {

    // ============================= Control logic =============================
    size_t numInputs = unwrapped.inputs.size();
    size_t selectWidth = llvm::Log2_64_Ceil(numInputs);
    Value truncatedSelect =
        select.data.getType().getIntOrFloatBitWidth() > selectWidth
            ? s.truncate(select.data, selectWidth)
            : select.data;

    // Decimal-to-1-hot decoder. 'shl' operands must be identical in size.
    auto selectZext = s.zext(truncatedSelect, numInputs);
    auto select1h = s.shl(s.constant(numInputs, 1), selectZext);
    auto &res = unwrapped.outputs[0];

    // Mux input valid signals.
    auto selectedInputValid =
        s.mux(truncatedSelect, unwrapped.getInputValids());
    // Result is valid when the selected input and the select input is valid.
    auto selAndInputValid = s.bAnd({selectedInputValid, select.valid});
    res.valid->setValue(selAndInputValid);
    auto resValidAndReady = s.bAnd({selAndInputValid, res.ready});

    // Select is ready when result is valid and ready (result transacting).
    select.ready->setValue(resValidAndReady);

    // Assign each input ready signal if it is currently selected.
    for (auto [inIdx, in] : llvm::enumerate(unwrapped.inputs)) {
      // Extract the selection bit for this input.
      auto isSelected = s.bit(select1h, inIdx);

      // '&' that with the result valid and ready, and assign to the input
      // ready signal.
      auto activeAndResultValidAndReady =
          s.bAnd({isSelected, resValidAndReady});
      in.ready->setValue(activeAndResultValidAndReady);
    }
  }
};

class BranchConversionPattern : public OpConversionPattern<BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BranchOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto cond = io.inputs[0];
    auto trueRes = io.outputs[0];
    auto falseRes = io.outputs[1];

    // Connect valid signal of both results.
    trueRes.valid->setValue(rtlb.bAnd({cond.data, cond.valid}));
    falseRes.valid->setValue(rtlb.bAnd({rtlb.bNot(cond.data), cond.valid}));

    // Connect ready signal of condition.
    auto selectedResultReady =
        rtlb.mux(cond.data, {falseRes.ready, trueRes.ready});
    auto condReady = rtlb.bAnd({selectedResultReady, cond.valid});
    cond.ready->setValue(condReady);

    rewriter.replaceOp(
        op, llvm::SmallVector<Value>{trueRes.channel, falseRes.channel});
    return success();
  }
};

class SinkConversionPattern : public OpConversionPattern<SinkOp> {
public:
  using OpConversionPattern<SinkOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    io.inputs[0].ready->setValue(
        RTLBuilder(op.getLoc(), rewriter).constant(1, 1));
    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class SourceConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    io.outputs[0].valid->setValue(rtlb.constant(1, 1));
    io.outputs[0].data->setValue(rtlb.constant(0, 0));
    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class PackConversionPattern : public OpConversionPattern<PackOp> {
public:
  using OpConversionPattern<PackOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PackOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, llvm::SmallVector<Value>{operands.getToken()},
                       rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto &input = io.inputs[0];
    auto &output = io.outputs[0];

    Value packedData;
    if (operands.getInputs().size() > 1)
      packedData = rtlb.pack(operands.getInputs());
    else
      packedData = operands.getInputs()[0];

    output.data->setValue(packedData);
    connect(input, output);
    rewriter.replaceOp(op, output.channel);
    return success();
  }
};

class UnpackConversionPattern : public OpConversionPattern<UnpackOp> {
public:
  using OpConversionPattern<UnpackOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(
        op.getLoc(), llvm::SmallVector<Value>{operands.getInput()},
        // Only generate an output channel for the token typed output.
        llvm::SmallVector<Type>{op.getToken().getType()}, rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto &input = io.inputs[0];
    auto &output = io.outputs[0];

    llvm::SmallVector<Value> unpackedValues;
    if (op.getInput().getType().cast<ValueType>().getInnerTypes().size() != 1)
      unpackedValues = rtlb.unpack(input.data);
    else
      unpackedValues.push_back(input.data);

    connect(input, output);
    llvm::SmallVector<Value> outputs;
    outputs.push_back(output.channel);
    outputs.append(unpackedValues.begin(), unpackedValues.end());
    rewriter.replaceOp(op, outputs);
    return success();
  }
};

class BufferConversionPattern : public OpConversionPattern<BufferOp> {
public:
  using OpConversionPattern<BufferOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto crRes = getClockAndReset(op);
    if (failed(crRes))
      return failure();
    auto [clock, reset] = *crRes;

    // ... esi.buffer should in theory provide a correct (latency-insensitive)
    // implementation...
    Type channelType = operands.getInput().getType();
    rewriter.replaceOpWithNewOp<esi::ChannelBufferOp>(
        op, channelType, clock, reset, operands.getInput(), op.getSizeAttr(),
        nullptr);
    return success();
  };
};

} // namespace

static bool isDCType(Type type) { return type.isa<TokenType, ValueType>(); }

// Returns true if the given `op` is considered as legal - i.e. it does not
// contain any dc-typed values.
static bool isLegalOp(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::none_of(funcOp.getArgumentTypes(), isDCType) &&
           llvm::none_of(funcOp.getResultTypes(), isDCType) &&
           llvm::none_of(funcOp.getFunctionBody().getArgumentTypes(), isDCType);
  }

  bool operandsOK = llvm::none_of(op->getOperandTypes(), isDCType);
  bool resultsOK = llvm::none_of(op->getResultTypes(), isDCType);
  return operandsOK && resultsOK;
}

//===----------------------------------------------------------------------===//
// HW Top-module Related Functions
//===----------------------------------------------------------------------===//

namespace {
class DCToHWPass : public DCToHWBase<DCToHWPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Lowering to HW requires that every DC-typed value is used exactly once.
    // Check whether this precondition is met, and if not, exit.
    auto walkRes = mod.walk([&](Operation *op) {
      for (auto res : op->getResults()) {
        if (res.getType().isa<dc::TokenType, dc::ValueType>()) {
          if (res.use_empty()) {
            op->emitOpError() << "DCToHW: value " << res << " is unused.";
            return WalkResult::interrupt();
          }
          if (!res.hasOneUse()) {
            op->emitOpError()
                << "DCToHW: value " << res << " has multiple uses.";
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted()) {
      mod->emitOpError()
          << "DCToHW: failed to verify that all values "
             "are used exactly once. Remember to run the "
             "fork/sink materialization pass before HW lowering.";
      signalPassFailure();
      return;
    }

    ESITypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(isLegalOp);

    // All top-level logic of a handshake module will be the interconnectivity
    // between instantiated modules.
    target.addIllegalDialect<dc::DCDialect>();

    RewritePatternSet patterns(mod.getContext());

    patterns.insert<ForkConversionPattern, JoinConversionPattern,
                    SelectConversionPattern, BranchConversionPattern,
                    PackConversionPattern, UnpackConversionPattern,
                    BufferConversionPattern, SourceConversionPattern,
                    SinkConversionPattern, TypeConversionPattern>(
        typeConverter, mod.getContext());

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createDCToHWPass() {
  return std::make_unique<DCToHWPass>();
}
