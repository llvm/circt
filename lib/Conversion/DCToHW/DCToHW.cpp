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
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/SingleUse.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace circt::dc;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

static Type esiWrap(Type type) {
  return esi::ChannelType::get(type.getContext(), type);
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
    addConversion([](Type type) -> Type { return esiWrap(type); });
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

/// Returns a submodule name resulting from an operation, without discriminating
/// type information.
static std::string getBareSubModuleName(Operation *oldOp) {
  // The dialect name is separated from the operation name by '.', which is not
  // valid in SystemVerilog module names. In case this name is used in
  // SystemVerilog output, replace '.' with '_'.
  std::string subModuleName = oldOp->getName().getStringRef().str();
  std::replace(subModuleName.begin(), subModuleName.end(), '.', '_');
  return subModuleName;
}

static std::string getCallName(Operation *op) {
  auto callOp = dyn_cast<handshake::InstanceOp>(op);
  return callOp ? callOp.getModule().str() : getBareSubModuleName(op);
}

/// Extracts the type of the data-carrying type of opType. If opType is an ESI
/// channel, getHandshakeBundleDataType extracts the data-carrying type, else,
/// assume that opType itself is the data-carrying type.
static Type getOperandDataType(Value op) {
  auto opType = op.getType();
  if (auto channelType = opType.dyn_cast<esi::ChannelType>())
    return channelType.getInner();
  return opType;
}

/// Filters NoneType's from the input.
static SmallVector<Type> filterNoneTypes(ArrayRef<Type> input) {
  SmallVector<Type> filterRes;
  llvm::copy_if(input, std::back_inserter(filterRes),
                [](Type type) { return !type.isa<NoneType>(); });
  return filterRes;
}

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
  Value data;
};

/// A helper struct that acts like a wire. Can be used to interact with the
/// RTLBuilder when multiple built components should be connected.
struct HandshakeWire {
  HandshakeWire(BackedgeBuilder &bb) {
    MLIRContext *ctx = dataType.getContext();
    auto i1Type = IntegerType::get(ctx, 1);
    valid = std::make_shared<Backedge>(bb.get(i1Type));
    ready = std::make_shared<Backedge>(bb.get(i1Type));
  }

  // Functions that allow to treat a wire like an input or output port.
  // **Careful**: Such a port will not be updated when backedges are resolved.
  InputHandshake getAsInput() { return {*valid, ready}; }
  OutputHandshake getAsOutput() { return {valid, *ready}; }

  std::shared_ptr<Backedge> valid;
  std::shared_ptr<Backedge> ready;
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
  llvm::SmallVector<Value> getOutputReadys() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.ready; });
  }
};

// A class containing a bunch of syntactic sugar to reduce builder function
// verbosity.
// @todo: should be moved to support.
struct RTLBuilder {
  RTLBuilder(hw::ModulePortInfo info, OpBuilder &builder, Location loc,
             Value clk = Value(), Value rst = Value())
      : info(std::move(info)), b(builder), loc(loc), clk(clk), rst(rst) {}

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

    return b.createOrFold<comb::XorOp>(loc, value, allOnes, false);
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
      structType = tupleToStruct(values.getTypes());
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

  hw::ModulePortInfo info;
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

static void
addSequentialIOOperandsIfNeeded(Operation *op,
                                llvm::SmallVectorImpl<Value> &operands) {
  if (op->hasTrait<mlir::OpTrait::HasClock>()) {
    // Parent should at this point be a hw.module and have clock and reset
    // ports.
    auto parent = cast<hw::HWModuleOp>(op->getParentOp());
    operands.push_back(parent.getArgument(parent.getNumArguments() - 2));
    operands.push_back(parent.getArgument(parent.getNumArguments() - 1));
  }
}

template <typename T>
class DCConversionPattern : public OpConversionPattern<T> {
public:
  DCConversionPattern(ESITypeConverter &typeConverter, MLIRContext *context,
                      OpBuilder &submoduleBuilder, DCLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, context),
        submoduleBuilder(submoduleBuilder), ls(ls) {}

  using OpAdaptor = typename T::Adaptor;

  void setAllReadyWithCond(RTLBuilder &s, ArrayRef<InputHandshake> inputs,
                           OutputHandshake &output, Value cond) const {
    auto validAndReady = s.bAnd({output.ready, cond});
    for (auto &input : inputs)
      input.ready->setValue(validAndReady);
  }

  void buildJoinLogic(RTLBuilder &s, ArrayRef<InputHandshake> inputs,
                      OutputHandshake &output) const {
    llvm::SmallVector<Value> valids;
    for (auto &input : inputs)
      valids.push_back(input.valid);
    Value allValid = s.bAnd(valids);
    output.valid->setValue(allValid);
    setAllReadyWithCond(s, inputs, output, allValid);
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

    // ============================== Data logic ===============================
    res.data->setValue(s.mux(truncatedSelect, unwrapped.getInputDatas()));
  }

  // Builds fork logic between the single input and multiple outputs' control
  // networks. Caller is expected to handle data separately.
  void buildForkLogic(RTLBuilder &s, BackedgeBuilder &bb, InputHandshake &input,
                      ArrayRef<OutputHandshake> outputs) const {
    auto c0I1 = s.constant(1, 0);
    llvm::SmallVector<Value> doneWires;
    for (auto [i, output] : llvm::enumerate(outputs)) {
      auto doneBE = bb.get(s.b.getI1Type());
      auto emitted = s.bAnd({doneBE, s.bNot(*input.ready)});
      auto emittedReg = s.reg("emitted_" + std::to_string(i), emitted, c0I1);
      auto outValid = s.bAnd({s.bNot(emittedReg), input.valid});
      output.valid->setValue(outValid);
      auto validReady = s.bAnd({output.ready, outValid});
      auto done = s.bOr({validReady, emittedReg}, "done" + std::to_string(i));
      doneBE.setValue(done);
      doneWires.push_back(done);
    }
    input.ready->setValue(s.bAnd(doneWires, "allDone"));
  }

  // Builds a unit-rate actor around an inner operation. 'unitBuilder' is a
  // function which takes the set of unwrapped data inputs, and returns a
  // value which should be assigned to the output data value.
  void buildUnitRateJoinLogic(
      RTLBuilder &s, UnwrappedIO &unwrappedIO,
      llvm::function_ref<Value(ValueRange)> unitBuilder) const {
    assert(unwrappedIO.outputs.size() == 1 &&
           "Expected exactly one output for unit-rate join actor");
    // Control logic.
    this->buildJoinLogic(s, unwrappedIO.inputs, unwrappedIO.outputs[0]);

    // Data logic.
    auto unitRes = unitBuilder(unwrappedIO.getInputDatas());
    unwrappedIO.outputs[0].data->setValue(unitRes);
  }

  void buildUnitRateForkLogic(
      RTLBuilder &s, BackedgeBuilder &bb, UnwrappedIO &unwrappedIO,
      llvm::function_ref<llvm::SmallVector<Value>(Value)> unitBuilder) const {
    assert(unwrappedIO.inputs.size() == 1 &&
           "Expected exactly one input for unit-rate fork actor");
    // Control logic.
    this->buildForkLogic(s, bb, unwrappedIO.inputs[0], unwrappedIO.outputs);

    // Data logic.
    auto unitResults = unitBuilder(unwrappedIO.inputs[0].data);
    assert(unitResults.size() == unwrappedIO.outputs.size() &&
           "Expected unit builder to return one result per output");
    for (auto [res, outport] : llvm::zip(unitResults, unwrappedIO.outputs))
      outport.data->setValue(res);
  }

  void buildExtendLogic(RTLBuilder &s, UnwrappedIO &unwrappedIO,
                        bool signExtend) const {
    size_t outWidth =
        toValidType(static_cast<Value>(*unwrappedIO.outputs[0].data).getType())
            .getIntOrFloatBitWidth();
    buildUnitRateJoinLogic(s, unwrappedIO, [&](ValueRange inputs) {
      if (signExtend)
        return s.sext(inputs[0], outWidth);
      return s.zext(inputs[0], outWidth);
    });
  }

  void buildTruncateLogic(RTLBuilder &s, UnwrappedIO &unwrappedIO,
                          unsigned targetWidth) const {
    size_t outWidth =
        toValidType(static_cast<Value>(*unwrappedIO.outputs[0].data).getType())
            .getIntOrFloatBitWidth();
    buildUnitRateJoinLogic(s, unwrappedIO, [&](ValueRange inputs) {
      return s.truncate(inputs[0], outWidth);
    });
  }

  /// Return the number of bits needed to index the given number of values.
  static size_t getNumIndexBits(uint64_t numValues) {
    return numValues > 1 ? llvm::Log2_64_Ceil(numValues) : 1;
  }

  Value buildPriorityArbiter(RTLBuilder &s, ArrayRef<Value> inputs,
                             Value defaultValue,
                             DenseMap<size_t, Value> &indexMapping) const {
    auto numInputs = inputs.size();
    auto priorityArb = defaultValue;

    for (size_t i = numInputs; i > 0; --i) {
      size_t inputIndex = i - 1;
      size_t oneHotIndex = size_t{1} << inputIndex;
      auto constIndex = s.constant(numInputs, oneHotIndex);
      indexMapping[inputIndex] = constIndex;
      priorityArb = s.mux(inputs[inputIndex], {priorityArb, constIndex});
    }
    return priorityArb;
  }

private:
  OpBuilder &submoduleBuilder;
  DCLoweringState &ls;
};

class ForkConversionPattern : public DCConversionPattern<ForkOp> {
public:
  using DCConversionPattern<ForkOp>::DCConversionPattern;
  void buildModule(ForkOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrapped = unwrapIO(s, bb, ports);
    buildUnitRateForkLogic(s, bb, unwrapped, [&](Value input) {
      return llvm::SmallVector<Value>(unwrapped.outputs.size(), input);
    });
  }
};

static UnwrappedIO unwrapIO(Operation *op, ValueRange operands,
                            ConversionPatternRewriter &rewriter,
                            BackedgeBuilder &bb) {
  RTLBuilder rtlb(rewriter);
  UnwrappedIO unwrapped;
  for (auto in : operands) {
    assert(isa<esi::ChannelType>(in.getType()));
    InputHandshake hs;
    auto ready = std::make_shared<Backedge>(bb.get(rtlb.b.getI1Type()));
    auto [data, valid] = rtlb.unwrap(in, *ready);
    hs.valid = valid;
    hs.ready = ready;
    hs.channel = in;
    unwrapped.inputs.push_back(hs);
  }
  for (auto outputType : op->getResults()) {
    esi::ChannelType channelType = dyn_cast<esi::ChannelType>(outputType);
    assert(channelType);
    OutputHandshake hs;
    Type innerType = channelType.getInner();
    auto data = std::make_shared<Backedge>(bb.get(innerType));
    auto valid = std::make_shared<Backedge>(bb.get(rewriter.getI1Type()));
    auto [dataCh, ready] = rtlb.wrap(*data, *valid);
    hs.valid = valid;
    hs.ready = ready;
    hs.channel = dataCh;
    unwrapped.outputs.push_back(hs);
  }
  return unwrapped;
}

class JoinConversionPattern : public DCConversionPattern<JoinOp> {
public:
  using DCConversionPattern<JoinOp>::DCConversionPattern;

  LogicalResult
  matchAndRewrite(JoinOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter);
    auto io = unwrapIO(op, operands, rewriter, bb);

    Value allValid = s.bAnd(io.getInputValids());
    output.valid->setValue(allValid);
    setAllReadyWithCond(s, io.inputs, output, allValid);
    rewriter.replaceOpWith(op, io.outputs[0].channel);
    return success();
  }
};

class MergeConversionPattern : public DCConversionPattern<MergeOp> {
public:
  using DCConversionPattern<MergeOp>::DCConversionPattern;

  LogicalResult
  matchAndRewrite(MergeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter);
    auto io = unwrapIO(op, operands, rewriter, bb);

    // Extract select signal from the unwrapped IO.
    auto select = unwrappedIO.inputs[0];
    io.inputs.erase(unwrappedIO.inputs.begin());
    buildMuxLogic(rewriter, unwrappedIO, select);

    rewriter.replaceOpWith(op, io.outputs[0].channel);
    return success();
  }

  // Builds mux logic for the given inputs and outputs.
  // Note: it is assumed that the caller has removed the 'select' signal from
  // the 'unwrapped' inputs and provide it as a separate argument.
  void buildMuxLogic(OpBuilder &b, UnwrappedIO &unwrapped,
                     InputHandshake &select) const {
    auto s = RTLBuilder(b);

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

    // ============================== Data logic ===============================
    res.data->setValue(s.mux(truncatedSelect, unwrapped.getInputDatas()));
  }
};

class ReturnConversionPattern : public OpConversionPattern<dc::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(dc::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Locate existing output op, Append operands to output op, and move to
    // the end of the block.
    auto parent = cast<hw::HWModuleOp>(op->getParentOp());
    auto outputOp = *parent.getBodyBlock()->getOps<hw::OutputOp>().begin();
    outputOp->setOperands(adaptor.getOperands());
    outputOp->moveAfter(&parent.getBodyBlock()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

class BranchConversionPattern : public DCConversionPattern<BranchOp> {
public:
  LogicalResult
  matchAndRewrite(BranchOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto unwrappedIO = unwrapIO(op, operands, rewriter, bb);
    auto cond = unwrappedIO.inputs[0];
    auto arg = unwrappedIO.inputs[1];
    auto trueRes = unwrappedIO.outputs[0];
    auto falseRes = unwrappedIO.outputs[1];

    auto condArgValid = s.bAnd({cond.valid, arg.valid});

    // Connect valid signal of both results.
    trueRes.valid->setValue(s.bAnd({cond.data, condArgValid}));
    falseRes.valid->setValue(s.bAnd({s.bNot(cond.data), condArgValid}));

    // Connecte data signals of both results.
    trueRes.data->setValue(arg.data);
    falseRes.data->setValue(arg.data);

    // Connect ready signal of input and condition.
    auto selectedResultReady =
        s.mux(cond.data, {falseRes.ready, trueRes.ready});
    auto condArgReady = s.bAnd({selectedResultReady, condArgValid});
    arg.ready->setValue(condArgReady);
    cond.ready->setValue(condArgReady);

    rewriter.replaceOpWith(
        op, llvm::SmallVector<Value>{trueRes.channel, falseRes.channel});
    return success();
  }
};

class SinkConversionPattern : public DCConversionPattern<SinkOp> {
public:
  using DCConversionPattern<SinkOp>::DCConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter);
    auto io = unwrapIO(op, operands, rewriter, bb);
    io.inputs[0].ready->setValue(RTLBuilder(rewriter).constant(1, 1));
    rewriter.replaceOpWith(op, io.outputs[0].channel);
    return success();
  }
};

class SourceConversionPattern : public DCConversionPattern<SourceOp> {
public:
  using DCConversionPattern<SourceOp>::DCConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter);
    auto io = unwrapIO(op, operands, rewriter, bb);
    auto rtlb = RTLBuilder(rewriter);
    io.outputs[0].valid->setValue(rtlb.constant(1, 1));
    io.outputs[0].data->setValue(rtlb.constant(0, 0));
    rewriter.replaceOpWith(op, io.outputs[0].channel);
    return success();
  }
};

class BufferConversionPattern : public DCConversionPattern<BufferOp> {
public:
  using DCConversionPattern<BufferOp>::DCConversionPattern;

  LogicalResult
  matchAndRewrite(BufferOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter);
    auto io = unwrapIO(op, operands, rewriter, bb);
    auto input = io.inputs[0];
    auto output = io.outputs[0];
    InputHandshake lastStage;
    SmallVector<int64_t> initValues;

    // For now, always build seq buffers.
    if (op.getInitValues())
      initValues = op.getInitValueArray();

    lastStage =
        buildSeqBufferLogic(s, bb, toValidType(op.getDataType()),
                            op.getNumSlots(), input, output, initValues);

    // Connect the last stage to the output handshake.
    output.data->setValue(lastStage.data);
    output.valid->setValue(lastStage.valid);
    lastStage.ready->setValue(output.ready);

    rewriter.replaceOpWith(op, output.channel);
    return success();
  };

  struct SeqBufferStage {
    SeqBufferStage(Type dataType, InputHandshake &preStage, BackedgeBuilder &bb,
                   RTLBuilder &s, size_t index,
                   std::optional<int64_t> initValue)
        : dataType(dataType), preStage(preStage), s(s), bb(bb), index(index) {

      // Todo: Change when i0 support is added.
      c0s = createZeroDataConst(s, s.loc, dataType);
      currentStage.ready = std::make_shared<Backedge>(bb.get(s.b.getI1Type()));

      auto hasInitValue = s.constant(1, initValue.has_value());
      auto validBE = bb.get(s.b.getI1Type());
      auto validReg = s.reg(getRegName("valid"), validBE, hasInitValue);
      auto readyBE = bb.get(s.b.getI1Type());

      Value initValueCs = c0s;
      if (initValue.has_value())
        initValueCs = s.constant(dataType.getIntOrFloatBitWidth(), *initValue);

      // This could/should be revised but needs a larger rethinking to avoid
      // introducing new bugs. Implement similarly to HandshakeToFIRRTL.
      Value dataReg =
          buildDataBufferLogic(validReg, initValueCs, validBE, readyBE);
      buildControlBufferLogic(validReg, readyBE, dataReg);
    }

    StringAttr getRegName(StringRef name) {
      return s.b.getStringAttr(name + std::to_string(index) + "_reg");
    }

    void buildControlBufferLogic(Value validReg, Backedge &readyBE,
                                 Value dataReg) {
      auto c0I1 = s.constant(1, 0);
      auto readyRegWire = bb.get(s.b.getI1Type());
      auto readyReg = s.reg(getRegName("ready"), readyRegWire, c0I1);

      // Create the logic to drive the current stage valid and potentially
      // data.
      currentStage.valid = s.mux(readyReg, {validReg, readyReg},
                                 "controlValid" + std::to_string(index));

      // Create the logic to drive the current stage ready.
      auto notReadyReg = s.bNot(readyReg);
      readyBE.setValue(notReadyReg);

      auto succNotReady = s.bNot(*currentStage.ready);
      auto neitherReady = s.bAnd({succNotReady, notReadyReg});
      auto ctrlNotReady = s.mux(neitherReady, {readyReg, validReg});
      auto bothReady = s.bAnd({*currentStage.ready, readyReg});

      // Create a mux for emptying the register when both are ready.
      auto resetSignal = s.mux(bothReady, {ctrlNotReady, c0I1});
      readyRegWire.setValue(resetSignal);

      // Add same logic for the data path if necessary.
      auto ctrlDataRegBE = bb.get(dataType);
      auto ctrlDataReg = s.reg(getRegName("ctrl_data"), ctrlDataRegBE, c0s);
      auto dataResult = s.mux(readyReg, {dataReg, ctrlDataReg});
      currentStage.data = dataResult;

      auto dataNotReadyMux = s.mux(neitherReady, {ctrlDataReg, dataReg});
      auto dataResetSignal = s.mux(bothReady, {dataNotReadyMux, c0s});
      ctrlDataRegBE.setValue(dataResetSignal);
    }

    Value buildDataBufferLogic(Value validReg, Value initValue,
                               Backedge &validBE, Backedge &readyBE) {
      // Create a signal for when the valid register is empty or the successor
      // is ready to accept new token.
      auto notValidReg = s.bNot(validReg);
      auto emptyOrReady = s.bOr({notValidReg, readyBE});
      preStage.ready->setValue(emptyOrReady);

      // Create a mux that drives the register input. If the emptyOrReady
      // signal is asserted, the mux selects the predValid signal. Otherwise,
      // it selects the register output, keeping the output registered
      // unchanged.
      auto validRegMux = s.mux(emptyOrReady, {validReg, preStage.valid});

      // Now we can drive the valid register.
      validBE.setValue(validRegMux);

      // Create a mux that drives the date register.
      auto dataRegBE = bb.get(dataType);
      auto dataReg =
          s.reg(getRegName("data"),
                s.mux(emptyOrReady, {dataRegBE, preStage.data}), initValue);
      dataRegBE.setValue(dataReg);
      return dataReg;
    }

    InputHandshake getOutput() { return currentStage; }

    Type dataType;
    InputHandshake &preStage;
    InputHandshake currentStage;
    RTLBuilder &s;
    BackedgeBuilder &bb;
    size_t index;

    // A zero-valued constant of equal type as the data type of this buffer.
    Value c0s;
  };

  InputHandshake buildSeqBufferLogic(RTLBuilder &s, BackedgeBuilder &bb,
                                     Type dataType, unsigned size,
                                     InputHandshake &input,
                                     OutputHandshake &output,
                                     llvm::ArrayRef<int64_t> initValues) const {
    // Prime the buffer building logic with an initial stage, which just
    // wraps the input handshake.
    InputHandshake currentStage = input;

    for (unsigned i = 0; i < size; ++i) {
      bool isInitialized = i < initValues.size();
      auto initValue =
          isInitialized ? std::optional<int64_t>(initValues[i]) : std::nullopt;
      currentStage = SeqBufferStage(dataType, currentStage, bb, s, i, initValue)
                         .getOutput();
    }

    return currentStage;
  };
};

class FuncOpConversionPattern : public OpConversionPattern<dc::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dc::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    ModulePortInfo ports =
        getPortInfoForOpTypes(op, op.getArgumentTypes(), op.getResultTypes());

    HWModuleLike hwModule;
    if (op.isExternal()) {
      hwModule = rewriter.create<hw::HWModuleExternOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
    } else {
      auto hwModuleOp = rewriter.create<hw::HWModuleOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
      auto args = hwModuleOp.getArguments().drop_back(2);
      rewriter.inlineBlockBefore(&op.getBody().front(),
                                 hwModuleOp.getBodyBlock()->getTerminator(),
                                 args);
      hwModule = hwModuleOp;
    }

    // Was any predeclaration associated with this func? If so, replace uses
    // with the newly created module and erase the predeclaration.
    if (auto predecl =
            op->getAttrOfType<FlatSymbolRefAttr>(kPredeclarationAttr)) {
      auto *parentOp = op->getParentOp();
      auto *predeclModule =
          SymbolTable::lookupSymbolIn(parentOp, predecl.getValue());
      if (predeclModule) {
        if (failed(SymbolTable::replaceAllSymbolUses(
                predeclModule, hwModule.getModuleNameAttr(), parentOp)))
          return failure();
        rewriter.eraseOp(predeclModule);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW Top-module Related Functions
//===----------------------------------------------------------------------===//

static LogicalResult convertFuncOp(ESITypeConverter &typeConverter,
                                   ConversionTarget &target,
                                   handshake::FuncOp op,
                                   OpBuilder &moduleBuilder) {

  std::map<std::string, unsigned> instanceNameCntr;
  NameUniquer instanceUniquer = [&](Operation *op) {
    std::string instName = getCallName(op);
    if (auto idAttr = op->getAttrOfType<IntegerAttr>("handshake_id"); idAttr) {
      // We use a special naming convention for operations which have a
      // 'handshake_id' attribute.
      instName += "_id" + std::to_string(idAttr.getValue().getZExtValue());
    } else {
      // Fallback to just prefixing with an integer.
      instName += std::to_string(instanceNameCntr[instName]++);
    }
    return instName;
  };

  auto ls =
      DCLoweringState{op->getParentOfType<mlir::ModuleOp>(), instanceUniquer};
  RewritePatternSet patterns(op.getContext());
  patterns.insert<FuncOpConversionPattern, ReturnConversionPattern>(
      op.getContext());

  patterns.insert<
      ForkConversionPattern, JoinConversionPattern, MergeConversionPattern,
      BranchConversionPattern, PackConversionPattern, UnpackConversionPattern,
      BufferConversionPattern, SourceConversionPattern, SinkConversionPattern>(
      typeConverter, op.getContext(), moduleBuilder, ls);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return op->emitOpError() << "error during conversion";
  return success();
}

namespace {
class DCToHWPass : public DCToHWBase<DCToHWPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Lowering to HW requires that every DC-typed value is used exactly once.
    // Check whether this precondition is met, and if not, exit.
    for (auto f : mod.getOps<dc::FuncOp>()) {
      if (auto res = verifyAllValuesHasOneUse(f); failed(res)) {
        f.emitOpError() << "DCToHW: failed to verify that all values "
                           "are used exactly once. Remember to run the "
                           "fork/sink materialization pass before HW lowering.";
        signalPassFailure();
        return;
      }
    }

    ESITypeConverter typeConverter;
    ConversionTarget target(getContext());
    // All top-level logic of a handshake module will be the interconnectivity
    // between instantiated modules.
    target.addIllegalDialect<dc::DCDialect>();
    target.addLegalDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect>();

    OpBuilder submoduleBuilder(mod.getContext());
    submoduleBuilder.setInsertionPointToStart(mod.getBody());
    for (auto f : mod.getOps<dc::FuncOp>()) {
      if (failed(convertFuncOp(typeConverter, target, f, submoduleBuilder))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

// TODO: remember to consider fork-sink materialization; must be run prior.
// Just check it, like we do for handshake.

std::unique_ptr<mlir::Pass> circt::createDCToHWPass() {
  return std::make_unique<DCToHWPass>();
}
