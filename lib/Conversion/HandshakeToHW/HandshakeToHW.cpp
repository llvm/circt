//===- HandshakeToHW.cpp - Translate Handshake into HW ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HandshakeToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

namespace {

// Shared state used by various functions; captured in a struct to reduce the
// number of arguments that we have to pass around.
struct HandshakeLoweringState {
  ModuleOp parentModule;
  NameUniquer nameUniquer;
};

// Wraps a type into an ESI ChannelType type. The inner type is converted to
// ensure comprehensability by the RTL dialects.
static Type esiWrapper(Type t) {
  // Translate index types to something HW understands.
  if (t.isa<IndexType>())
    t = IntegerType::get(t.getContext(), 64);

  // Already a channel type.
  if (t.isa<esi::ChannelType>())
    return t;

  return esi::ChannelType::get(t.getContext(), t);
}

// A type converter is needed to perform the in-flight materialization of "raw"
// (non-ESI channel) types to their ESI channel correspondents. This comes into
// effect when backedges exist in the input IR.
class ESITypeConverter : public TypeConverter {
public:
  ESITypeConverter() {
    addConversion([](Type type) -> Type { return esiWrapper(type); });

    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> llvm::Optional<mlir::Value> {
          if (inputs.size() != 1)
            return llvm::None;
          return inputs[0];
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> llvm::Optional<mlir::Value> {
          if (inputs.size() != 1)
            return llvm::None;
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

/// Returns a set of types which may uniquely identify the provided op. Return
/// value is <inputTypes, outputTypes>.
using DiscriminatingTypes = std::pair<SmallVector<Type>, SmallVector<Type>>;
static DiscriminatingTypes getHandshakeDiscriminatingTypes(Operation *op) {
  return TypeSwitch<Operation *, DiscriminatingTypes>(op)
      .Case<MemoryOp>([&](auto memOp) {
        return DiscriminatingTypes{{},
                                   {memOp.getMemRefType().getElementType()}};
      })
      .Default([&](auto) {
        // By default, all in- and output types which is not a control type
        // (NoneType) are discriminating types.
        std::vector<Type> inTypes, outTypes;
        llvm::transform(op->getOperands(), std::back_inserter(inTypes),
                        getOperandDataType);
        llvm::transform(op->getResults(), std::back_inserter(outTypes),
                        getOperandDataType);
        return DiscriminatingTypes{filterNoneTypes(inTypes),
                                   filterNoneTypes(outTypes)};
      });
}

/// Get type name. Currently we only support integer or index types.
/// The emitted type aligns with the getFIRRTLType() method. Thus all integers
/// other than signed integers will be emitted as unsigned.
static std::string getTypeName(Location loc, Type type) {
  std::string typeName;
  // Builtin types
  if (type.isIntOrIndex()) {
    if (auto indexType = type.dyn_cast<IndexType>())
      typeName += "_ui" + std::to_string(indexType.kInternalStorageBitWidth);
    else if (type.isSignedInteger())
      typeName += "_si" + std::to_string(type.getIntOrFloatBitWidth());
    else
      typeName += "_ui" + std::to_string(type.getIntOrFloatBitWidth());
  } else
    emitError(loc) << "unsupported data type '" << type << "'";

  return typeName;
}

namespace {

/// A class to be used with getPortInfoForOp. Provides an opaque interface for
/// generating the port names of an operation; handshake operations generate
/// names by the Handshake NamedIOInterface;  and other operations, such as
/// arith ops, are assigned default names.
class HandshakePortNameGenerator {
public:
  explicit HandshakePortNameGenerator(Operation *op)
      : builder(op->getContext()) {
    auto namedOpInterface = dyn_cast<handshake::NamedIOInterface>(op);
    if (namedOpInterface)
      inferFromNamedOpInterface(namedOpInterface);
    else if (auto funcOp = dyn_cast<handshake::FuncOp>(op))
      inferFromFuncOp(funcOp);
    else
      inferDefault(op);
  }

  StringAttr inputName(unsigned idx) { return inputs[idx]; }
  StringAttr outputName(unsigned idx) { return outputs[idx]; }

private:
  using IdxToStrF = const std::function<std::string(unsigned)> &;
  void infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF) {
    llvm::transform(
        llvm::enumerate(op->getOperandTypes()), std::back_inserter(inputs),
        [&](auto it) { return builder.getStringAttr(inF(it.index())); });
    llvm::transform(
        llvm::enumerate(op->getResultTypes()), std::back_inserter(outputs),
        [&](auto it) { return builder.getStringAttr(outF(it.index())); });
  }

  void inferDefault(Operation *op) {
    infer(
        op, [](unsigned idx) { return "in" + std::to_string(idx); },
        [](unsigned idx) { return "out" + std::to_string(idx); });
  }

  void inferFromNamedOpInterface(handshake::NamedIOInterface op) {
    infer(
        op, [&](unsigned idx) { return op.getOperandName(idx); },
        [&](unsigned idx) { return op.getResultName(idx); });
  }

  void inferFromFuncOp(handshake::FuncOp op) {
    auto inF = [&](unsigned idx) { return op.getArgName(idx).str(); };
    auto outF = [&](unsigned idx) { return op.getResName(idx).str(); };
    llvm::transform(
        llvm::enumerate(op.getArgumentTypes()), std::back_inserter(inputs),
        [&](auto it) { return builder.getStringAttr(inF(it.index())); });
    llvm::transform(
        llvm::enumerate(op.getResultTypes()), std::back_inserter(outputs),
        [&](auto it) { return builder.getStringAttr(outF(it.index())); });
  }

  Builder builder;
  llvm::SmallVector<StringAttr> inputs;
  llvm::SmallVector<StringAttr> outputs;
};

/// Construct a name for creating HW sub-module.
static std::string getSubModuleName(Operation *oldOp) {
  if (auto instanceOp = dyn_cast<handshake::InstanceOp>(oldOp); instanceOp)
    return instanceOp.getModule().str();

  std::string subModuleName = getBareSubModuleName(oldOp);

  // Add value of the constant operation.
  if (auto constOp = dyn_cast<handshake::ConstantOp>(oldOp)) {
    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
      auto intType = intAttr.getType();

      if (intType.isSignedInteger())
        subModuleName += "_c" + std::to_string(intAttr.getSInt());
      else if (intType.isUnsignedInteger())
        subModuleName += "_c" + std::to_string(intAttr.getUInt());
      else
        subModuleName += "_c" + std::to_string((uint64_t)intAttr.getInt());
    } else
      oldOp->emitError("unsupported constant type");
  }

  // Add discriminating in- and output types.
  auto [inTypes, outTypes] = getHandshakeDiscriminatingTypes(oldOp);
  if (!inTypes.empty())
    subModuleName += "_in";
  for (auto inType : inTypes)
    subModuleName += getTypeName(oldOp->getLoc(), inType);

  if (!outTypes.empty())
    subModuleName += "_out";
  for (auto outType : outTypes)
    subModuleName += getTypeName(oldOp->getLoc(), outType);

  // Add memory ID.
  if (auto memOp = dyn_cast<handshake::MemoryOp>(oldOp))
    subModuleName += "_id" + std::to_string(memOp.getId());

  // Add compare kind.
  if (auto comOp = dyn_cast<mlir::arith::CmpIOp>(oldOp))
    subModuleName += "_" + stringifyEnum(comOp.getPredicate()).str();

  // Add buffer information.
  if (auto bufferOp = dyn_cast<handshake::BufferOp>(oldOp)) {
    subModuleName += "_" + std::to_string(bufferOp.getNumSlots()) + "slots";
    if (bufferOp.isSequential())
      subModuleName += "_seq";
    else
      subModuleName += "_fifo";
  }

  // Add control information.
  if (auto ctrlInterface = dyn_cast<handshake::ControlInterface>(oldOp);
      ctrlInterface && ctrlInterface.isControl()) {
    // Add some additional discriminating info for non-typed operations.
    subModuleName += "_" + std::to_string(oldOp->getNumOperands()) + "ins_" +
                     std::to_string(oldOp->getNumResults()) + "outs";
    subModuleName += "_ctrl";
  } else {
    assert(
        (!inTypes.empty() || !outTypes.empty()) &&
        "Insufficient discriminating type info generated for the operation!");
  }

  return subModuleName;
}

} // namespace

//===----------------------------------------------------------------------===//
// HW Sub-module Related Functions
//===----------------------------------------------------------------------===//

/// Check whether a submodule with the same name has been created elsewhere in
/// the top level module. Return the matched module operation if true, otherwise
/// return nullptr.
static Operation *checkSubModuleOp(mlir::ModuleOp parentModule,
                                   StringRef modName) {
  if (auto mod = parentModule.lookupSymbol<HWModuleOp>(modName))
    return mod;
  if (auto mod = parentModule.lookupSymbol<HWModuleExternOp>(modName))
    return mod;
  return nullptr;
}

static Operation *checkSubModuleOp(mlir::ModuleOp parentModule,
                                   Operation *oldOp) {
  auto *moduleOp = checkSubModuleOp(parentModule, getSubModuleName(oldOp));

  if (isa<handshake::InstanceOp>(oldOp))
    assert(moduleOp &&
           "handshake.instance target modules should always have been lowered "
           "before the modules that reference them!");
  return moduleOp;
}

static ModulePortInfo getPortInfoForOp(ConversionPatternRewriter &rewriter,
                                       Operation *op, TypeRange inputs,
                                       TypeRange outputs) {
  ModulePortInfo ports({}, {});
  HandshakePortNameGenerator portNames(op);

  // Add all inputs of funcOp.
  unsigned inIdx = 0;
  for (auto &arg : llvm::enumerate(inputs)) {
    ports.inputs.push_back({portNames.inputName(arg.index()),
                            PortDirection::INPUT, esiWrapper(arg.value()),
                            arg.index(), StringAttr{}});
    inIdx++;
  }

  // Add all outputs of funcOp.
  for (auto &res : llvm::enumerate(outputs)) {
    ports.outputs.push_back({portNames.outputName(res.index()),
                             PortDirection::OUTPUT, esiWrapper(res.value()),
                             res.index(), StringAttr{}});
  }

  // Add clock and reset signals.
  if (op->hasTrait<mlir::OpTrait::HasClock>()) {
    ports.inputs.push_back({rewriter.getStringAttr("clock"),
                            PortDirection::INPUT, rewriter.getI1Type(), inIdx++,
                            StringAttr{}});
    ports.inputs.push_back({rewriter.getStringAttr("reset"),
                            PortDirection::INPUT, rewriter.getI1Type(), inIdx,
                            StringAttr{}});
  }

  return ports;
}

/// Returns a vector of PortInfo's which defines the FIRRTL interface of the
/// to-be-converted op.
static ModulePortInfo getPortInfoForOp(ConversionPatternRewriter &rewriter,
                                       Operation *op) {
  return getPortInfoForOp(rewriter, op, op->getOperandTypes(),
                          op->getResultTypes());
}

namespace {

// Input handshakes contain a resolved valid and (optional )data signal, and
// a to-be-assigned ready signal.
struct InputHandshake {
  Value valid;
  std::shared_ptr<Backedge> ready;
  Value data;
};

// Output handshakes contain a resolved ready, and to-be-assigned valid and
// (optional) data signals.
struct OutputHandshake {
  std::shared_ptr<Backedge> valid;
  Value ready;
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
  llvm::SmallVector<Value> getInputDatas() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.data; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getOutputValids() {
    return extractValues<std::shared_ptr<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<Value> getOutputReadys() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.ready; });
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
  RTLBuilder(OpBuilder &builder, Location loc, Value clk = Value(),
             Value rst = Value())
      : b(builder), loc(loc), clk(clk), rst(rst) {}

  Value constant(unsigned width, int64_t value, Location *extLoc = nullptr) {
    return b.create<hw::ConstantOp>(getLoc(extLoc), APInt(width, value));
  }
  std::pair<Value, Value> wrap(Value data, Value valid,
                               Location *extLoc = nullptr) {
    auto wrapOp = b.create<esi::WrapValidReadyOp>(getLoc(extLoc), data, valid);
    return {wrapOp.getResult(0), wrapOp.getResult(1)};
  }
  std::pair<Value, Value> unwrap(Value channel, Value ready,
                                 Location *extLoc = nullptr) {
    auto unwrapOp =
        b.create<esi::UnwrapValidReadyOp>(getLoc(extLoc), channel, ready);
    return {unwrapOp.getResult(0), unwrapOp.getResult(1)};
  }

  // Various syntactic sugar functions.
  Value reg(StringRef name, Value in, Value rstValue, Value clk = Value(),
            Value rst = Value(), Location *extLoc = nullptr) {
    Value resolvedClk = clk ? clk : this->clk;
    Value resolvedRst = rst ? rst : this->rst;
    assert(resolvedClk &&
           "No global clock provided to this RTLBuilder - a clock "
           "signal must be provided to the reg(...) function.");
    assert(resolvedRst &&
           "No global reset provided to this RTLBuilder - a reset "
           "signal must be provided to the reg(...) function.");

    return b.create<seq::CompRegOp>(getLoc(extLoc), in.getType(), in,
                                    resolvedClk, name, resolvedRst, rstValue,
                                    StringAttr());
  }

  // Bitwise 'and'.
  Value bAnd(ValueRange values, Location *extLoc = nullptr) {
    return b.create<comb::AndOp>(getLoc(extLoc), values).getResult();
  }

  // Bitwise 'not'.
  Value bNot(Value value, Location *extLoc = nullptr) {
    return comb::createOrFoldNot(getLoc(extLoc), value, b);
  }

  Value shl(Value value, Value shift, Location *extLoc = nullptr) {
    return b.create<comb::ShlOp>(getLoc(extLoc), value, shift).getResult();
  }

  Value concat(ValueRange values, Location *extLoc = nullptr) {
    return b.create<comb::ConcatOp>(getLoc(extLoc), values).getResult();
  }

  // Extract bits v[hi:lo] (inclusive).
  Value extract(Value v, unsigned lo, unsigned hi, Location *extLoc = nullptr) {
    unsigned width = hi - lo + 1;
    return b.create<comb::ExtractOp>(getLoc(extLoc), v, lo, width).getResult();
  }

  // Truncates 'value' to its lower 'width' bits.
  Value truncate(Value value, unsigned width, Location *extLoc = nullptr) {
    return extract(value, 0, width - 1, extLoc);
  }

  Value zext(Value value, unsigned outWidth, Location *extLoc = nullptr) {
    unsigned inWidth = value.getType().getIntOrFloatBitWidth();
    assert(inWidth < outWidth &&
           "zext: input width must be smaller than output "
           "width.");
    if (inWidth == outWidth)
      return value;
    auto c0 = constant(outWidth - inWidth, 0, extLoc);
    return concat({c0, value}, extLoc);
  }

  Value sext(Value value, unsigned outWidth, Location *extLoc = nullptr) {
    return comb::createOrFoldSExt(getLoc(extLoc), value,
                                  b.getIntegerType(outWidth), b);
  }

  // Extracts a single bit v[bit].
  Value bit(Value v, unsigned index, Location *extLoc = nullptr) {
    return extract(v, index, index, extLoc);
  }

  // Creates a hw.array of the given values.
  Value arrayCreate(ValueRange values, Location *extLoc = nullptr) {
    return b.create<hw::ArrayCreateOp>(getLoc(extLoc), values).getResult();
  }

  // Extract the 'index'th value from the input array.
  Value arrayGet(Value array, Value index, Location *extLoc = nullptr) {
    return b.create<hw::ArrayGetOp>(getLoc(extLoc), array, index).getResult();
  }

  // Muxes a range of values.
  // The select signal is expected to be a decimal value which selects starting
  // from the lowest index of value.
  Value mux(ValueRange values, Value index, Location *extLoc = nullptr) {
    if (values.size() == 2)
      return b.create<comb::MuxOp>(getLoc(extLoc), index, values[1], values[0]);

    return arrayGet(arrayCreate(values, extLoc), index, extLoc);
  }

  Location getLoc(Location *extLoc = nullptr) { return extLoc ? *extLoc : loc; }
  OpBuilder &b;
  Location loc;
  Value clk, rst;
};

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
class HandshakeConversionPattern : public OpConversionPattern<T> {
public:
  HandshakeConversionPattern(ESITypeConverter &typeConverter,
                             MLIRContext *context, OpBuilder &submoduleBuilder,
                             HandshakeLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, context),
        submoduleBuilder(submoduleBuilder), ls(ls) {}

  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check if a submodule has already been created for the op. If so,
    // instantiate the submodule. Else, run the pattern-defined module builder.
    hw::HWModuleLike implModule = checkSubModuleOp(ls.parentModule, op);
    if (!implModule) {
      auto portInfo = ModulePortInfo(getPortInfoForOp(rewriter, op));

      implModule = submoduleBuilder.create<hw::HWModuleOp>(
          op.getLoc(), submoduleBuilder.getStringAttr(getSubModuleName(op)),
          portInfo, [&](OpBuilder &b, hw::HWModulePortAccessor &ports) {
            // if 'op' has clock trait, extract these and provide them to the
            // RTL builder.
            Value clk, rst;
            if (op->template hasTrait<mlir::OpTrait::HasClock>()) {
              clk = ports.getInput("clock");
              rst = ports.getInput("reset");
            }

            BackedgeBuilder bb(b, op.getLoc());
            RTLBuilder s(b, op.getLoc(), clk, rst);
            this->buildModule(op, bb, s, ports);
          });
    }

    // Instantiate the submodule.
    llvm::SmallVector<Value> operands = adaptor.getOperands();
    addSequentialIOOperandsIfNeeded(op, operands);
    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, implModule, rewriter.getStringAttr(ls.nameUniquer(op)), operands);
    return success();
  }

  virtual void buildModule(T op, BackedgeBuilder &bb, RTLBuilder &builder,
                           hw::HWModulePortAccessor &ports) const = 0;

  // Syntactic sugar functions.
  // Unwraps an ESI-interfaced module into its constituent handshake signals.
  // Backedges are created for the to-be-resolved signals, and output ports
  // are assigned to their wrapped counterparts.
  UnwrappedIO unwrapIO(RTLBuilder &s, BackedgeBuilder &bb,
                       hw::HWModulePortAccessor &ports) const {
    UnwrappedIO unwrapped;
    for (auto port : ports.getInputs()) {
      if (!isa<esi::ChannelType>(port.getType()))
        continue;
      InputHandshake hs;
      auto ready = std::make_shared<Backedge>(bb.get(s.b.getI1Type()));
      auto [data, valid] = s.unwrap(port, *ready);
      hs.data = data;
      hs.valid = valid;
      hs.ready = ready;
      unwrapped.inputs.push_back(hs);
    }
    for (auto &outputInfo : ports.getModulePortInfo().outputs) {
      if (!isa<esi::ChannelType>(outputInfo.type))
        continue;
      OutputHandshake hs;
      auto data = std::make_shared<Backedge>(
          bb.get(cast<esi::ChannelType>(outputInfo.type).getInner()));
      auto valid = std::make_shared<Backedge>(bb.get(s.b.getI1Type()));
      auto [dataCh, ready] = s.wrap(*data, *valid);
      hs.data = data;
      hs.valid = valid;
      hs.ready = ready;
      ports.setOutput(outputInfo.name, dataCh);
      unwrapped.outputs.push_back(hs);
    }
    return unwrapped;
  }

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
    setAllReadyWithCond(s, inputs, output, allValid);
  }

  // Builds mux logic for the given inputs and outputs.
  // Note: it is assumed that the caller has removed the 'select' signal from
  // the 'unwrapped' inputs and provide it as a separate argument.
  void buildMuxLogic(RTLBuilder &s, UnwrappedIO &unwrapped,
                     InputHandshake &select) const {
    // ============================= Control logic =============================
    // Decimal-to-1-hot decoder. 'shl' operands must be identical in size.
    size_t size = unwrapped.inputs.size();
    Value truncatedSelect = select.data.getType().getIntOrFloatBitWidth() > size
                                ? s.truncate(select.data, size)
                                : select.data;
    auto c1s = s.constant(size, 1);
    auto truncSelectZext = s.zext(truncatedSelect, size);
    auto select1h = s.shl(c1s, truncSelectZext);
    auto &res = unwrapped.outputs[0];

    // Mux input valid signals.
    auto selectedInputValid = s.mux(unwrapped.getInputValids(), select.data);
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

      // '&' that with the result valid and ready, and assign to the input ready
      // signal.
      auto activeAndResultValidAndReady =
          s.bAnd({isSelected, resValidAndReady});
      in.ready->setValue(activeAndResultValidAndReady);
    }

    // ============================== Data logic ===============================
    res.data->setValue(s.mux(unwrapped.getInputDatas(), select.data));
  }

  void buildForkLogic(RTLBuilder &s, BackedgeBuilder &bb, InputHandshake &input,
                      ArrayRef<OutputHandshake> outputs,
                      hw::HWModulePortAccessor &ports) const {
    auto c0I1 = s.constant(1, 0);
    llvm::SmallVector<Value> doneWires;
    for (auto [i, output] : llvm::enumerate(outputs)) {
      auto done = bb.get(s.b.getI1Type());
      auto emitted = s.bAnd({done, s.bNot(*input.ready)});
      auto emittedReg = s.reg("emitted_" + std::to_string(i), emitted, c0I1);
      auto outValid = s.bAnd({s.bNot(emittedReg), input.valid});
      output.data->setValue(input.data);
      output.valid->setValue(outValid);
      auto validReady = s.bAnd({output.ready, input.valid});
      done.setValue(s.bAnd({validReady, emittedReg}));
      doneWires.push_back(done);
    }
    input.ready->setValue(s.bAnd(doneWires));
  }

  // Builds a unit-rate actor around an inner operation. 'unitBuilder' is a
  // function which takes the set of unwrapped data inputs, and returns a value
  // which should be assigned to the output data value.
  void
  buildUnitRateLogic(RTLBuilder &s, UnwrappedIO &unwrappedIO,
                     llvm::function_ref<Value(ValueRange)> unitBuilder) const {
    assert(unwrappedIO.outputs.size() == 1 &&
           "Expected exactly one output for unit-rate actor");
    // Control logic.
    this->buildJoinLogic(s, unwrappedIO.inputs, unwrappedIO.outputs[0]);
    unwrappedIO.outputs[0].valid->setValue(
        s.bAnd(unwrappedIO.getInputValids()));

    // Data logic.
    auto unitRes = unitBuilder(unwrappedIO.getInputDatas());
    unwrappedIO.outputs[0].data->setValue(unitRes);
  }

  void buildExtendLogic(RTLBuilder &s, UnwrappedIO &unwrappedIO,
                        bool signExtend) const {
    size_t outWidth = static_cast<Value>(*unwrappedIO.outputs[0].data)
                          .getType()
                          .getIntOrFloatBitWidth();
    buildUnitRateLogic(s, unwrappedIO, [&](ValueRange inputs) {
      if (signExtend)
        return s.sext(inputs[0], outWidth);
      return s.zext(inputs[0], outWidth);
    });
  }

  void buildTruncateLogic(RTLBuilder &s, UnwrappedIO &unwrappedIO,
                          unsigned targetWidth) const {
    size_t outWidth = static_cast<Value>(*unwrappedIO.outputs[0].data)
                          .getType()
                          .getIntOrFloatBitWidth();
    buildUnitRateLogic(s, unwrappedIO, [&](ValueRange inputs) {
      return s.truncate(inputs[0], outWidth);
    });
  }

private:
  OpBuilder &submoduleBuilder;
  HandshakeLoweringState &ls;
};

class ForkConversionPattern : public HandshakeConversionPattern<ForkOp> {
public:
  using HandshakeConversionPattern<ForkOp>::HandshakeConversionPattern;
  void buildModule(ForkOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrapped = unwrapIO(s, bb, ports);
    buildForkLogic(s, bb, unwrapped.inputs[0], unwrapped.outputs, ports);
  }
};

class JoinConversionPattern : public HandshakeConversionPattern<JoinOp> {
public:
  using HandshakeConversionPattern<JoinOp>::HandshakeConversionPattern;
  void buildModule(JoinOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);
    buildJoinLogic(s, unwrappedIO.inputs, unwrappedIO.outputs[0]);
  };
};

class MuxConversionPattern : public HandshakeConversionPattern<MuxOp> {
public:
  using HandshakeConversionPattern<MuxOp>::HandshakeConversionPattern;
  void buildModule(MuxOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);

    // Extract select signal from the unwrapped IO.
    auto select = unwrappedIO.inputs[0];
    unwrappedIO.inputs.erase(unwrappedIO.inputs.begin());
    buildMuxLogic(s, unwrappedIO, select);
  };
};

class SelectConversionPattern
    : public HandshakeConversionPattern<arith::SelectOp> {
public:
  using HandshakeConversionPattern<arith::SelectOp>::HandshakeConversionPattern;
  void buildModule(arith::SelectOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);

    // Extract select signal from the unwrapped IO.
    auto select = unwrappedIO.inputs[0];
    unwrappedIO.inputs.erase(unwrappedIO.inputs.begin());

    // Swap order of inputs to match MuxOp (0 => input[0], 1 => input[1]).
    std::swap(unwrappedIO.inputs[0], unwrappedIO.inputs[1]);
    buildMuxLogic(s, unwrappedIO, select);
  };
};

class ReturnConversionPattern
    : public OpConversionPattern<handshake::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Locate existing output op, Append operands to output op, and move to the
    // end of the block.
    auto parent = cast<hw::HWModuleOp>(op->getParentOp());
    auto outputOp = *parent.getBodyBlock()->getOps<hw::OutputOp>().begin();
    outputOp->setOperands(adaptor.getOperands());
    outputOp->moveAfter(&parent.getBodyBlock()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

// Converts an arbitrary operation into a unit rate actor. A unit rate actor
// will transact once all inputs are valid and its output is ready.
template <typename TIn, typename TOut = TIn>
class UnitRateConversionPattern : public HandshakeConversionPattern<TIn> {
public:
  using HandshakeConversionPattern<TIn>::HandshakeConversionPattern;
  void buildModule(TIn op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    this->buildUnitRateLogic(s, unwrappedIO, [&](ValueRange inputs) {
      // Create TOut - it is assumed that TOut trivially
      // constructs from the input data signals of TIn.
      return s.b.create<TOut>(op.getLoc(), inputs);
    });
  };
};

template <typename TIn, bool signExtend>
class ExtendConversionPattern : public HandshakeConversionPattern<TIn> {
public:
  using HandshakeConversionPattern<TIn>::HandshakeConversionPattern;
  void buildModule(TIn op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    this->buildExtendLogic(s, unwrappedIO, /*signExtend=*/signExtend);
  };
};

class ComparisonConversionPattern
    : public HandshakeConversionPattern<arith::CmpIOp> {
public:
  using HandshakeConversionPattern<arith::CmpIOp>::HandshakeConversionPattern;
  void buildModule(arith::CmpIOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    auto buildCompareLogic = [&](comb::ICmpPredicate predicate) {
      return buildUnitRateLogic(s, unwrappedIO, [&](ValueRange inputs) {
        return s.b.create<comb::ICmpOp>(op.getLoc(), predicate, inputs[0],
                                        inputs[1]);
      });
    };

    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return buildCompareLogic(comb::ICmpPredicate::eq);
    case arith::CmpIPredicate::ne:
      return buildCompareLogic(comb::ICmpPredicate::ne);
    case arith::CmpIPredicate::slt:
      return buildCompareLogic(comb::ICmpPredicate::slt);
    case arith::CmpIPredicate::ult:
      return buildCompareLogic(comb::ICmpPredicate::ult);
    case arith::CmpIPredicate::sle:
      return buildCompareLogic(comb::ICmpPredicate::sle);
    case arith::CmpIPredicate::ule:
      return buildCompareLogic(comb::ICmpPredicate::ule);
    case arith::CmpIPredicate::sgt:
      return buildCompareLogic(comb::ICmpPredicate::sgt);
    case arith::CmpIPredicate::ugt:
      return buildCompareLogic(comb::ICmpPredicate::ugt);
    case arith::CmpIPredicate::sge:
      return buildCompareLogic(comb::ICmpPredicate::sge);
    case arith::CmpIPredicate::uge:
      return buildCompareLogic(comb::ICmpPredicate::uge);
    }
    assert(false && "invalid CmpIOp");
  };
};

class TruncateConversionPattern
    : public HandshakeConversionPattern<arith::TruncIOp> {
public:
  using HandshakeConversionPattern<arith::TruncIOp>::HandshakeConversionPattern;
  void buildModule(arith::TruncIOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    unsigned targetBits = op.getResult().getType().getIntOrFloatBitWidth();
    buildTruncateLogic(s, unwrappedIO, targetBits);
  };
};

class IndexCastConversionPattern
    : public HandshakeConversionPattern<arith::IndexCastOp> {
public:
  using HandshakeConversionPattern<
      arith::IndexCastOp>::HandshakeConversionPattern;
  void buildModule(arith::IndexCastOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    unsigned sourceBits = op.getIn().getType().getIntOrFloatBitWidth();
    unsigned targetBits = op.getResult().getType().getIntOrFloatBitWidth();
    if (targetBits < sourceBits)
      buildTruncateLogic(s, unwrappedIO, targetBits);
    else
      buildExtendLogic(s, unwrappedIO, /*signExtend=*/true);
  };
};

template <typename T>
class ExtModuleConversionPattern : public OpConversionPattern<T> {
public:
  ExtModuleConversionPattern(ESITypeConverter &typeConverter,
                             MLIRContext *context, OpBuilder &submoduleBuilder,
                             HandshakeLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, context),
        submoduleBuilder(submoduleBuilder), ls(ls) {}
  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    hw::HWModuleLike implModule = checkSubModuleOp(ls.parentModule, op);
    if (!implModule) {
      auto portInfo = ModulePortInfo(getPortInfoForOp(rewriter, op));
      implModule = submoduleBuilder.create<hw::HWModuleExternOp>(
          op.getLoc(), submoduleBuilder.getStringAttr(getSubModuleName(op)),
          portInfo);
    }

    llvm::SmallVector<Value> operands = adaptor.getOperands();
    addSequentialIOOperandsIfNeeded(op, operands);
    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, implModule, rewriter.getStringAttr(ls.nameUniquer(op)), operands);
    return success();
  }

private:
  OpBuilder &submoduleBuilder;
  HandshakeLoweringState &ls;
};

class FuncOpConversionPattern : public OpConversionPattern<handshake::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    ModulePortInfo ports = getPortInfoForOp(rewriter, op, op.getArgumentTypes(),
                                            op.getResultTypes());
    auto hwModule = rewriter.create<hw::HWModuleOp>(
        op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
    auto args = hwModule.getArguments().drop_back(2);
    rewriter.mergeBlockBefore(&op.getBody().front(),
                              hwModule.getBodyBlock()->getTerminator(), args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW Top-module Related Functions
//===----------------------------------------------------------------------===//

static bool isMemrefType(Type t) { return t.isa<mlir::MemRefType>(); }
static LogicalResult verifyHandshakeFuncOp(handshake::FuncOp &funcOp) {
  // @TODO: memory I/O is not yet supported. Figure out how to support memory
  // services in ESI.
  if (llvm::any_of(funcOp.getArgumentTypes(), isMemrefType) ||
      llvm::any_of(funcOp.getResultTypes(), isMemrefType))
    return emitError(funcOp.getLoc())
           << "memref ports are not yet supported in handshake-to-hw lowering.";
  return success();
}

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

  auto ls = HandshakeLoweringState{op->getParentOfType<mlir::ModuleOp>(),
                                   instanceUniquer};
  RewritePatternSet patterns(op.getContext());
  patterns.insert<FuncOpConversionPattern, ReturnConversionPattern>(
      op.getContext());
  patterns.insert<JoinConversionPattern, ForkConversionPattern>(
      typeConverter, op.getContext(), moduleBuilder, ls);

  patterns.insert<ExtModuleConversionPattern<handshake::ConstantOp>,
                  ExtModuleConversionPattern<handshake::BufferOp>,
                  ExtModuleConversionPattern<handshake::SinkOp>,
                  ExtModuleConversionPattern<handshake::ConditionalBranchOp>,
                  MuxConversionPattern, SelectConversionPattern,
                  UnitRateConversionPattern<arith::AddIOp, comb::AddOp>,
                  UnitRateConversionPattern<arith::SubIOp, comb::SubOp>,
                  UnitRateConversionPattern<arith::MulIOp, comb::MulOp>,
                  UnitRateConversionPattern<arith::DivUIOp, comb::DivSOp>,
                  UnitRateConversionPattern<arith::DivSIOp, comb::DivUOp>,
                  UnitRateConversionPattern<arith::RemUIOp, comb::ModUOp>,
                  UnitRateConversionPattern<arith::RemSIOp, comb::ModSOp>,
                  UnitRateConversionPattern<arith::AndIOp, comb::AndOp>,
                  UnitRateConversionPattern<arith::OrIOp, comb::OrOp>,
                  UnitRateConversionPattern<arith::XOrIOp, comb::XorOp>,
                  UnitRateConversionPattern<arith::ShLIOp, comb::OrOp>,
                  UnitRateConversionPattern<arith::ShRUIOp, comb::ShrUOp>,
                  UnitRateConversionPattern<arith::ShRSIOp, comb::ShrSOp>,
                  ComparisonConversionPattern,
                  ExtendConversionPattern<arith::ExtUIOp, /*signExtend=*/false>,
                  ExtendConversionPattern<arith::ExtSIOp, /*signExtend=*/true>,
                  TruncateConversionPattern, IndexCastConversionPattern>(
      typeConverter, op.getContext(), moduleBuilder, ls);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return op->emitOpError() << "error during conversion";
  return success();
}

namespace {
class HandshakeToHWPass : public HandshakeToHWBase<HandshakeToHWPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Lowering to HW requires that every value is used exactly once. Check
    // whether this precondition is met, and if not, exit.
    if (llvm::any_of(mod.getOps<handshake::FuncOp>(), [](auto f) {
          return failed(verifyAllValuesHasOneUse(f));
        })) {
      signalPassFailure();
      return;
    }

    // Resolve the instance graph to get a top-level module.
    std::string topLevel;
    handshake::InstanceGraph uses;
    SmallVector<std::string> sortedFuncs;
    if (resolveInstanceGraph(mod, uses, topLevel, sortedFuncs).failed()) {
      signalPassFailure();
      return;
    }

    ESITypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.addLegalDialect<HWDialect>();
    target.addIllegalDialect<handshake::HandshakeDialect>();

    // Convert the handshake.func operations in post-order wrt. the instance
    // graph. This ensures that any referenced submodules (through
    // handshake.instance) has already been lowered, and their HW module
    // equivalents are available.
    OpBuilder submoduleBuilder(mod.getContext());
    submoduleBuilder.setInsertionPointToStart(mod.getBody());
    for (auto &funcName : llvm::reverse(sortedFuncs)) {
      auto funcOp = mod.lookupSymbol<handshake::FuncOp>(funcName);
      assert(funcOp && "handshake.func not found in module!");
      if (failed(verifyHandshakeFuncOp(funcOp)) ||
          failed(
              convertFuncOp(typeConverter, target, funcOp, submoduleBuilder))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createHandshakeToHWPass() {
  return std::make_unique<HandshakeToHWPass>();
}
