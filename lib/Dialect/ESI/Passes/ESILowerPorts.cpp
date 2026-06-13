//===- ESILowerPorts.cpp - Lower ESI ports pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/MathExtras.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_LOWERESIPORTS
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;

// Returns either the string dialect attr stored in 'op' going by the name
// 'attrName' or 'def' if the attribute doesn't exist in 'op'.
inline static StringRef getStringAttributeOr(Operation *op, StringRef attrName,
                                             StringRef def) {
  auto attr = op->getAttrOfType<StringAttr>(attrName);
  if (attr)
    return attr.getValue();
  return def;
}

namespace {

/// Per-channel ESI signaling protocol policies (ValidReady, FIFO, ValidOnly).
/// These are stateless policy types with only static methods; they are never
/// instantiated. They are used as the policy type parameter of the
/// channel-port lowering containers below, which share their logic between
/// scalar channel ports and arrays of channel ports.
///
/// Each protocol has one or more control signals associated with the channel
/// data to, at minimum, indicate the validity of the data. The protocol defines
/// how these signals are named and whether there is a backpressure handshake
/// signal.
///
/// Each policy provides:
///  - getValiditySuffix(module): suffix for the forward validity signal.
///  - hasBackpressure: compile-time flag, true iff the protocol has a
///    backpressure handshake signal.
///  - getBackpressureSuffix(module): suffix for the backpressure handshake
///    signal (only defined when 'hasBackpressure' is true).
///  - wrap(b, loc, chanTy, data, validity): recreate a channel of type 'chanTy'
///    from its 'data' and 'validity' signals; returns {channel, backpressure}
///    (the backpressure value is null if the protocol has no backpressure
///    signal).
///  - unwrap(b, loc, chan, backpressure): break 'chan' into its {data,
///    validity} signals ('backpressure' is unused for protocols without a
///    backpressure signal).
///
/// Note: if we add a credit control signaling protocol in the future, this will
/// have to be re-thought.

/// ValidReady: 'valid' validity, 'ready' backpressure handshake.
struct ValidReadyProtocol {
  static constexpr bool hasBackpressure = true;
  static StringRef getValiditySuffix(Operation *module) {
    return getStringAttributeOr(module, extModPortValidSuffix, "_valid");
  }
  static StringRef getBackpressureSuffix(Operation *module) {
    return getStringAttributeOr(module, extModPortReadySuffix, "_ready");
  }
  static std::pair<Value, Value> wrap(OpBuilder &b, Location loc,
                                      ChannelType chanTy, Value data,
                                      Value validity) {
    auto wrap = WrapValidReadyOp::create(b, loc, data, validity);
    return {wrap.getChanOutput(), wrap.getReady()};
  }
  static std::pair<Value, Value> unwrap(OpBuilder &b, Location loc, Value chan,
                                        Value backpressure) {
    auto unwrap = UnwrapValidReadyOp::create(b, loc, chan, backpressure);
    return {unwrap.getRawOutput(), unwrap.getValid()};
  }
};

/// FIFO: 'empty' validity (empty == !valid), 'rden' (read-enable)
/// "backpressure" handshake.
struct FIFOProtocol {
  static constexpr bool hasBackpressure = true;
  static StringRef getValiditySuffix(Operation *module) {
    return getStringAttributeOr(module, extModPortEmptySuffix, "_empty");
  }
  static StringRef getBackpressureSuffix(Operation *module) {
    return getStringAttributeOr(module, extModPortRdenSuffix, "_rden");
  }
  static std::pair<Value, Value> wrap(OpBuilder &b, Location loc,
                                      ChannelType chanTy, Value data,
                                      Value validity) {
    auto wrap = WrapFIFOOp::create(
        b, loc, ArrayRef<Type>({chanTy, b.getI1Type()}), data, validity);
    return {wrap.getChanOutput(), wrap.getRden()};
  }
  static std::pair<Value, Value> unwrap(OpBuilder &b, Location loc, Value chan,
                                        Value backpressure) {
    auto unwrap = UnwrapFIFOOp::create(b, loc, chan, backpressure);
    return {unwrap.getData(), unwrap.getEmpty()};
  }
};

/// ValidOnly: forward 'valid' validity, no backpressure handshake.
struct ValidOnlyProtocol {
  static constexpr bool hasBackpressure = false;
  static StringRef getValiditySuffix(Operation *module) {
    return getStringAttributeOr(module, extModPortValidSuffix, "_valid");
  }
  static std::pair<Value, Value> wrap(OpBuilder &b, Location loc,
                                      ChannelType chanTy, Value data,
                                      Value validity) {
    auto wrap = WrapValidOnlyOp::create(b, loc, data, validity);
    return {wrap.getChanOutput(), Value()};
  }
  static std::pair<Value, Value> unwrap(OpBuilder &b, Location loc, Value chan,
                                        Value /*backpressure*/) {
    auto unwrap = UnwrapValidOnlyOp::create(b, loc, chan);
    return {unwrap.getRawOutput(), unwrap.getValid()};
  }
};

/// Return true if 'type' contains an ESI channel nested inside an aggregate.
/// Used to detect (and reject) ports which embed channels in a way this pass
/// cannot lower (e.g. arrays of arrays of channels, or structs of channels).
static bool containsChannel(Type type) {
  if (auto arr = dyn_cast<hw::ArrayType>(type)) {
    Type elem = arr.getElementType();
    return isa<esi::ChannelType>(elem) || containsChannel(elem);
  }
  if (auto str = dyn_cast<hw::StructType>(type))
    return llvm::any_of(str.getElements(), [](const auto &field) {
      return isa<esi::ChannelType>(field.type) || containsChannel(field.type);
    });
  return false;
}

/// Lower a single ESI channel port into its constituent wire-level signals
/// using the 'Protocol' signaling policy.
template <typename Protocol>
class ScalarChannelPort : public PortConversion {
public:
  using PortConversion::PortConversion;

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  // Port info for the lowered signals. 'backpressurePort' is only valid when
  // 'Protocol::hasBackpressure' is true.
  PortInfo dataPort, validityPort, backpressurePort;
};

/// Lower a port which is an array of ESI channels into arrays of the
/// constituent wire-level signals (one array element per channel) using the
/// 'Protocol' signaling policy.
template <typename Protocol>
class ArrayChannelPort : public PortConversion {
public:
  using PortConversion::PortConversion;

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  // Port info for the lowered signal arrays. 'backpressurePort' is only valid
  // when 'Protocol::hasBackpressure' is true.
  PortInfo dataPort, validityPort, backpressurePort;
};

// Emit an error about an unknown signaling standard on 'port'.
static FailureOr<std::unique_ptr<PortConversion>>
emitUnknownSignaling(Operation *op, hw::PortInfo port,
                     ChannelSignaling signaling) {
  auto error =
      op->emitOpError("encountered unknown signaling standard on port '")
      << stringifyEnum(signaling) << "'";
  error.attachNote(port.loc);
  return error;
}

/// Instantiate the channel-port lowering container 'PortKind' (e.g.
/// ScalarChannelPort or ArrayChannelPort) with the protocol policy selected by
/// 'signaling'. Returns failure for an unknown signaling standard.
template <template <typename> class PortKind>
static FailureOr<std::unique_ptr<PortConversion>>
buildChannelPort(PortConverterImpl &converter, hw::PortInfo port,
                 ChannelSignaling signaling) {
  switch (signaling) {
  case ChannelSignaling::ValidReady:
    return {std::make_unique<PortKind<ValidReadyProtocol>>(converter, port)};
  case ChannelSignaling::FIFO:
    return {std::make_unique<PortKind<FIFOProtocol>>(converter, port)};
  case ChannelSignaling::ValidOnly:
    return {std::make_unique<PortKind<ValidOnlyProtocol>>(converter, port)};
  }
  return emitUnknownSignaling(converter.getModule(), port, signaling);
}

class ESIPortConversionBuilder : public PortConversionBuilder {
public:
  using PortConversionBuilder::PortConversionBuilder;
  FailureOr<std::unique_ptr<PortConversion>> build(hw::PortInfo port) override {
    return llvm::TypeSwitch<Type, FailureOr<std::unique_ptr<PortConversion>>>(
               port.type)
        .Case([&](esi::ChannelType chanTy)
                  -> FailureOr<std::unique_ptr<PortConversion>> {
          return buildChannelPort<ScalarChannelPort>(converter, port,
                                                     chanTy.getSignaling());
        })
        .Case([&](hw::ArrayType arrTy)
                  -> FailureOr<std::unique_ptr<PortConversion>> {
          auto chanTy = dyn_cast<esi::ChannelType>(arrTy.getElementType());
          if (!chanTy) {
            // Channels nested deeper than a top-level array of channels are
            // not supported.
            if (containsChannel(arrTy)) {
              auto error = converter.getModule().emitOpError(
                  "cannot lower port containing channels nested inside an "
                  "aggregate other than a single array of channels");
              error.attachNote(port.loc);
              return error;
            }
            return PortConversionBuilder::build(port);
          }
          return buildChannelPort<ArrayChannelPort>(converter, port,
                                                    chanTy.getSignaling());
        })
        .Default([&](auto) { return PortConversionBuilder::build(port); });
  }
};
} // namespace

/// Extract element 'idx' from the array-typed value 'array'.
static Value getArrayElement(OpBuilder &b, Location loc, Value array,
                             size_t idx) {
  auto arrTy = cast<hw::ArrayType>(array.getType());
  IntegerType idxType = b.getIntegerType(
      std::max(1u, llvm::Log2_64_Ceil(arrTy.getNumElements())));
  Value idxVal = hw::ConstantOp::create(b, loc, idxType, idx);
  return hw::ArrayGetOp::create(b, loc, array, idxVal);
}

/// Pack a list of element values (with 'elements[i]' destined for array index
/// 'i') into an hw.array. hw.array_create takes its operands in reverse order
/// (operand 0 becomes the highest index), so the list is reversed to keep
/// 'elements[i]' at array index 'i'.
static Value packArray(OpBuilder &b, Location loc, ArrayRef<Value> elements) {
  SmallVector<Value> reversed(elements.rbegin(), elements.rend());
  return hw::ArrayCreateOp::create(b, loc, reversed);
}

//===----------------------------------------------------------------------===//
// ScalarChannelPort
//===----------------------------------------------------------------------===//

template <typename Protocol>
void ScalarChannelPort<Protocol>::buildInputSignals() {
  Operation *module = converter.getModule();
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);
  auto chanTy = cast<ChannelType>(origPort.type);

  StringRef inSuffix = getStringAttributeOr(module, extModPortInSuffix, "");
  StringRef outSuffix = getStringAttributeOr(module, extModPortOutSuffix, "");

  // The data and forward validity signals come into the module alongside the
  // data.
  Value data =
      converter.createNewInput(origPort, inSuffix, chanTy.getInner(), dataPort);
  Value validity = converter.createNewInput(
      origPort, Protocol::getValiditySuffix(module) + inSuffix, i1,
      validityPort);

  Value backpressure;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    // Recreate the original channel value from the lowered signals. (A later
    // pass takes care of eliminating the ESI ops.)
    auto [chan, bp] = Protocol::wrap(b, b.getLoc(), chanTy, data, validity);
    backpressure = bp;
    body->getArgument(origPort.argNum).replaceAllUsesWith(chan);
  }

  // The backpressure handshake signal (if any) leaves the module.
  if constexpr (Protocol::hasBackpressure)
    converter.createNewOutput(
        origPort, Protocol::getBackpressureSuffix(module) + outSuffix, i1,
        backpressure, backpressurePort);
}

template <typename Protocol>
void ScalarChannelPort<Protocol>::mapInputSignals(
    OpBuilder &b, Operation *inst, Value, SmallVectorImpl<Value> &newOperands,
    ArrayRef<Backedge> newResults) {
  Value backpressure;
  if constexpr (Protocol::hasBackpressure)
    backpressure = newResults[backpressurePort.argNum];
  auto [data, validity] = Protocol::unwrap(
      b, inst->getLoc(), inst->getOperand(origPort.argNum), backpressure);
  newOperands[dataPort.argNum] = data;
  newOperands[validityPort.argNum] = validity;
}

template <typename Protocol>
void ScalarChannelPort<Protocol>::buildOutputSignals() {
  Operation *module = converter.getModule();
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);
  auto chanTy = cast<ChannelType>(origPort.type);

  StringRef inSuffix = getStringAttributeOr(module, extModPortInSuffix, "");
  StringRef outSuffix = getStringAttributeOr(module, extModPortOutSuffix, "");

  // The backpressure handshake signal (if any) comes into the module.
  Value backpressure;
  if constexpr (Protocol::hasBackpressure)
    backpressure = converter.createNewInput(
        origPort, Protocol::getBackpressureSuffix(module) + inSuffix, i1,
        backpressurePort);

  Value data, validity;
  if (body) {
    auto *terminator = body->getTerminator();
    ImplicitLocOpBuilder b(origPort.loc, terminator);
    auto unwrapped = Protocol::unwrap(
        b, b.getLoc(), terminator->getOperand(origPort.argNum), backpressure);
    data = unwrapped.first;
    validity = unwrapped.second;
  }

  // The data and forward validity signals leave the module.
  converter.createNewOutput(origPort, outSuffix, chanTy.getInner(), data,
                            dataPort);
  converter.createNewOutput(origPort,
                            Protocol::getValiditySuffix(module) + outSuffix, i1,
                            validity, validityPort);
}

template <typename Protocol>
void ScalarChannelPort<Protocol>::mapOutputSignals(
    OpBuilder &b, Operation *inst, Value, SmallVectorImpl<Value> &newOperands,
    ArrayRef<Backedge> newResults) {
  auto chanTy = cast<ChannelType>(origPort.type);
  auto [chan, backpressure] =
      Protocol::wrap(b, inst->getLoc(), chanTy, newResults[dataPort.argNum],
                     newResults[validityPort.argNum]);
  inst->getResult(origPort.argNum).replaceAllUsesWith(chan);
  if constexpr (Protocol::hasBackpressure)
    newOperands[backpressurePort.argNum] = backpressure;
}

//===----------------------------------------------------------------------===//
// ArrayChannelPort
//===----------------------------------------------------------------------===//

template <typename Protocol>
void ArrayChannelPort<Protocol>::buildInputSignals() {
  Operation *module = converter.getModule();
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);
  auto arrTy = cast<hw::ArrayType>(origPort.type);
  auto chanTy = cast<ChannelType>(arrTy.getElementType());
  size_t numElems = arrTy.getNumElements();
  auto dataArrTy = hw::ArrayType::get(chanTy.getInner(), numElems);
  auto validityArrTy = hw::ArrayType::get(i1, numElems);

  StringRef inSuffix = getStringAttributeOr(module, extModPortInSuffix, "");
  StringRef outSuffix = getStringAttributeOr(module, extModPortOutSuffix, "");

  // The data and forward validity signal arrays come into the module alongside
  // the data.
  Value dataArr =
      converter.createNewInput(origPort, inSuffix, dataArrTy, dataPort);
  Value validityArr = converter.createNewInput(
      origPort, Protocol::getValiditySuffix(module) + inSuffix, validityArrTy,
      validityPort);

  Value backpressureArr;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    // Recreate the original array of channels by wrapping each element's
    // lowered signals back into a channel.
    SmallVector<Value> chans, backpressures;
    for (size_t i = 0; i < numElems; ++i) {
      Value data = getArrayElement(b, b.getLoc(), dataArr, i);
      Value validity = getArrayElement(b, b.getLoc(), validityArr, i);
      auto [chan, bp] = Protocol::wrap(b, b.getLoc(), chanTy, data, validity);
      chans.push_back(chan);
      if constexpr (Protocol::hasBackpressure)
        backpressures.push_back(bp);
    }
    body->getArgument(origPort.argNum)
        .replaceAllUsesWith(packArray(b, b.getLoc(), chans));
    if (!backpressures.empty())
      backpressureArr = packArray(b, b.getLoc(), backpressures);
  }

  // The backpressure handshake signal array (if any) leaves the module.
  if constexpr (Protocol::hasBackpressure)
    converter.createNewOutput(
        origPort, Protocol::getBackpressureSuffix(module) + outSuffix,
        validityArrTy, backpressureArr, backpressurePort);
}

template <typename Protocol>
void ArrayChannelPort<Protocol>::mapInputSignals(
    OpBuilder &b, Operation *inst, Value, SmallVectorImpl<Value> &newOperands,
    ArrayRef<Backedge> newResults) {
  Location loc = inst->getLoc();
  Value chanArr = inst->getOperand(origPort.argNum);
  size_t numElems = cast<hw::ArrayType>(origPort.type).getNumElements();

  Value backpressureArr;
  if constexpr (Protocol::hasBackpressure)
    backpressureArr = newResults[backpressurePort.argNum];

  // Unwrap each channel element into its data and validity signals.
  SmallVector<Value> datas, validities;
  for (size_t i = 0; i < numElems; ++i) {
    Value chan = getArrayElement(b, loc, chanArr, i);
    Value backpressure;
    if constexpr (Protocol::hasBackpressure)
      backpressure = getArrayElement(b, loc, backpressureArr, i);
    auto [data, validity] = Protocol::unwrap(b, loc, chan, backpressure);
    datas.push_back(data);
    validities.push_back(validity);
  }
  newOperands[dataPort.argNum] = packArray(b, loc, datas);
  newOperands[validityPort.argNum] = packArray(b, loc, validities);
}

template <typename Protocol>
void ArrayChannelPort<Protocol>::buildOutputSignals() {
  Operation *module = converter.getModule();
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);
  auto arrTy = cast<hw::ArrayType>(origPort.type);
  auto chanTy = cast<ChannelType>(arrTy.getElementType());
  size_t numElems = arrTy.getNumElements();
  auto dataArrTy = hw::ArrayType::get(chanTy.getInner(), numElems);
  auto validityArrTy = hw::ArrayType::get(i1, numElems);

  StringRef inSuffix = getStringAttributeOr(module, extModPortInSuffix, "");
  StringRef outSuffix = getStringAttributeOr(module, extModPortOutSuffix, "");

  // The backpressure handshake signal array (if any) comes into the module.
  Value backpressureArr;
  if constexpr (Protocol::hasBackpressure)
    backpressureArr = converter.createNewInput(
        origPort, Protocol::getBackpressureSuffix(module) + inSuffix,
        validityArrTy, backpressurePort);

  Value dataArr, validityArr;
  if (body) {
    auto *terminator = body->getTerminator();
    ImplicitLocOpBuilder b(origPort.loc, terminator);
    Value chanArr = terminator->getOperand(origPort.argNum);
    // Unwrap each channel element into its data and validity signals.
    SmallVector<Value> datas, validities;
    for (size_t i = 0; i < numElems; ++i) {
      Value chan = getArrayElement(b, b.getLoc(), chanArr, i);
      Value backpressure;
      if constexpr (Protocol::hasBackpressure)
        backpressure = getArrayElement(b, b.getLoc(), backpressureArr, i);
      auto [data, validity] =
          Protocol::unwrap(b, b.getLoc(), chan, backpressure);
      datas.push_back(data);
      validities.push_back(validity);
    }
    dataArr = packArray(b, b.getLoc(), datas);
    validityArr = packArray(b, b.getLoc(), validities);
  }

  // The data and forward validity signal arrays leave the module.
  converter.createNewOutput(origPort, outSuffix, dataArrTy, dataArr, dataPort);
  converter.createNewOutput(origPort,
                            Protocol::getValiditySuffix(module) + outSuffix,
                            validityArrTy, validityArr, validityPort);
}

template <typename Protocol>
void ArrayChannelPort<Protocol>::mapOutputSignals(
    OpBuilder &b, Operation *inst, Value, SmallVectorImpl<Value> &newOperands,
    ArrayRef<Backedge> newResults) {
  Location loc = inst->getLoc();
  auto arrTy = cast<hw::ArrayType>(origPort.type);
  auto chanTy = cast<ChannelType>(arrTy.getElementType());
  size_t numElems = arrTy.getNumElements();

  Value dataArr = newResults[dataPort.argNum];
  Value validityArr = newResults[validityPort.argNum];

  // Wrap each element's data and validity signals back into a channel.
  SmallVector<Value> chans, backpressures;
  for (size_t i = 0; i < numElems; ++i) {
    Value data = getArrayElement(b, loc, dataArr, i);
    Value validity = getArrayElement(b, loc, validityArr, i);
    auto [chan, bp] = Protocol::wrap(b, loc, chanTy, data, validity);
    chans.push_back(chan);
    if (bp)
      backpressures.push_back(bp);
  }
  inst->getResult(origPort.argNum).replaceAllUsesWith(packArray(b, loc, chans));
  if constexpr (Protocol::hasBackpressure)
    newOperands[backpressurePort.argNum] = packArray(b, loc, backpressures);
}

namespace {
/// Convert all the ESI ports on modules to some lower construct. SV
/// interfaces for now on external modules, ready/valid to modules defined
/// internally. In the future, it may be possible to select a different
/// format.
struct ESIPortsPass : public circt::esi::impl::LowerESIPortsBase<ESIPortsPass> {
  void runOnOperation() override;

private:
  bool updateFunc(HWModuleExternOp mod);
  void updateInstance(HWModuleExternOp mod, InstanceOp inst);
  ESIHWBuilder *build;
};
} // anonymous namespace

/// Iterate through the `hw.module[.extern]`s and lower their ports.
void ESIPortsPass::runOnOperation() {
  ModuleOp top = getOperation();
  ESIHWBuilder b(top);
  build = &b;

  // Find all externmodules and try to modify them. Remember the modified
  // ones.
  DenseMap<SymbolRefAttr, HWModuleExternOp> externModsMutated;
  for (auto mod : top.getOps<HWModuleExternOp>())
    if (mod->hasAttrOfType<UnitAttr>(extModBundleSignalsAttrName) &&
        updateFunc(mod))
      externModsMutated[FlatSymbolRefAttr::get(mod)] = mod;

  // Find all instances and update them.
  top.walk([&externModsMutated, this](InstanceOp inst) {
    auto mapIter = externModsMutated.find(inst.getModuleNameAttr());
    if (mapIter != externModsMutated.end())
      updateInstance(mapIter->second, inst);
  });

  // Find all modules and run port conversion on them.
  circt::hw::InstanceGraph &instanceGraph =
      getAnalysis<circt::hw::InstanceGraph>();

  for (auto mod : top.getOps<HWMutableModuleLike>()) {
    if (failed(
            PortConverter<ESIPortConversionBuilder>(instanceGraph, mod).run()))
      return signalPassFailure();
  }

  build = nullptr;
}

/// Convert all input and output ChannelTypes into SV Interfaces. For inputs,
/// just switch the type to `ModportType`. For outputs, append a `ModportType`
/// to the inputs and remove the output channel from the results. Returns true
/// if 'mod' was updated. Delay updating the instances to amortize the IR walk
/// over all the module updates.
bool ESIPortsPass::updateFunc(HWModuleExternOp mod) {
  auto *ctxt = &getContext();

  bool updated = false;

  SmallVector<Attribute> newArgNames, newResultNames;
  SmallVector<Location> newArgLocs, newResultLocs;

  // Reconstruct the list of operand types, changing the type whenever an ESI
  // port is found.
  SmallVector<Type, 16> newArgTypes;
  size_t nextArgNo = 0;
  for (auto argTy : mod.getInputTypes()) {
    auto chanTy = dyn_cast<ChannelType>(argTy);
    newArgNames.push_back(mod.getInputNameAttr(nextArgNo));
    newArgLocs.push_back(mod.getInputLoc(nextArgNo));
    nextArgNo++;

    if (!chanTy) {
      newArgTypes.push_back(argTy);
      continue;
    }

    // When we find one, construct an interface, and add the 'source' modport
    // to the type list.
    auto iface = build->getOrConstructInterface(chanTy);
    newArgTypes.push_back(iface.getModportType(ESIHWBuilder::sourceStr));
    updated = true;
  }

  // Iterate through the results and append to one of the two below lists. The
  // first for non-ESI-ports. The second, ports which have been re-located to
  // an operand.
  SmallVector<Type, 8> newResultTypes;
  SmallVector<DictionaryAttr, 4> newResultAttrs;
  for (size_t resNum = 0, numRes = mod.getNumOutputPorts(); resNum < numRes;
       ++resNum) {
    Type resTy = mod.getOutputTypes()[resNum];
    auto chanTy = dyn_cast<ChannelType>(resTy);
    auto resNameAttr = mod.getOutputNameAttr(resNum);
    auto resLocAttr = mod.getOutputLoc(resNum);
    if (!chanTy) {
      newResultTypes.push_back(resTy);
      newResultNames.push_back(resNameAttr);
      newResultLocs.push_back(resLocAttr);
      continue;
    }

    // When we find one, construct an interface, and add the 'sink' modport to
    // the type list.
    sv::InterfaceOp iface = build->getOrConstructInterface(chanTy);
    sv::ModportType sinkPort = iface.getModportType(ESIHWBuilder::sinkStr);
    newArgTypes.push_back(sinkPort);
    newArgNames.push_back(resNameAttr);
    newArgLocs.push_back(resLocAttr);
    updated = true;
  }

  mod->removeAttr(extModBundleSignalsAttrName);
  if (!updated)
    return false;

  // Set the new types.
  auto newFuncType = FunctionType::get(ctxt, newArgTypes, newResultTypes);
  auto newModType =
      hw::detail::fnToMod(newFuncType, newArgNames, newResultNames);
  mod.setHWModuleType(newModType);
  mod.setInputLocs(newArgLocs);
  mod.setOutputLocs(newResultLocs);
  return true;
}

static StringRef getOperandName(Value operand) {
  if (BlockArgument arg = dyn_cast<BlockArgument>(operand)) {
    auto *op = arg.getParentBlock()->getParentOp();
    if (HWModuleLike mod = dyn_cast_or_null<HWModuleLike>(op))
      return mod.getInputName(arg.getArgNumber());
  } else {
    auto *srcOp = operand.getDefiningOp();
    if (auto instOp = dyn_cast<InstanceOp>(srcOp))
      return instOp.getInstanceName();

    if (auto srcName = srcOp->getAttrOfType<StringAttr>("name"))
      return srcName.getValue();
  }
  return "";
}

/// Create a reasonable name for a SV interface instance.
static std::string &constructInstanceName(Value operand, sv::InterfaceOp iface,
                                          std::string &name) {
  llvm::raw_string_ostream s(name);
  // Drop the "IValidReady_" part of the interface name.
  s << llvm::toLower(iface.getSymName()[12]) << iface.getSymName().substr(13);

  // Indicate to where the source is connected.
  if (operand.hasOneUse()) {
    Operation *dstOp = *operand.getUsers().begin();
    if (auto instOp = dyn_cast<InstanceOp>(dstOp))
      s << "To" << llvm::toUpper(instOp.getInstanceName()[0])
        << instOp.getInstanceName().substr(1);
    else if (auto dstName = dstOp->getAttrOfType<StringAttr>("name"))
      s << "To" << dstName.getValue();
  }

  // Indicate to where the sink is connected.
  StringRef operName = getOperandName(operand);
  if (!operName.empty())
    s << "From" << llvm::toUpper(operName[0]) << operName.substr(1);
  return s.str();
}

/// Update an instance of an updated module by adding `esi.(un)wrap.iface`
/// around the instance. Create a new instance at the end from the lists built
/// up before.
void ESIPortsPass::updateInstance(HWModuleExternOp mod, InstanceOp inst) {
  using namespace circt::sv;
  circt::ImplicitLocOpBuilder instBuilder(inst.getLoc(), inst);

  // op counter for error reporting purposes.
  size_t opNum = 0;
  // List of new operands.
  SmallVector<Value, 16> newOperands;

  // Fill the new operand list with old plain operands and mutated ones.
  std::string nameStringBuffer; // raw_string_ostream uses std::string.
  for (auto op : inst.getOperands()) {
    auto instChanTy = dyn_cast<ChannelType>(op.getType());
    if (!instChanTy) {
      newOperands.push_back(op);
      ++opNum;
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sourceStr) !=
        mod.getInputTypes()[opNum]) {
      inst.emitOpError("ESI ChannelType (operand #")
          << opNum << ") doesn't match module!";
      ++opNum;
      newOperands.push_back(op);
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.unwrap.iface` and the other end to the instance.
    auto ifaceInst =
        InterfaceInstanceOp::create(instBuilder, iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(op, iface, nameStringBuffer)));
    GetModportOp sinkModport =
        GetModportOp::create(instBuilder, ifaceInst, ESIHWBuilder::sinkStr);
    UnwrapSVInterfaceOp::create(instBuilder, op, sinkModport);
    GetModportOp sourceModport =
        GetModportOp::create(instBuilder, ifaceInst, ESIHWBuilder::sourceStr);
    // Finally, add the correct modport to the list of operands.
    newOperands.push_back(sourceModport);
  }

  // Go through the results and get both a list of the plain old values being
  // produced and their types.
  SmallVector<Value, 8> newResults;
  SmallVector<Type, 8> newResultTypes;
  for (size_t resNum = 0, numRes = inst.getNumResults(); resNum < numRes;
       ++resNum) {
    Value res = inst.getResult(resNum);
    auto instChanTy = dyn_cast<ChannelType>(res.getType());
    if (!instChanTy) {
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sinkStr) !=
        mod.getInputTypes()[opNum]) {
      inst.emitOpError("ESI ChannelType (result #")
          << resNum << ", operand #" << opNum << ") doesn't match module!";
      ++opNum;
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.wrap.iface` and the other end to the instance. Append it to the
    // operand list.
    auto ifaceInst =
        InterfaceInstanceOp::create(instBuilder, iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(res, iface, nameStringBuffer)));
    GetModportOp sourceModport =
        GetModportOp::create(instBuilder, ifaceInst, ESIHWBuilder::sourceStr);
    auto newChannel =
        WrapSVInterfaceOp::create(instBuilder, res.getType(), sourceModport);
    // Connect all the old users of the output channel with the newly
    // wrapped replacement channel.
    res.replaceAllUsesWith(newChannel);
    GetModportOp sinkModport =
        GetModportOp::create(instBuilder, ifaceInst, ESIHWBuilder::sinkStr);
    // And add the modport on the other side to the new operand list.
    newOperands.push_back(sinkModport);
  }

  // Create the new instance!
  auto newInst = hw::InstanceOp::create(
      instBuilder, mod, inst.getInstanceNameAttr(), newOperands,
      inst.getParameters(), inst.getInnerSymAttr());

  // Go through the old list of non-ESI result values, and replace them with
  // the new non-ESI results.
  for (size_t resNum = 0, numRes = newResults.size(); resNum < numRes;
       ++resNum) {
    newResults[resNum].replaceAllUsesWith(newInst.getResult(resNum));
  }
  // Erase the old instance!
  inst.erase();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPortLoweringPass() {
  return std::make_unique<ESIPortsPass>();
}
