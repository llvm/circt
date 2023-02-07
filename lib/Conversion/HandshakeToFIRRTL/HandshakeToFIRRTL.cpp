//===- HandshakeToFIRRTL.cpp - Translate Handshake into FIRRTL ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to FIRRTL Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HandshakeToFIRRTL.h"
#include "../PassDetail.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <set>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::firrtl;

using ValueVector = llvm::SmallVector<Value, 3>;
using ValueVectorList = std::vector<ValueVector>;
using NameUniquer = std::function<std::string(Operation *)>;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static void legalizeFModule(FModuleOp moduleOp) {
  SmallVector<Operation *, 8> connectOps;
  moduleOp.walk([&](ConnectOp op) { connectOps.push_back(op); });
  for (auto op : connectOps)
    op->moveBefore(&moduleOp.getBodyBlock()->back());
}

/// Return the number of bits needed to index the given number of values.
static size_t getNumIndexBits(uint64_t numValues) {
  return numValues > 1 ? llvm::Log2_64_Ceil(numValues) : 1;
}

/// Get the corresponding FIRRTL type given the built-in data type. Current
/// supported data types are integer (signed, unsigned, and signless), index,
/// and none.
static FIRRTLBaseType getFIRRTLType(Type type) {
  MLIRContext *context = type.getContext();
  return TypeSwitch<Type, FIRRTLBaseType>(type)
      .Case<IntegerType>([&](IntegerType integerType) -> FIRRTLBaseType {
        unsigned width = integerType.getWidth();

        switch (integerType.getSignedness()) {
        case IntegerType::Signed:
          return SIntType::get(context, width);
        case IntegerType::Unsigned:
          return UIntType::get(context, width);
        // ISSUE: How to handle signless integers? Should we use the
        // AsSIntPrimOp or AsUIntPrimOp to convert?
        case IntegerType::Signless:
          return UIntType::get(context, width);
        }
        llvm_unreachable("invalid IntegerType");
      })
      .Case<IndexType>([&](IndexType indexType) -> FIRRTLBaseType {
        // Currently we consider index type as 64-bits unsigned integer.
        unsigned width = indexType.kInternalStorageBitWidth;
        return UIntType::get(context, width);
      })
      .Case<TupleType>([&](TupleType tupleType) -> FIRRTLBaseType {
        using BundleElement = BundleType::BundleElement;
        llvm::SmallVector<BundleElement> elements;
        for (auto it : llvm::enumerate(tupleType.getTypes()))
          elements.emplace_back(BundleElement(
              StringAttr::get(context, llvm::formatv("field{0}", it.index())),
              false, getFIRRTLType(it.value())));
        return BundleType::get(context, elements);
      })
      .Default([&](Type) { return FIRRTLBaseType(); });
}

/// Creates a new FIRRTL bundle type based on an array of port infos.
static FIRRTLBaseType portInfosToBundleType(MLIRContext *ctx,
                                            ArrayRef<PortInfo> ports) {
  using BundleElement = BundleType::BundleElement;
  llvm::SmallVector<BundleElement, 4> elements;
  for (auto &port : ports) {
    elements.push_back(BundleElement(port.name,
                                     port.direction == Direction::Out,
                                     port.type.cast<FIRRTLBaseType>()));
  }
  return BundleType::get(ctx, elements);
}

/// Return a FIRRTL bundle type (with data, valid, and ready subfields) given a
/// standard data type.
static FIRRTLBaseType getBundleType(Type type) {
  // If the input is already converted to a bundle type elsewhere, itself will
  // be returned after cast.
  if (auto bundleType = type.dyn_cast<BundleType>())
    return bundleType;

  MLIRContext *context = type.getContext();
  using BundleElement = BundleType::BundleElement;
  llvm::SmallVector<BundleElement, 3> elements;

  // Add valid and ready subfield to the bundle.
  auto validId = StringAttr::get(context, "valid");
  auto readyId = StringAttr::get(context, "ready");
  auto signalType = UIntType::get(context, 1);
  elements.push_back(BundleElement(validId, false, signalType));
  elements.push_back(BundleElement(readyId, true, signalType));

  // Add data subfield to the bundle if dataType is not a null.
  auto dataType = getFIRRTLType(type);
  if (dataType) {
    auto dataId = StringAttr::get(context, "data");
    elements.push_back(BundleElement(dataId, false, dataType));
  }

  auto bundleType = BundleType::get(context, elements);
  return bundleType;
}

static bool isControlOp(Operation *op) {
  auto control = op->getAttrOfType<BoolAttr>("control");
  return control && control.getValue();
}

/// A class to be used with getPortInfoForOp. Provides an opaque interface for
/// generating the port names of an operation; handshake operations generate
/// names by the Handshake NamedIOInterface;  and other operations, such as
/// arith ops, are assigned default names.
class PortNameGenerator {
public:
  explicit PortNameGenerator(Operation *op) : builder(op->getContext()) {
    auto namedOpInterface = dyn_cast<handshake::NamedIOInterface>(op);
    if (namedOpInterface)
      inferFromNamedOpInterface(namedOpInterface);
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

  Builder builder;
  llvm::SmallVector<StringAttr> inputs;
  llvm::SmallVector<StringAttr> outputs;
};

/// Returns a vector of PortInfo's which defines the FIRRTL interface of the
/// to-be-converted op.
llvm::SmallVector<PortInfo>
getPortInfoForOp(ConversionPatternRewriter &rewriter, Operation *op) {
  llvm::SmallVector<PortInfo> ports;
  auto loc = op->getLoc();
  bool hasClock = op->hasTrait<mlir::OpTrait::HasClock>();
  PortNameGenerator portNames(op);

  // Add all inputs of oldOp.
  for (auto portType : llvm::enumerate(op->getOperandTypes())) {
    auto bundlePortType = getBundleType(portType.value());

    if (!bundlePortType)
      op->emitError("Unsupported data type. Supported data types: integer "
                    "(signed, unsigned, signless), index, none.");

    ports.push_back({portNames.inputName(portType.index()), bundlePortType,
                     Direction::In, StringAttr{}, loc});
  }

  // Add all outputs of oldOp.
  for (auto portType : llvm::enumerate(op->getResultTypes())) {
    auto bundlePortType = getBundleType(portType.value());

    if (!bundlePortType)
      op->emitError("Unsupported data type. Supported data types: integer "
                    "(signed, unsigned, signless), index, none.");

    ports.push_back({portNames.outputName(portType.index()), bundlePortType,
                     Direction::Out, StringAttr{}, loc});
  }

  // Add clock and reset signals.
  if (hasClock) {
    ports.push_back({rewriter.getStringAttr("clock"),
                     rewriter.getType<ClockType>(), Direction::In, StringAttr{},
                     loc});
    ports.push_back({rewriter.getStringAttr("reset"),
                     rewriter.getType<UIntType>(1), Direction::In, StringAttr{},
                     loc});
  }

  return ports;
}

/// Returns the bundle type associated with an external memory (memref
/// input argument). The bundle type is deduced from the handshake.extmemory
/// operator which references the memref input argument.
static FIRRTLBaseType getMemrefBundleType(ConversionPatternRewriter &rewriter,
                                          Value blockArg, bool flip) {
  assert(blockArg.getType().isa<MemRefType>() &&
         "expected blockArg to be a memref");

  auto extmemUsers = blockArg.getUsers();
  assert(std::distance(extmemUsers.begin(), extmemUsers.end()) == 1 &&
         "Expected a single user of an external memory");
  auto extmemOp = dyn_cast<ExternalMemoryOp>(*extmemUsers.begin());
  assert(extmemOp &&
         "Expected a handshake.extmemory to reference the memref argument");
  // Get a handle to the submodule which will wrap the external memory
  auto extmemPortInfo = getPortInfoForOp(rewriter, extmemOp);

  if (flip) {
    for (auto &pi : extmemPortInfo)
      pi.direction =
          pi.direction == Direction::In ? Direction::Out : Direction::In;
  }

  // Drop the first port info; this one will be a handshake associated with the
  // memref type.
  extmemPortInfo.erase(extmemPortInfo.begin());

  return portInfosToBundleType(rewriter.getContext(), extmemPortInfo);
}

static Value createConstantOp(FIRRTLBaseType opType, APInt value,
                              Location insertLoc, OpBuilder &builder) {
  assert(opType.isa<IntType>() && "can only create constants from IntTypes");
  if (auto intOpType = opType.dyn_cast<firrtl::IntType>()) {
    auto type = builder.getIntegerType(intOpType.getWidthOrSentinel(),
                                       intOpType.isSigned());
    return builder.create<firrtl::ConstantOp>(
        insertLoc, opType, builder.getIntegerAttr(type, value));
  }

  return Value();
}

/// Creates a Value that has an assigned zero value. For bundles, this
/// corresponds to assigning zero to each element recursively.
static Value createZeroDataConst(FIRRTLBaseType dataType, Location insertLoc,
                                 OpBuilder &builder) {
  return TypeSwitch<Type, Value>(dataType)
      .Case<IntType>([&](auto dataType) {
        return createConstantOp(dataType,
                                APInt(dataType.getBitWidthOrSentinel(), 0),
                                insertLoc, builder);
      })
      .Case<BundleType>([&](auto bundleType) {
        auto width = circt::firrtl::getBitWidth(bundleType);
        assert(width && "width must be inferred");
        auto zero =
            builder.create<firrtl::ConstantOp>(insertLoc, APSInt(*width, 0));
        return builder.create<BitCastOp>(insertLoc, bundleType, zero);
      })
      .Default([&](Type) -> Value { llvm_unreachable("Unknown type"); });
}

/// Transforms a FIRRTL data-carrying type to a builtin type.
static Type getBundleElementType(Type type) {
  if (auto intType = type.dyn_cast<firrtl::IntType>())
    return IntegerType::get(type.getContext(), intType.getWidthOrSentinel(),
                            intType.isSigned() ? IntegerType::Signed
                                               : IntegerType::Unsigned);
  auto bundleType = type.cast<BundleType>();
  SmallVector<Type> elementTypes;
  for (auto element : bundleType.getElements())
    elementTypes.push_back(getBundleElementType(element.type));
  return TupleType::get(type.getContext(), elementTypes);
}

/// Extracts the data-carrying type of bundle. This function assumes that the
/// bundle represents a handshaked input, therefore it looks at the data
/// element.
static Type getHandshakeBundleDataType(BundleType bundle) {
  if (auto dataType = bundle.getElementType("data"))
    return getBundleElementType(dataType);
  return NoneType::get(bundle.getContext());
}

/// Extracts the type of the data-carrying type of opType. If opType is a
/// bundle, getHandshakeBundleDataType extracts the data-carrying type, else,
/// assume that opType itself is the data-carrying type.
static Type getOperandDataType(Value op) {
  auto opType = op.getType();
  if (auto bundleType = opType.dyn_cast<BundleType>(); bundleType)
    return getHandshakeBundleDataType(bundleType);
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
  } else if (auto tupleType = type.dyn_cast<TupleType>()) {
    typeName += "_tuple";
    for (auto elementType : tupleType.getTypes())
      typeName += getTypeName(loc, elementType);
  }
  // FIRRTL types
  else if (type.isa<SIntType, UIntType>()) {
    if (auto sintType = type.dyn_cast<SIntType>(); sintType)
      typeName += "_si" + std::to_string(sintType.getWidthOrSentinel());
    else {
      auto uintType = type.cast<UIntType>();
      typeName += "_ui" + std::to_string(uintType.getWidthOrSentinel());
    }
  } else if (auto bundleType = type.dyn_cast<BundleType>()) {
    typeName += "_tuple";
    for (auto element : bundleType.getElements())
      typeName += getTypeName(loc, element.type);
  } else
    emitError(loc) << "unsupported data type '" << type << "'";

  return typeName;
}

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

/// Construct a name for creating FIRRTL sub-module.
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

    if (auto initValues = bufferOp.getInitValues()) {
      subModuleName += "_init";
      for (const Attribute e : *initValues) {
        assert(e.isa<IntegerAttr>());
        subModuleName +=
            "_" + std::to_string(e.dyn_cast<IntegerAttr>().getInt());
      }
    }
  }

  // Add control information.
  if (auto ctrlInterface = dyn_cast<handshake::ControlInterface>(oldOp);
      ctrlInterface && isControlOp(ctrlInterface)) {
    // Add some additional discriminating info for non-typed operations.
    subModuleName += "_" + std::to_string(oldOp->getNumOperands()) + "ins_" +
                     std::to_string(oldOp->getNumResults()) + "outs";
    subModuleName += "_ctrl";
  } else {
    assert((!inTypes.empty() || !outTypes.empty()) &&
           "Non-control operators must provide discriminating type info");
  }

  return subModuleName;
}

static std::pair<ArrayRef<Value>, ArrayRef<Value>>
splitArrayRef(ArrayRef<Value> ref, unsigned pos) {
  return {ref.take_front(pos), ref.drop_front(pos)};
}

/// Construct a tree of 1-bit muxes to multiplex arbitrary numbers of signals
/// using a binary-encoded select value.
static Value createMuxTree(ArrayRef<Value> inputs, Value select,
                           Location insertLoc, OpBuilder &builder) {
  Type muxDataType = inputs[0].getType();
  auto createBits = [&](Value select, size_t idx) {
    return builder.create<BitsPrimOp>(insertLoc, select, idx, idx);
  };

  std::function<Value(ArrayRef<Value>)> buildTreeRec =
      [&](ArrayRef<Value> operands) {
        assert(!operands.empty());
        Value retVal;
        // Base case
        if (operands.size() == 1)
          retVal = operands.front();
        else {
          // Mux case. In each layer we take a look at the significant bit wrt.
          // the # of operands provided, and split the operands at the log2
          // boundary. By doing so, every subsequent layer can the ignore the
          // MSBs considered by its preceding layer.
          unsigned selectBit = llvm::Log2_64_Ceil(operands.size()) - 1;
          // Split the operands at the selected index
          unsigned splitIdx = 1 << selectBit;

          auto [front, back] = splitArrayRef(operands, splitIdx);
          auto lowerTree = buildTreeRec(front);
          auto upperTree = buildTreeRec(back);
          auto layerSelect = createBits(select, selectBit);
          retVal = builder
                       .create<MuxPrimOp>(insertLoc, muxDataType, layerSelect,
                                          upperTree, lowerTree)
                       .getResult();
        }
        return retVal;
      };

  return buildTreeRec(inputs);
}

/// Construct a tree of 1-bit muxes to multiplex arbitrary numbers of signals
/// using a one-hot select value. Assumes select has a UIntType.
static Value createOneHotMuxTree(ArrayRef<Value> inputs, Value select,
                                 Location insertLoc,
                                 ConversionPatternRewriter &rewriter) {
  // Confirm the select input can be a one-hot encoding for the inputs.
  int32_t numInputs = inputs.size();
  assert(numInputs == select.getType().cast<UIntType>().getWidthOrSentinel() &&
         "one-hot select can't mux inputs");

  // Start the mux tree with zero value.
  auto inputType = inputs[0].getType().cast<FIRRTLBaseType>();
  auto inputWidth = inputType.getBitWidthOrSentinel();
  auto muxValue =
      createConstantOp(inputType, APInt(inputWidth, 0), insertLoc, rewriter);

  // Iteratively chain together muxes from the high bit to the low bit.
  for (size_t i = numInputs; i > 0; --i) {
    size_t inputIndex = i - 1;

    Value input = inputs[inputIndex];

    Value selectBit =
        rewriter.create<BitsPrimOp>(insertLoc, select, inputIndex, inputIndex);

    muxValue = rewriter.create<MuxPrimOp>(insertLoc, input.getType(), selectBit,
                                          input, muxValue);
  }

  return muxValue;
}

/// Construct a decoder by dynamically shifting 1 bit by the input amount.
/// See http://www.imm.dtu.dk/~masca/chisel-book.pdf Section 5.2.
static Value createDecoder(Value input, Location insertLoc,
                           ConversionPatternRewriter &rewriter) {
  auto *context = rewriter.getContext();

  // Get a type for a single unsigned bit.
  auto bitType = UIntType::get(context, 1);

  // Create a constant of for one bit.
  auto bit =
      rewriter.create<firrtl::ConstantOp>(insertLoc, bitType, APInt(1, 1));

  // Shift the bit dynamically by the input amount.
  auto shift = rewriter.create<DShlPrimOp>(insertLoc, bit, input);

  return shift;
}

/// Construct an arbiter based on a simple priority-encoding scheme. In addition
/// to returning the arbiter result, the index for each input is added to a
/// mapping for other lowerings to make use of.
static Value createPriorityArbiter(ArrayRef<Value> inputs, Value defaultValue,
                                   DenseMap<size_t, Value> &indexMapping,
                                   Location insertLoc,
                                   ConversionPatternRewriter &rewriter) {
  auto numInputs = inputs.size();
  auto indexType = UIntType::get(rewriter.getContext(), numInputs);
  auto priorityArb = defaultValue;

  for (size_t i = numInputs; i > 0; --i) {
    size_t inputIndex = i - 1;
    size_t oneHotIndex = size_t{1} << inputIndex;

    auto constIndex = createConstantOp(indexType, APInt(numInputs, oneHotIndex),
                                       insertLoc, rewriter);

    priorityArb = rewriter.create<MuxPrimOp>(
        insertLoc, indexType, inputs[inputIndex], constIndex, priorityArb);

    indexMapping[inputIndex] = constIndex;
  }

  return priorityArb;
}

/// Construct the logic to assign the ready outputs for ControlMergeOp and
/// MergeOp. The logic is identical for each output. If the fired value is
/// asserted, and the win value holds the output's index, that output is ready.
static void createMergeArgReady(ArrayRef<Value> outputs, Value fired,
                                Value winner, Value defaultValue,
                                DenseMap<size_t, Value> &indexMappings,
                                Location insertLoc,
                                ConversionPatternRewriter &rewriter) {
  auto bitType = fired.getType();
  auto indexType = winner.getType();

  Value winnerOrDefault = rewriter.create<MuxPrimOp>(
      insertLoc, indexType, fired, winner, defaultValue);

  for (size_t i = 0, e = outputs.size(); i != e; ++i) {
    auto constIndex = indexMappings[i];
    assert(constIndex && "index mapping not found");

    Value argReadyWire = rewriter.create<EQPrimOp>(insertLoc, bitType,
                                                   winnerOrDefault, constIndex);

    rewriter.create<ConnectOp>(insertLoc, outputs[i], argReadyWire);
  }
}

static void extractValues(ArrayRef<ValueVector *> valueVectors, size_t index,
                          SmallVectorImpl<Value> &result) {
  for (auto *elt : valueVectors)
    result.push_back((*elt)[index]);
}

//===----------------------------------------------------------------------===//
// FIRRTL Top-module Related Functions
//===----------------------------------------------------------------------===//

/// Currently we are not considering clock and registers, thus the generated
/// circuit is pure combinational logic. If graph cycle exists, at least one
/// buffer is required to be inserted for breaking the cycle, which will be
/// supported in the next patch.
template <typename TModuleOp>
static FailureOr<TModuleOp>
createTopModuleOp(handshake::FuncOp funcOp, unsigned numClocks,
                  ConversionPatternRewriter &rewriter, bool setFlattenAttr) {
  llvm::SmallVector<PortInfo> ports;

  // Add all inputs of funcOp.
  for (auto &[i, argType] :
       llvm::enumerate(funcOp.getFunctionType().getInputs())) {
    auto portName = funcOp.getArgName(i);
    FIRRTLBaseType bundlePortType;
    if (argType.template isa<MemRefType>()) {
      if (funcOp.isExternal())
        return funcOp.emitError(
            "external functions with memory arguments are not supported");

      BlockArgument arg = funcOp.getArgument(i);
      bundlePortType = getMemrefBundleType(rewriter, arg, /*flip=*/true);
    } else
      bundlePortType = getBundleType(argType);

    if (!bundlePortType)
      return funcOp.emitError(
          "Unsupported data type. Supported data types: integer "
          "(signed, unsigned, signless), index, none.");

    ports.push_back({portName, bundlePortType, Direction::In, StringAttr{},
                     funcOp.getLoc()});
  }

  auto funcLoc = funcOp.getLoc();

  // Add all outputs of funcOp.
  for (auto portType : llvm::enumerate(funcOp.getResultTypes())) {
    auto portName = funcOp.getResName(portType.index());
    auto bundlePortType = getBundleType(portType.value());

    if (!bundlePortType)
      return funcOp.emitError(
          "Unsupported data type. Supported data types: integer "
          "(signed, unsigned, signless), index, none.");

    ports.push_back(
        {portName, bundlePortType, Direction::Out, StringAttr{}, funcLoc});
  }

  // Add clock and reset signals.
  if (numClocks == 1) {
    ports.push_back({rewriter.getStringAttr("clock"),
                     rewriter.getType<ClockType>(), Direction::In, StringAttr{},
                     funcLoc});
    ports.push_back({rewriter.getStringAttr("reset"),
                     rewriter.getType<UIntType>(1), Direction::In, StringAttr{},
                     funcLoc});
  } else if (numClocks > 1) {
    for (unsigned i = 0; i < numClocks; ++i) {
      auto clockName = "clock" + std::to_string(i);
      auto resetName = "reset" + std::to_string(i);
      ports.push_back({rewriter.getStringAttr(clockName),
                       rewriter.getType<ClockType>(), Direction::In,
                       StringAttr{}, funcLoc});
      ports.push_back({rewriter.getStringAttr(resetName),
                       rewriter.getType<UIntType>(1), Direction::In,
                       StringAttr{}, funcLoc});
    }
  }
  // Create a FIRRTL module and inline the funcOp into the FIRRTL module.
  auto topModuleOp = rewriter.create<TModuleOp>(
      funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()), ports);

  if (setFlattenAttr)
    topModuleOp->setAttr(
        "annotations",
        rewriter.getArrayAttr(rewriter.getDictionaryAttr(
            llvm::SmallVector<NamedAttribute>{rewriter.getNamedAttr(
                "class", rewriter.getStringAttr(
                             "firrtl.transforms.FlattenAnnotation"))})));

  return topModuleOp;
}

/// Inlines the region of the handshake function into the FIRRTL module.
static void inlineFuncRegion(handshake::FuncOp funcOp, FModuleOp topModuleOp,
                             ConversionPatternRewriter &rewriter) {

  rewriter.inlineRegionBefore(funcOp.getBody(), topModuleOp.getBody(),
                              topModuleOp.getBody().end());

  // In the following section, we manually merge the two regions and manually
  // replace arguments. This is an alternative to using rewriter.mergeBlocks; we
  // do this to ensure that argument SSA values are replaced instantly, instead
  // of late, as would be the case for mergeBlocks.

  // Merge the second block (inlined from funcOp) of the top-module into the
  // entry block.
  auto &blockIterator = topModuleOp.getBody().getBlocks();
  Block *entryBlock = &blockIterator.front();
  Block *secondBlock = &*std::next(blockIterator.begin());

  // Replace uses of each argument of the second block with the corresponding
  // argument of the entry block.
  for (auto &oldArg : enumerate(secondBlock->getArguments()))
    oldArg.value().replaceAllUsesWith(entryBlock->getArgument(oldArg.index()));

  // Move all operations of the second block to the entry block.
  entryBlock->getOperations().splice(entryBlock->end(),
                                     secondBlock->getOperations());
  rewriter.eraseBlock(secondBlock);
}

//===----------------------------------------------------------------------===//
// FIRRTL Sub-module Related Functions
//===----------------------------------------------------------------------===//

/// Check whether a submodule with the same name has been created elsewhere in
/// the FIRRTL circt. Return the matched submodule if true, otherwise return
/// nullptr.
///
static FModuleLike checkSubModuleOp(CircuitOp circuitOp, StringRef modName) {
  return circuitOp.lookupSymbol<FModuleLike>(modName);
}

static FModuleLike checkSubModuleOp(CircuitOp circuitOp, Operation *oldOp) {
  auto moduleOp = checkSubModuleOp(circuitOp, getSubModuleName(oldOp));

  if (isa<handshake::InstanceOp>(oldOp))
    assert(moduleOp &&
           "handshake.instance target modules should always have been lowered "
           "before the modules that reference them!");
  return moduleOp;
}

/// All standard expressions and handshake elastic components will be converted
/// to a FIRRTL sub-module and be instantiated in the top-module.
static FModuleOp createSubModuleOp(FModuleOp topModuleOp, Operation *oldOp,
                                   ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPoint(topModuleOp);
  auto ports = getPortInfoForOp(rewriter, oldOp);
  return rewriter.create<FModuleOp>(
      topModuleOp.getLoc(), rewriter.getStringAttr(getSubModuleName(oldOp)),
      ports);
}

/// Extract all subfields of all ports of the sub-module.
static ValueVectorList extractSubfields(FModuleOp subModuleOp,
                                        Location insertLoc,
                                        ConversionPatternRewriter &rewriter) {
  ValueVectorList portList;
  for (auto &arg : subModuleOp.getArguments()) {
    ValueVector subfields;
    auto type = arg.getType().cast<FIRRTLBaseType>();
    if (auto bundleType = type.dyn_cast<BundleType>()) {
      // Extract all subfields of all bundle ports.
      for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i) {
        subfields.push_back(rewriter.create<SubfieldOp>(insertLoc, arg, i));
      }
    } else if (type.isa<ClockType>()) {
      // Extract clock signals.
      subfields.push_back(arg);
    } else if (auto intType = type.dyn_cast<UIntType>()) {
      // Extract reset signals.
      if (intType.getWidthOrSentinel() == 1)
        subfields.push_back(arg);
    }
    portList.push_back(subfields);
  }

  return portList;
}

//===----------------------------------------------------------------------===//
// Standard Expression Builder class
//===----------------------------------------------------------------------===//

namespace {
class StdExprBuilder : public StdExprVisitor<StdExprBuilder, bool> {
public:
  StdExprBuilder(ValueVectorList portList, Location insertLoc,
                 ConversionPatternRewriter &rewriter)
      : portList(portList), insertLoc(insertLoc), rewriter(rewriter) {}
  using StdExprVisitor::visitStdExpr;

  template <typename OpType, bool fstOpSigned = false, bool sndOpSigned = false>
  void buildBinaryLogic();

  bool visitInvalidOp(Operation *op) { return false; }

  bool visitStdExpr(arith::CmpIOp op);
  bool visitStdExpr(arith::ExtUIOp op);
  bool visitStdExpr(arith::ExtSIOp op);
  bool visitStdExpr(arith::TruncIOp op);
  bool visitStdExpr(arith::IndexCastOp op);

#define HANDLE(OPTYPE, FIRRTLTYPE)                                             \
  bool visitStdExpr(OPTYPE op) { return buildBinaryLogic<FIRRTLTYPE>(), true; }

  HANDLE(arith::AddIOp, AddPrimOp);
  HANDLE(arith::SubIOp, SubPrimOp);
  HANDLE(arith::MulIOp, MulPrimOp);
  HANDLE(arith::DivUIOp, DivPrimOp);
  HANDLE(arith::RemUIOp, RemPrimOp);
  HANDLE(arith::XOrIOp, XorPrimOp);
  HANDLE(arith::AndIOp, AndPrimOp);
  HANDLE(arith::OrIOp, OrPrimOp);
  HANDLE(arith::ShLIOp, DShlPrimOp);
  HANDLE(arith::ShRUIOp, DShrPrimOp);
#undef HANDLE

#define HANDLE_SIGNED(OPTYPE, FIRRTLTYPE, sndOpSigned)                         \
  bool visitStdExpr(OPTYPE op) {                                               \
    return buildBinaryLogic<FIRRTLTYPE, true, sndOpSigned>(), true;            \
  }
  HANDLE_SIGNED(arith::DivSIOp, DivPrimOp, true);
  HANDLE_SIGNED(arith::RemSIOp, RemPrimOp, true);
  HANDLE_SIGNED(arith::ShRSIOp, DShrPrimOp, false);
#undef HANDLE_SIGNED

  template <bool isSignedOp = false>
  bool buildSignExtendOp(unsigned dstWidth);
  bool buildTruncateOp(unsigned dstWidth);

private:
  ValueVectorList portList;
  Location insertLoc;
  ConversionPatternRewriter &rewriter;
};
} // namespace

bool StdExprBuilder::visitStdExpr(arith::CmpIOp op) {
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return buildBinaryLogic<EQPrimOp>(), true;
  case arith::CmpIPredicate::ne:
    return buildBinaryLogic<NEQPrimOp>(), true;
  case arith::CmpIPredicate::slt:
    return buildBinaryLogic<LTPrimOp, true, true>(), true;
  case arith::CmpIPredicate::ult:
    return buildBinaryLogic<LTPrimOp>(), true;
  case arith::CmpIPredicate::sle:
    return buildBinaryLogic<LEQPrimOp, true, true>(), true;
  case arith::CmpIPredicate::ule:
    return buildBinaryLogic<LEQPrimOp>(), true;
  case arith::CmpIPredicate::sgt:
    return buildBinaryLogic<GTPrimOp, true, true>(), true;
  case arith::CmpIPredicate::ugt:
    return buildBinaryLogic<GTPrimOp>(), true;
  case arith::CmpIPredicate::sge:
    return buildBinaryLogic<GEQPrimOp, true, true>(), true;
  case arith::CmpIPredicate::uge:
    return buildBinaryLogic<GEQPrimOp>(), true;
  }
  llvm_unreachable("invalid CmpIOp");
}

template <bool isSignedOp>
bool StdExprBuilder::buildSignExtendOp(unsigned dstWidth) {
  ValueVector arg0Subfield = portList[0];
  ValueVector resultSubfields = portList[1];

  Value arg0Valid = arg0Subfield[0];
  Value arg0Ready = arg0Subfield[1];
  Value arg0Data = arg0Subfield[2];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  if (isSignedOp)
    arg0Data = rewriter.create<AsSIntPrimOp>(insertLoc, arg0Data);

  Value resultDataOp =
      rewriter.create<PadPrimOp>(insertLoc, arg0Data, dstWidth);

  if (isSignedOp)
    resultDataOp = rewriter.create<AsUIntPrimOp>(insertLoc, resultDataOp);

  rewriter.create<ConnectOp>(insertLoc, resultData, resultDataOp);

  // Generate valid signal.
  rewriter.create<ConnectOp>(insertLoc, resultValid, arg0Valid);

  // Generate ready signal.
  auto argReadyOp = rewriter.create<AndPrimOp>(insertLoc, resultReady.getType(),
                                               resultReady, arg0Valid);
  rewriter.create<ConnectOp>(insertLoc, arg0Ready, argReadyOp);
  return true;
}

bool StdExprBuilder::buildTruncateOp(unsigned int dstWidth) {
  ValueVector arg0Subfield = portList[0];
  ValueVector resultSubfields = portList[1];

  Value arg0Valid = arg0Subfield[0];
  Value arg0Ready = arg0Subfield[1];
  Value arg0Data = arg0Subfield[2];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  Value resultDataOp =
      rewriter.create<BitsPrimOp>(insertLoc, arg0Data, dstWidth - 1, 0);
  rewriter.create<ConnectOp>(insertLoc, resultData, resultDataOp);

  // Generate valid signal.
  rewriter.create<ConnectOp>(insertLoc, resultValid, arg0Valid);

  // Generate ready signal.
  auto argReadyOp = rewriter.create<AndPrimOp>(insertLoc, resultReady.getType(),
                                               resultReady, arg0Valid);
  rewriter.create<ConnectOp>(insertLoc, arg0Ready, argReadyOp);
  return true;
}

bool StdExprBuilder::visitStdExpr(arith::ExtUIOp op) {
  return buildSignExtendOp(getFIRRTLType(getOperandDataType(op.getOperand()))
                               .getBitWidthOrSentinel());
}

bool StdExprBuilder::visitStdExpr(arith::ExtSIOp op) {
  return buildSignExtendOp<true>(
      getFIRRTLType(getOperandDataType(op.getOperand()))
          .getBitWidthOrSentinel());
}

bool StdExprBuilder::visitStdExpr(arith::TruncIOp op) {
  return buildTruncateOp(getFIRRTLType(getOperandDataType(op.getResult()))
                             .getBitWidthOrSentinel());
}

bool StdExprBuilder::visitStdExpr(arith::IndexCastOp op) {
  FIRRTLBaseType sourceType =
      getFIRRTLType(getOperandDataType(op.getOperand()));
  FIRRTLBaseType targetType = getFIRRTLType(getOperandDataType(op.getResult()));
  unsigned targetBits = targetType.getBitWidthOrSentinel();
  unsigned sourceBits = sourceType.getBitWidthOrSentinel();
  return (targetBits < sourceBits ? buildTruncateOp(targetBits)
                                  : buildSignExtendOp(targetBits));
}

/// Please refer to simple_addi.mlir test case.
template <typename OpType, bool fstOpSigned, bool sndOpSigned>
void StdExprBuilder::buildBinaryLogic() {
  ValueVector arg0Subfield = portList[0];
  ValueVector arg1Subfield = portList[1];
  ValueVector resultSubfields = portList[2];

  Value arg0Valid = arg0Subfield[0];
  Value arg0Ready = arg0Subfield[1];
  Value arg0Data = arg0Subfield[2];
  Value arg1Valid = arg1Subfield[0];
  Value arg1Ready = arg1Subfield[1];
  Value arg1Data = arg1Subfield[2];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  if (fstOpSigned)
    arg0Data = rewriter.create<AsSIntPrimOp>(insertLoc, arg0Data);

  if (sndOpSigned)
    arg1Data = rewriter.create<AsSIntPrimOp>(insertLoc, arg1Data);

  // Carry out the binary operation.
  Value resultDataOp = rewriter.create<OpType>(insertLoc, arg0Data, arg1Data);
  auto resultTy = resultDataOp.getType().cast<FIRRTLBaseType>();

  // Truncate the result type down if needed.
  auto resultWidth = resultData.getType()
                         .cast<FIRRTLBaseType>()
                         .getPassiveType()
                         .getBitWidthOrSentinel();

  if (resultWidth < resultTy.getBitWidthOrSentinel()) {
    resultDataOp = rewriter.create<BitsPrimOp>(insertLoc, resultDataOp,
                                               resultWidth - 1, 0);
  } else if (fstOpSigned) {
    // BitsPrimOp already casts to correct type, thus only do this when no
    // BitsPrimOp was created
    resultDataOp = rewriter.create<AsUIntPrimOp>(insertLoc, resultDataOp);
  }

  rewriter.create<ConnectOp>(insertLoc, resultData, resultDataOp);

  // Generate valid signal.
  auto resultValidOp = rewriter.create<AndPrimOp>(
      insertLoc, arg0Valid.getType(), arg0Valid, arg1Valid);
  rewriter.create<ConnectOp>(insertLoc, resultValid, resultValidOp);

  // Generate ready signals.
  auto argReadyOp = rewriter.create<AndPrimOp>(insertLoc, resultReady.getType(),
                                               resultReady, resultValidOp);
  rewriter.create<ConnectOp>(insertLoc, arg0Ready, argReadyOp);
  rewriter.create<ConnectOp>(insertLoc, arg1Ready, argReadyOp);
}

//===----------------------------------------------------------------------===//
// Handshake Builder class
//===----------------------------------------------------------------------===//

namespace {
class HandshakeBuilder : public HandshakeVisitor<HandshakeBuilder, bool> {
public:
  HandshakeBuilder(CircuitOp circuit, ValueVectorList portList,
                   Location insertLoc, ConversionPatternRewriter &rewriter)
      : circuit(circuit), portList(portList), insertLoc(insertLoc),
        rewriter(rewriter) {}
  using HandshakeVisitor::visitHandshake;

  bool visitInvalidOp(Operation *op) { return false; }

  bool visitHandshake(handshake::BranchOp op);
  bool visitHandshake(BufferOp op);
  bool visitHandshake(ConditionalBranchOp op);
  bool visitHandshake(handshake::ConstantOp op);
  bool visitHandshake(ControlMergeOp op);
  bool visitHandshake(ForkOp op);
  bool visitHandshake(JoinOp op);
  bool visitHandshake(LazyForkOp op);
  bool visitHandshake(handshake::LoadOp op);
  bool visitHandshake(MemoryOp op);
  bool visitHandshake(ExternalMemoryOp op);
  bool visitHandshake(MergeOp op);
  bool visitHandshake(MuxOp op);
  bool visitHandshake(handshake::SelectOp op);
  bool visitHandshake(SinkOp op);
  bool visitHandshake(SourceOp op);
  bool visitHandshake(SyncOp op);
  bool visitHandshake(handshake::StoreOp op);
  bool visitHandshake(PackOp op);
  bool visitHandshake(UnpackOp op);

  bool buildJoinLogic(SmallVector<ValueVector *, 4> inputs,
                      ValueVector *output);

  bool buildForkLogic(ValueVector *input, SmallVector<ValueVector *, 4> outputs,
                      Value clock, Value reset, bool isControl);

  // Builds a tree by chaining together the inputs with the specified OpType and
  // connecting the resulting value to the specified output. Also returns the
  // result value for convenience to the surrounding logic that may want to
  // re-use it.
  template <typename OpType>
  Value buildReductionTree(ArrayRef<Value> inputs, Value output);

  void buildAllReadyLogic(SmallVector<ValueVector *, 4> inputs,
                          ValueVector *output, Value condition);

  void buildControlBufferLogic(Value predValid, Value predReady,
                               Value succValid, Value succReady, Value clock,
                               Value reset, Value predData = nullptr,
                               Value succData = nullptr);
  void buildDataBufferLogic(Value predValid, Value validReg, Value predReady,
                            Value succReady, Value predData, Value dataReg);
  bool buildSeqBufferLogic(int64_t numStage, ValueVector *input,
                           ValueVector *output, Value clock, Value reset,
                           bool isControl, ArrayRef<int64_t> initValues = {});
  bool buildFIFOBufferLogic(int64_t numStage, ValueVector *input,
                            ValueVector *output, Value clock, Value reset,
                            bool isControl);

private:
  CircuitOp circuit;
  ValueVectorList portList;
  Location insertLoc;
  ConversionPatternRewriter &rewriter;
};
} // namespace

/// Please refer to test_sink.mlir test case.
bool HandshakeBuilder::visitHandshake(SinkOp op) {
  ValueVector argSubfields = portList.front();
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];

  // A Sink operation is always ready to accept tokens.
  auto signalType = argValid.getType().cast<FIRRTLBaseType>();
  Value highSignal =
      createConstantOp(signalType, APInt(1, 1), insertLoc, rewriter);
  rewriter.create<ConnectOp>(insertLoc, argReady, highSignal);

  rewriter.eraseOp(argValid.getDefiningOp());

  if (isControlOp(op))
    return true;

  // Non-control sink; must also have a data operand.
  assert(argSubfields.size() >= 3 &&
         "expected a data operand to a non-control sink op");
  Value argData = argSubfields[2];
  rewriter.eraseOp(argData.getDefiningOp());
  return true;
}

bool HandshakeBuilder::visitHandshake(SourceOp op) {
  ValueVector argSubfields = portList.front();
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];

  // A Source operation is always ready to provide tokens.
  auto signalType = argValid.getType().cast<FIRRTLBaseType>();
  Value highSignal =
      createConstantOp(signalType, APInt(1, 1), insertLoc, rewriter);
  rewriter.create<ConnectOp>(insertLoc, argValid, highSignal);

  rewriter.eraseOp(argReady.getDefiningOp());

  assert(isControlOp(op) && "source op provide control-only tokens");
  return true;
}

bool HandshakeBuilder::buildJoinLogic(SmallVector<ValueVector *, 4> inputs,
                                      ValueVector *output) {
  if (output == nullptr)
    return false;

  for (auto *input : inputs)
    if (input == nullptr)
      return false;

  // Unpack the output subfields.
  ValueVector outputSubfields = *output;

  // The output is triggered only after all inputs are valid.
  SmallVector<Value, 4> inputValids;
  extractValues(inputs, 0, inputValids);
  auto tmpValid =
      buildReductionTree<AndPrimOp>(inputValids, outputSubfields[0]);

  // The input will be ready to accept new token when old token is sent out.
  buildAllReadyLogic(inputs, output, tmpValid);

  return true;
}

template <typename OpType>
Value HandshakeBuilder::buildReductionTree(ArrayRef<Value> inputs,
                                           Value output) {
  size_t inputSize = inputs.size();
  assert(inputSize && "must pass inputs to reduce");

  auto tmpValue = inputs[0];

  for (size_t i = 1; i < inputSize; ++i)
    tmpValue = rewriter.create<OpType>(insertLoc, tmpValue.getType(), inputs[i],
                                       tmpValue);

  rewriter.create<ConnectOp>(insertLoc, output, tmpValue);

  return tmpValue;
}

void HandshakeBuilder::buildAllReadyLogic(SmallVector<ValueVector *, 4> inputs,
                                          ValueVector *output,
                                          Value condition) {
  auto outputSubfields = *output;
  auto outputReady = outputSubfields[1];

  auto validAndReady = rewriter.create<AndPrimOp>(
      insertLoc, outputReady.getType(), outputReady, condition);

  for (unsigned i = 0, e = inputs.size(); i < e; ++i) {
    auto currentInput = *inputs[i];
    auto inputReady = currentInput[1];
    rewriter.create<ConnectOp>(insertLoc, inputReady, validAndReady);
  }
}

/// Currently only support {control = true}.
/// Please refer to test_join.mlir test case.
bool HandshakeBuilder::visitHandshake(JoinOp op) {
  auto output = &portList.back();

  // Collect all input ports.
  SmallVector<ValueVector *, 4> inputs;
  for (unsigned i = 0, e = portList.size() - 1; i < e; ++i)
    inputs.push_back(&portList[i]);

  return buildJoinLogic(inputs, output);
}

// Joins all the input control signals and connects the resulting control
// signals in a fork like manner to the outputs. Data logic is forwarded
// directly between in- and outputs.
bool HandshakeBuilder::visitHandshake(SyncOp op) {
  size_t numRes = op->getNumResults();
  unsigned portNum = portList.size();
  assert(portNum == 2 * numRes + 2);

  // Create wires that will be used to connect the join and the fork logic
  auto bitType = UIntType::get(op->getContext(), 1);
  ValueVector connector;
  connector.push_back(rewriter.create<WireOp>(insertLoc, bitType, "allValid"));
  connector.push_back(rewriter.create<WireOp>(insertLoc, bitType, "allReady"));

  // Collect all input ports.
  SmallVector<ValueVector *, 4> inputs;
  for (unsigned i = 0, e = numRes; i < e; ++i)
    inputs.push_back(&portList[i]);

  // Collect all output ports.
  SmallVector<ValueVector *, 4> outputs;
  for (unsigned i = numRes, e = 2 * numRes; i < e; ++i)
    outputs.push_back(&portList[i]);

  // connect data ports
  for (auto [in, out] : llvm::zip(inputs, outputs)) {
    if (in->size() == 2)
      continue;

    rewriter.create<ConnectOp>(insertLoc, (*out)[2], (*in)[2]);
  }

  if (!buildJoinLogic(inputs, &connector))
    return false;

  // The clock and reset signals will be used for registers.
  auto clock = portList[portNum - 2][0];
  auto reset = portList[portNum - 1][0];

  // The state-keeping fork logic is required here, as the circuit isn't allowed
  // to wait for all the consumers to be ready.
  // Connecting the ready signals of the outputs to their corresponding valid
  // signals leads to combinatorial cycles. The paper which introduced
  // compositional dataflow circuits explicitly mentions this limitation:
  // http://arcade.cs.columbia.edu/df-memocode17.pdf
  return buildForkLogic(&connector, outputs, clock, reset, true);
}

/// Please refer to test_mux.mlir test case.
/// Lowers the MuxOp into primitive FIRRTL ops.
/// See http://www.cs.columbia.edu/~sedwards/papers/edwards2019compositional.pdf
/// Section 3.3.
bool HandshakeBuilder::visitHandshake(MuxOp op) {
  ValueVector selectSubfields = portList.front();
  Value selectValid = selectSubfields[0];
  Value selectReady = selectSubfields[1];
  Value selectData = selectSubfields[2];

  ValueVector resultSubfields = portList.back();
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData;
  if (!isControlOp(op))
    resultData = resultSubfields[2];

  // Walk through each arg data to collect the subfields.
  SmallVector<Value, 4> argValid;
  SmallVector<Value, 4> argReady;
  SmallVector<Value, 4> argData;
  for (unsigned i = 1, e = portList.size() - 1; i < e; ++i) {
    ValueVector argSubfields = portList[i];
    argValid.push_back(argSubfields[0]);
    argReady.push_back(argSubfields[1]);
    if (!isControlOp(op))
      argData.push_back(argSubfields[2]);
  }

  if (!isControlOp(op)) {
    // Mux the arg data.
    auto muxedData = createMuxTree(argData, selectData, insertLoc, rewriter);

    // Connect the selected data signal to the result data.
    rewriter.create<ConnectOp>(insertLoc, resultData, muxedData);
  }

  // Mux the arg valids.
  auto muxedValid = createMuxTree(argValid, selectData, insertLoc, rewriter);

  // And that with the select valid.
  auto muxedAndSelectValid = rewriter.create<AndPrimOp>(
      insertLoc, muxedValid.getType(), muxedValid, selectValid);

  // Connect that to the result valid.
  rewriter.create<ConnectOp>(insertLoc, resultValid, muxedAndSelectValid);

  // And the result valid with the result ready.
  auto resultValidAndReady =
      rewriter.create<AndPrimOp>(insertLoc, muxedAndSelectValid.getType(),
                                 muxedAndSelectValid, resultReady);

  // Connect that to the select ready.
  rewriter.create<ConnectOp>(insertLoc, selectReady, resultValidAndReady);

  // Since addresses coming from Handshake are IndexType and have a hardcoded
  // 64-bit width in this pass, we may need to truncate down to the actual
  // width used to index into the decoder.
  size_t bitsNeeded = getNumIndexBits(argValid.size());
  size_t selectBits =
      selectData.getType().cast<FIRRTLBaseType>().getBitWidthOrSentinel();

  if (selectBits > bitsNeeded) {
    auto tailAmount = selectBits - bitsNeeded;
    auto tailType = UIntType::get(op.getContext(), bitsNeeded);
    selectData = rewriter.create<TailPrimOp>(insertLoc, tailType, selectData,
                                             tailAmount);
  }

  // Create a decoder for the select data.
  auto decodedSelect = createDecoder(selectData, insertLoc, rewriter);

  // Walk through each input.
  for (unsigned i = 0, e = argValid.size(); i != e; ++i) {
    // Select the bit for this arg.
    auto oneHot = rewriter.create<BitsPrimOp>(insertLoc, decodedSelect, i, i);

    // And that with the result valid and ready.
    auto oneHotAndResultValidAndReady = rewriter.create<AndPrimOp>(
        insertLoc, oneHot.getType(), oneHot, resultValidAndReady);

    // Connect that to this arg ready.
    rewriter.create<ConnectOp>(insertLoc, argReady[i],
                               oneHotAndResultValidAndReady);
  }

  return true;
}

bool HandshakeBuilder::visitHandshake(handshake::SelectOp op) {
  ValueVector selectSubfields = portList[0];
  Value selectValid = selectSubfields[0];
  Value selectReady = selectSubfields[1];
  Value selectData = selectSubfields[2];

  ValueVector resultSubfields = portList[3];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  ValueVector trueSubfields = portList[1];
  Value trueValid = trueSubfields[0];
  Value trueReady = trueSubfields[1];
  Value trueData = trueSubfields[2];

  ValueVector falseSubfields = portList[2];
  Value falseValid = falseSubfields[0];
  Value falseReady = falseSubfields[1];
  Value falseData = falseSubfields[2];

  auto bitType = UIntType::get(rewriter.getContext(), 1);

  // Mux the true and false data.
  auto muxedData =
      createMuxTree({falseData, trueData}, selectData, insertLoc, rewriter);

  // Connect the selected data signal to the result data.
  rewriter.create<ConnectOp>(insertLoc, resultData, muxedData);

  // 'and' the arg valids and select valid
  Value allValid = rewriter.create<WireOp>(insertLoc, bitType, "allValid");
  buildReductionTree<AndPrimOp>({trueValid, falseValid, selectValid}, allValid);

  // Connect that to the result valid.
  rewriter.create<ConnectOp>(insertLoc, resultValid, allValid);

  // 'and' the result valid with the result ready.
  auto resultValidAndReady =
      rewriter.create<AndPrimOp>(insertLoc, bitType, allValid, resultReady);

  // Connect that to the 'ready' signal of all inputs. This implies that all
  // inputs + select is transacted when all are valid (and the output is ready),
  // but only the selected data is forwarded.
  rewriter.create<ConnectOp>(insertLoc, selectReady, resultValidAndReady);
  rewriter.create<ConnectOp>(insertLoc, trueReady, resultValidAndReady);
  rewriter.create<ConnectOp>(insertLoc, falseReady, resultValidAndReady);

  return true;
}

/// Please refer to test_merge.mlir test case.
/// Lowers the MergeOp into primitive FIRRTL ops. This is a simplification of
/// the ControlMergeOp lowering, since it doesn't need to wait for more than one
/// output to become ready.
bool HandshakeBuilder::visitHandshake(MergeOp op) {
  auto *context = rewriter.getContext();

  size_t numPorts = portList.size();
  size_t numInputs = numPorts - 1;

  // The last output has the result's ready, valid, and data signal.
  ValueVector resultSubfields = portList[numInputs];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData;

  if (!isControlOp(op))
    resultData = resultSubfields[2];

  // Walk through each arg data to collect the subfields.
  SmallVector<Value, 4> argValid;
  SmallVector<Value, 4> argReady;
  SmallVector<Value, 4> argData;
  for (size_t i = 0; i < numInputs; ++i) {
    ValueVector argSubfields = portList[i];
    argValid.push_back(argSubfields[0]);
    argReady.push_back(argSubfields[1]);
    if (!isControlOp(op))
      argData.push_back(argSubfields[2]);
  }

  // Define some common types and values that will be used.
  auto bitType = UIntType::get(context, 1);
  auto indexType = UIntType::get(context, numInputs);
  auto noWinner =
      createConstantOp(indexType, APInt(numInputs, 0), insertLoc, rewriter);

  // Declare wire for arbitration winner.
  auto win = rewriter.create<WireOp>(insertLoc, indexType, "win");

  // Declare wires for if each output is done.
  auto resultDone = rewriter.create<WireOp>(insertLoc, bitType, "resultDone");

  // Create predicates to assert if the win wire holds a valid index.
  auto hasWinnerCondition = rewriter.create<OrRPrimOp>(insertLoc, bitType, win);

  // Create an arbiter based on a simple priority-encoding scheme to assign an
  // index to the win wire. In the case that no input is valid, set a sentinel
  // value to indicate no winner was chosen. The constant values are remembered
  // in a map so they can be re-used later to assign the arg ready outputs.
  DenseMap<size_t, Value> argIndexValues;
  auto priorityArb = createPriorityArbiter(argValid, noWinner, argIndexValues,
                                           insertLoc, rewriter);

  rewriter.create<ConnectOp>(insertLoc, win, priorityArb);

  // Create the logic to assign the result outputs. The result valid and data
  // outputs will always be assigned. The win wire from the arbiter is used to
  // index into a tree of muxes to select the chosen input's signal(s). The
  // result outputs are gated on the win wire being non-zero.
  rewriter.create<ConnectOp>(insertLoc, resultValid, hasWinnerCondition);

  if (!isControlOp(op)) {
    auto resultDataMux = createOneHotMuxTree(argData, win, insertLoc, rewriter);
    rewriter.create<ConnectOp>(insertLoc, resultData, resultDataMux);
  }

  // Create the logic to set the done wires for the result. The done wire is
  // asserted when the output is valid and ready, or the emitted register is
  // set.
  auto resultValidAndReady = rewriter.create<AndPrimOp>(
      insertLoc, bitType, hasWinnerCondition, resultReady);

  rewriter.create<ConnectOp>(insertLoc, resultDone, resultValidAndReady);

  // Create the logic to assign the arg ready outputs. The logic is identical
  // for each arg. If the fired wire is asserted, and the win wire holds an
  // arg's index, that arg is ready.
  createMergeArgReady(argReady, resultDone, win, noWinner, argIndexValues,
                      insertLoc, rewriter);

  return true;
}

/// Please refer to test_cmerge.mlir test case.
/// Lowers the ControlMergeOp into primitive FIRRTL ops.
/// See http://www.cs.columbia.edu/~sedwards/papers/edwards2019compositional.pdf
/// Section 3.4.
bool HandshakeBuilder::visitHandshake(ControlMergeOp op) {
  auto *context = rewriter.getContext();

  bool isControl = isControlOp(op);
  unsigned numPorts = portList.size();
  unsigned numInputs = numPorts - 4;

  // The clock and reset signals will be used for registers.
  Value clock = portList[numPorts - 2][0];
  Value reset = portList[numPorts - 1][0];

  // The second to last output has the result's ready and valid signals, and
  // possibly data signal if isControl is not set.
  ValueVector resultSubfields = portList[numInputs];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];

  // The last output indicates which input is active now.
  ValueVector controlSubfields = portList[numInputs + 1];
  Value controlValid = controlSubfields[0];
  Value controlReady = controlSubfields[1];
  Value controlData = controlSubfields[2];

  // Walk through each arg data to collect the subfields.
  SmallVector<Value, 4> argValid;
  SmallVector<Value, 4> argReady;
  SmallVector<Value, 4> argData;
  for (unsigned i = 0; i < numInputs; ++i) {
    ValueVector argSubfields = portList[i];
    argValid.push_back(argSubfields[0]);
    argReady.push_back(argSubfields[1]);
    if (!isControl)
      argData.push_back(argSubfields[2]);
  }

  // Define some common types and values that will be used.
  auto bitType = UIntType::get(context, 1);
  auto indexType = UIntType::get(context, numInputs);
  auto noWinner =
      createConstantOp(indexType, APInt(numInputs, 0), insertLoc, rewriter);
  auto falseConst = createConstantOp(bitType, APInt(1, 0), insertLoc, rewriter);

  // Declare register for storing arbitration winner.
  auto won = rewriter.create<RegResetOp>(insertLoc, indexType, clock, reset,
                                         noWinner, "won");

  // Declare wire for arbitration winner.
  auto win = rewriter.create<WireOp>(insertLoc, indexType, "win");

  // Declare wire for whether the circuit just fired and emitted both outputs.
  auto fired = rewriter.create<WireOp>(insertLoc, bitType, "fired");

  // Declare registers for storing if each output has been emitted.
  auto resultEmitted = rewriter.create<RegResetOp>(
      insertLoc, bitType, clock, reset, falseConst, "resultEmitted");

  auto controlEmitted = rewriter.create<RegResetOp>(
      insertLoc, bitType, clock, reset, falseConst, "controlEmitted");

  // Declare wires for if each output is done.
  auto resultDone = rewriter.create<WireOp>(insertLoc, bitType, "resultDone");

  auto controlDone = rewriter.create<WireOp>(insertLoc, bitType, "controlDone");

  // Create predicates to assert if the win wire or won register hold a valid
  // index.
  auto hasWinnerCondition = rewriter.create<OrRPrimOp>(insertLoc, bitType, win);

  auto hadWinnerCondition = rewriter.create<OrRPrimOp>(insertLoc, bitType, won);

  // Create an arbiter based on a simple priority-encoding scheme to assign an
  // index to the win wire. If the won register is set, just use that. In
  // the case that won is not set and no input is valid, set a sentinel value to
  // indicate no winner was chosen. The constant values are remembered in a map
  // so they can be re-used later to assign the arg ready outputs.
  DenseMap<size_t, Value> argIndexValues;
  auto priorityArb = createPriorityArbiter(argValid, noWinner, argIndexValues,
                                           insertLoc, rewriter);

  priorityArb = rewriter.create<MuxPrimOp>(
      insertLoc, indexType, hadWinnerCondition, won, priorityArb);

  rewriter.create<ConnectOp>(insertLoc, win, priorityArb);

  // Create the logic to assign the result and control outputs. The result valid
  // output will always be assigned, and if isControl is not set, the result
  // data output will also be assigned. The control valid and data outputs will
  // always be assigned. The win wire from the arbiter is used to index into a
  // tree of muxes to select the chosen input's signal(s), and is fed directly
  // to the control output. Both the result and control valid outputs are gated
  // on the win wire being set to something other than the sentinel value.
  auto resultNotEmitted =
      rewriter.create<NotPrimOp>(insertLoc, bitType, resultEmitted);

  auto resultValidWire = rewriter.create<AndPrimOp>(
      insertLoc, bitType, hasWinnerCondition, resultNotEmitted);
  rewriter.create<ConnectOp>(insertLoc, resultValid, resultValidWire);

  if (!isControl) {
    Value resultData = resultSubfields[2];
    auto resultDataMux = createOneHotMuxTree(argData, win, insertLoc, rewriter);
    rewriter.create<ConnectOp>(insertLoc, resultData, resultDataMux);
  }

  auto controlNotEmitted =
      rewriter.create<NotPrimOp>(insertLoc, bitType, controlEmitted);

  auto controlValidWire = rewriter.create<AndPrimOp>(
      insertLoc, bitType, hasWinnerCondition, controlNotEmitted);
  rewriter.create<ConnectOp>(insertLoc, controlValid, controlValidWire);

  // Use the one-hot win wire to select the index to output in the control data.
  size_t controlOutputBits = getNumIndexBits(numInputs);
  auto controlOutputType = UIntType::get(context, controlOutputBits);
  SmallVector<Value, 8> controlOutputs;
  for (size_t i = 0; i < numInputs; ++i)
    controlOutputs.push_back(createConstantOp(
        controlOutputType, APInt(controlOutputBits, i), insertLoc, rewriter));

  auto controlOutput =
      createOneHotMuxTree(controlOutputs, win, insertLoc, rewriter);
  rewriter.create<ConnectOp>(insertLoc, controlData, controlOutput);

  // Create the logic to set the won register. If the fired wire is asserted, we
  // have finished this round and can and reset the register to the sentinel
  // value that indicates there is no winner. Otherwise, we need to hold the
  // value of the win register until we can fire.
  auto wonMux =
      rewriter.create<MuxPrimOp>(insertLoc, indexType, fired, noWinner, win);
  rewriter.create<ConnectOp>(insertLoc, won, wonMux);

  // Create the logic to set the done wires for the result and control. For both
  // outputs, the done wire is asserted when the output is valid and ready, or
  // the emitted register for that output is set.
  auto resultValidAndReady = rewriter.create<AndPrimOp>(
      insertLoc, bitType, resultValidWire, resultReady);

  auto resultDoneWire = rewriter.create<OrPrimOp>(
      insertLoc, bitType, resultEmitted, resultValidAndReady);
  rewriter.create<ConnectOp>(insertLoc, resultDone, resultDoneWire);

  auto controlValidAndReady = rewriter.create<AndPrimOp>(
      insertLoc, bitType, controlValidWire, controlReady);

  auto controlDoneWire = rewriter.create<OrPrimOp>(
      insertLoc, bitType, controlEmitted, controlValidAndReady);
  rewriter.create<ConnectOp>(insertLoc, controlDone, controlDoneWire);

  // Create the logic to set the fired wire. It is asserted when both result and
  // control are done.
  auto firedWire =
      rewriter.create<AndPrimOp>(insertLoc, bitType, resultDone, controlDone);
  rewriter.create<ConnectOp>(insertLoc, fired, firedWire);

  // Create the logic to assign the emitted registers. If the fired wire is
  // asserted, we have finished this round and can reset the registers to 0.
  // Otherwise, we need to hold the values of the done registers until we can
  // fire.
  auto resultEmittedWire = rewriter.create<MuxPrimOp>(insertLoc, bitType, fired,
                                                      falseConst, resultDone);
  rewriter.create<ConnectOp>(insertLoc, resultEmitted, resultEmittedWire);

  auto controlEmittedWire = rewriter.create<MuxPrimOp>(
      insertLoc, bitType, fired, falseConst, controlDone);
  rewriter.create<ConnectOp>(insertLoc, controlEmitted, controlEmittedWire);

  // Create the logic to assign the arg ready outputs. The logic is identical
  // for each arg. If the fired wire is asserted, and the win wire holds an
  // arg's index, that arg is ready.
  createMergeArgReady(argReady, fired, win, noWinner, argIndexValues, insertLoc,
                      rewriter);

  return true;
}

/// Please refer to test_branch.mlir test case.
bool HandshakeBuilder::visitHandshake(handshake::BranchOp op) {
  ValueVector argSubfields = portList[0];
  ValueVector resultSubfields = portList[1];
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];

  rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
  rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);

  if (!isControlOp(op)) {
    Value argData = argSubfields[2];
    Value resultData = resultSubfields[2];
    rewriter.create<ConnectOp>(insertLoc, resultData, argData);
  }
  return true;
}

/// Two outputs conditional branch operation.
/// Please refer to test_conditional_branch.mlir test case.
bool HandshakeBuilder::visitHandshake(ConditionalBranchOp op) {
  ValueVector conditionSubfields = portList[0];
  ValueVector argSubfields = portList[1];
  ValueVector trueResultSubfields = portList[2];
  ValueVector falseResultSubfields = portList[3];

  Value conditionValid = conditionSubfields[0];
  Value conditionReady = conditionSubfields[1];
  Value conditionData = conditionSubfields[2];
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];
  Value trueResultValid = trueResultSubfields[0];
  Value trueResultReady = trueResultSubfields[1];
  Value falseResultValid = falseResultSubfields[0];
  Value falseResultReady = falseResultSubfields[1];

  auto conditionArgValid = rewriter.create<AndPrimOp>(
      insertLoc, conditionValid.getType(), conditionValid, argValid);

  auto conditionNot = rewriter.create<NotPrimOp>(
      insertLoc, conditionData.getType(), conditionData);

  // Connect valid signal of both results.
  rewriter.create<ConnectOp>(
      insertLoc, trueResultValid,
      rewriter.create<AndPrimOp>(insertLoc, conditionData.getType(),
                                 conditionData, conditionArgValid));

  rewriter.create<ConnectOp>(
      insertLoc, falseResultValid,
      rewriter.create<AndPrimOp>(insertLoc, conditionNot.getType(),
                                 conditionNot, conditionArgValid));

  // Connect data signal of both results if applied.
  if (!isControlOp(op)) {
    Value argData = argSubfields[2];
    Value trueResultData = trueResultSubfields[2];
    Value falseResultData = falseResultSubfields[2];
    rewriter.create<ConnectOp>(insertLoc, trueResultData, argData);
    rewriter.create<ConnectOp>(insertLoc, falseResultData, argData);
  }

  // Connect ready signal of input and condition.
  auto selectedResultReady = rewriter.create<MuxPrimOp>(
      insertLoc, trueResultReady.getType(), conditionData, trueResultReady,
      falseResultReady);

  auto conditionArgReady =
      rewriter.create<AndPrimOp>(insertLoc, selectedResultReady.getType(),
                                 selectedResultReady, conditionArgValid);

  rewriter.create<ConnectOp>(insertLoc, argReady, conditionArgReady);
  rewriter.create<ConnectOp>(insertLoc, conditionReady, conditionArgReady);

  return true;
}

/// Please refer to test_lazy_fork.mlir test case.
bool HandshakeBuilder::visitHandshake(LazyForkOp op) {
  ValueVector argSubfields = portList.front();
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];

  // The input will be ready to accept new token when all outputs are ready.
  Value *tmpReady = &portList[1][1];
  for (unsigned i = 2, e = portList.size(); i < e; ++i) {
    Value resultReady = portList[i][1];
    *tmpReady = rewriter.create<AndPrimOp>(insertLoc, resultReady.getType(),
                                           resultReady, *tmpReady);
  }
  rewriter.create<ConnectOp>(insertLoc, argReady, *tmpReady);

  // All outputs must be ready for the LazyFork to send the token.
  auto resultValidOp = rewriter.create<AndPrimOp>(insertLoc, argValid.getType(),
                                                  argValid, *tmpReady);
  for (unsigned i = 1, e = portList.size(); i < e; ++i) {
    ValueVector resultfield = portList[i];
    Value resultValid = resultfield[0];
    rewriter.create<ConnectOp>(insertLoc, resultValid, resultValidOp);

    if (!isControlOp(op)) {
      Value argData = argSubfields[2];
      Value resultData = resultfield[2];
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    }
  }
  return true;
}

bool HandshakeBuilder::buildForkLogic(ValueVector *input,
                                      SmallVector<ValueVector *, 4> outputs,
                                      Value clock, Value reset,
                                      bool isControl) {
  if (input == nullptr)
    return false;

  auto argSubfields = *input;
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];

  unsigned resultNum = outputs.size();

  // Values that are useful.
  auto bitType = UIntType::get(rewriter.getContext(), 1);
  auto falseConst = createConstantOp(bitType, APInt(1, 0), insertLoc, rewriter);

  // Create done wire for all results.
  SmallVector<Value, 4> doneWires;
  for (unsigned i = 0; i < resultNum; ++i) {
    auto doneWire =
        rewriter.create<WireOp>(insertLoc, bitType, "done" + std::to_string(i));
    doneWires.push_back(doneWire);
  }

  // Create an AndPrimOp chain for generating the ready signal. Only if all
  // result ports are handshaked (done), the argument port is ready to accept
  // the next token.
  Value allDoneWire = rewriter.create<WireOp>(insertLoc, bitType, "allDone");
  buildReductionTree<AndPrimOp>(doneWires, allDoneWire);

  // Connect the allDoneWire to the input ready.
  rewriter.create<ConnectOp>(insertLoc, argReady, allDoneWire);

  // Create a notAllDoneWire for later use.
  auto notAllDoneWire =
      rewriter.create<WireOp>(insertLoc, bitType, "notAllDone");
  rewriter.create<ConnectOp>(
      insertLoc, notAllDoneWire,
      rewriter.create<NotPrimOp>(insertLoc, bitType, allDoneWire));

  // Create logic for each result port.
  unsigned idx = 0;
  for (auto doneWire : doneWires) {
    if (outputs[idx] == nullptr)
      return false;

    // Extract valid and ready from the current result port.
    auto resultSubfields = *outputs[idx];
    Value resultValid = resultSubfields[0];
    Value resultReady = resultSubfields[1];

    // If this is not a control component, extract data from the current result
    // port and connect it with the argument data.
    if (!isControl) {
      Value argData = argSubfields[2];
      Value resultData = resultSubfields[2];
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    }

    // Create a emitted register.
    auto emtdReg =
        rewriter.create<RegResetOp>(insertLoc, bitType, clock, reset,
                                    falseConst, "emtd" + std::to_string(idx));

    // Connect the emitted register with {doneWire && notallDoneWire}. Only if
    // notallDone, the emtdReg will be set to the value of doneWire. Otherwise,
    // all emtdRegs will be cleared to zero.
    auto emtd = rewriter.create<AndPrimOp>(insertLoc, bitType, doneWire,
                                           notAllDoneWire);
    rewriter.create<ConnectOp>(insertLoc, emtdReg, emtd);

    // Create a notEmtdWire for later use.
    auto notEmtdWire = rewriter.create<WireOp>(insertLoc, bitType,
                                               "notEmtd" + std::to_string(idx));
    rewriter.create<ConnectOp>(
        insertLoc, notEmtdWire,
        rewriter.create<NotPrimOp>(insertLoc, bitType, emtdReg));

    // Create valid signal and connect to the result valid. The reason of this
    // AndPrimOp is each result can only be emitted once.
    auto valid =
        rewriter.create<AndPrimOp>(insertLoc, bitType, notEmtdWire, argValid);
    rewriter.create<ConnectOp>(insertLoc, resultValid, valid);

    // Create validReady wire signal, which indicates a successful handshake in
    // the current clock cycle.
    auto validReadyWire = rewriter.create<WireOp>(
        insertLoc, bitType, "validReady" + std::to_string(idx));
    rewriter.create<ConnectOp>(
        insertLoc, validReadyWire,
        rewriter.create<AndPrimOp>(insertLoc, bitType, resultReady, valid));

    // Finally, we can drive the doneWire we created in the beginning with
    // {validReadyWire || emtdReg}, where emtdReg indicates a successful
    // handshake in a previous clock cycle.
    rewriter.create<ConnectOp>(
        insertLoc, doneWire,
        rewriter.create<OrPrimOp>(insertLoc, bitType, validReadyWire, emtdReg));

    // All done, move to the next result port.
    ++idx;
  }
  return true;
}

/// See http://www.cs.columbia.edu/~sedwards/papers/edwards2019compositional.pdf
/// Fig.10 for reference.
/// Please refer to test_fork.mlir test case.
bool HandshakeBuilder::visitHandshake(ForkOp op) {
  auto input = &portList.front();
  unsigned portNum = portList.size();

  // Collect all outputs ports.
  SmallVector<ValueVector *, 4> outputs;
  for (unsigned i = 1; i < portNum - 2; ++i)
    outputs.push_back(&portList[i]);

  // The clock and reset signals will be used for registers.
  auto clock = portList[portNum - 2][0];
  auto reset = portList[portNum - 1][0];

  return buildForkLogic(input, outputs, clock, reset, isControlOp(op));
}

/// Please refer to test_constant.mlir test case.
bool HandshakeBuilder::visitHandshake(handshake::ConstantOp op) {
  // The first input is control signal which will trigger the Constant
  // operation to emit tokens.
  ValueVector controlSubfields = portList.front();
  Value controlValid = controlSubfields[0];
  Value controlReady = controlSubfields[1];

  ValueVector resultSubfields = portList.back();
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  auto constantType = resultData.getType().cast<FIRRTLBaseType>();
  auto constantValue = op->getAttrOfType<IntegerAttr>("value").getValue();

  rewriter.create<ConnectOp>(insertLoc, resultValid, controlValid);
  rewriter.create<ConnectOp>(insertLoc, controlReady, resultReady);
  rewriter.create<ConnectOp>(
      insertLoc, resultData,
      createConstantOp(constantType, constantValue, insertLoc, rewriter));
  return true;
}

void HandshakeBuilder::buildControlBufferLogic(Value predValid, Value predReady,
                                               Value succValid, Value succReady,
                                               Value clock, Value reset,
                                               Value predData, Value succData) {
  auto bitType = UIntType::get(rewriter.getContext(), 1);
  auto falseConst = createConstantOp(bitType, APInt(1, 0), insertLoc, rewriter);

  // Create a wire and connect it to the register for the ready buffer.
  auto readyRegWire =
      rewriter.create<WireOp>(insertLoc, bitType, "readyRegWire");

  auto readyReg = rewriter.create<RegResetOp>(insertLoc, bitType, clock, reset,
                                              falseConst, "readyReg");
  rewriter.create<ConnectOp>(insertLoc, readyReg, readyRegWire);

  // Create the logic to drive the successor valid and potentially data.
  auto validResult = rewriter.create<MuxPrimOp>(insertLoc, bitType, readyReg,
                                                readyReg, predValid);
  rewriter.create<ConnectOp>(insertLoc, succValid, validResult);

  // Create the logic to drive the predecessor ready.
  auto notReady = rewriter.create<NotPrimOp>(insertLoc, bitType, readyReg);
  rewriter.create<ConnectOp>(insertLoc, predReady, notReady);

  // Create the logic for successor and register are both low.
  auto succNotReady = rewriter.create<NotPrimOp>(insertLoc, bitType, succReady);
  auto neitherReady =
      rewriter.create<AndPrimOp>(insertLoc, bitType, succNotReady, notReady);

  // Create a mux for taking the input when neither ready.
  auto ctrlNotReadyMux = rewriter.create<MuxPrimOp>(
      insertLoc, bitType, neitherReady, predValid, readyReg);

  // Create the logic for successor and register are both high.
  auto bothReady =
      rewriter.create<AndPrimOp>(insertLoc, bitType, succReady, readyReg);

  // Create a mux for emptying the register when both are ready.
  auto resetSignal = rewriter.create<MuxPrimOp>(insertLoc, bitType, bothReady,
                                                falseConst, ctrlNotReadyMux);
  rewriter.create<ConnectOp>(insertLoc, readyRegWire, resetSignal);

  // Add same logic for the data path if necessary.
  if (predData) {
    auto dataType = predData.getType().cast<FIRRTLBaseType>();
    auto ctrlDataRegWire =
        rewriter.create<WireOp>(insertLoc, dataType, "ctrlDataRegWire");

    auto ctrlZeroConst = createZeroDataConst(dataType, insertLoc, rewriter);

    auto ctrlDataReg = rewriter.create<RegResetOp>(
        insertLoc, dataType, clock, reset, ctrlZeroConst, "ctrlDataReg");

    rewriter.create<ConnectOp>(insertLoc, ctrlDataReg, ctrlDataRegWire);

    auto dataResult = rewriter.create<MuxPrimOp>(insertLoc, dataType, readyReg,
                                                 ctrlDataReg, predData);
    rewriter.create<ConnectOp>(insertLoc, succData, dataResult);

    auto dataNotReadyMux = rewriter.create<MuxPrimOp>(
        insertLoc, dataType, neitherReady, predData, ctrlDataReg);

    auto dataResetSignal = rewriter.create<MuxPrimOp>(
        insertLoc, dataType, bothReady, ctrlZeroConst, dataNotReadyMux);
    rewriter.create<ConnectOp>(insertLoc, ctrlDataRegWire, dataResetSignal);
  }
}

void HandshakeBuilder::buildDataBufferLogic(Value predValid, Value validReg,
                                            Value predReady, Value succReady,
                                            Value predData = nullptr,
                                            Value dataReg = nullptr) {
  auto bitType = UIntType::get(rewriter.getContext(), 1);

  // Create a signal for when the valid register is empty or the successor is
  // ready to accept new token.
  auto notValidReg = rewriter.create<NotPrimOp>(insertLoc, bitType, validReg);
  auto emptyOrReady =
      rewriter.create<OrPrimOp>(insertLoc, bitType, notValidReg, succReady);

  rewriter.create<ConnectOp>(insertLoc, predReady, emptyOrReady);

  // Create a mux that drives the register input. If the emptyOrReady signal
  // is asserted, the mux selects the predValid signal. Otherwise, it selects
  // the register output, keeping the output registered unchanged.
  auto validRegMux = rewriter.create<MuxPrimOp>(
      insertLoc, bitType, emptyOrReady, predValid, validReg);

  // Now we can drive the valid register.
  rewriter.create<ConnectOp>(insertLoc, validReg, validRegMux);

  // If data is not nullptr, create data logic.
  if (predData && dataReg) {
    auto dataType = predData.getType().cast<FIRRTLBaseType>();

    // Create a mux that drives the date register.
    auto dataRegMux = rewriter.create<MuxPrimOp>(
        insertLoc, dataType, emptyOrReady, predData, dataReg);
    rewriter.create<ConnectOp>(insertLoc, dataReg, dataRegMux);
  }
}

/// Connects src to all elements of dest recursively. If dest isn't a bundle
/// type, a normal connection is created.
static void connectSrcToAllElements(ImplicitLocOpBuilder &builder, Value dest,
                                    Value src) {
  if (auto bundleType = dest.getType().dyn_cast<BundleType>()) {
    for (int i = 0, e = bundleType.getNumElements(); i < e; ++i) {
      auto field = builder.create<SubfieldOp>(dest, i);
      connectSrcToAllElements(builder, field, src);
    }
  } else {
    builder.create<ConnectOp>(dest, src);
  }
}

FModuleOp buildInnerFIFO(CircuitOp circuit, StringRef moduleName,
                         unsigned depth, bool isControl,
                         FIRRTLBaseType dataType = FIRRTLBaseType()) {
  ImplicitLocOpBuilder builder(circuit.getLoc(), circuit.getContext());
  SmallVector<PortInfo> ports;
  auto bitType = UIntType::get(builder.getContext(), 1);
  auto loc = circuit.getLoc();
  auto strAttr = [&](StringRef str) { return builder.getStringAttr(str); };

  if (!isControl)
    ports.push_back(PortInfo{strAttr("dataIn"), dataType, Direction::In,
                             StringAttr{}, loc});

  ports.push_back(
      PortInfo{strAttr("readyIn"), bitType, Direction::In, StringAttr{}, loc});
  ports.push_back(
      PortInfo{strAttr("validIn"), bitType, Direction::In, StringAttr{}, loc});

  if (!isControl)
    ports.push_back(PortInfo{strAttr("dataOut"), dataType, Direction::Out,
                             StringAttr{}, loc});
  ports.push_back(PortInfo{strAttr("readyOut"), bitType, Direction::Out,
                           StringAttr{}, loc});

  ports.push_back(PortInfo{strAttr("validOut"), bitType, Direction::Out,
                           StringAttr{}, loc});

  // Add clock and reset signals.
  ports.push_back({strAttr("clock"), builder.getType<ClockType>(),
                   Direction::In, StringAttr{}, loc});
  ports.push_back({strAttr("reset"), builder.getType<UIntType>(1),
                   Direction::In, StringAttr{}, loc});

  builder.setInsertionPointToStart(circuit.getBodyBlock());
  auto moduleOp = builder.create<FModuleOp>(strAttr(moduleName), ports);
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());

  // Unpack module arguments.
  int portIdx = 0;
  Value dataIn = nullptr;
  Value dataOut = nullptr;
  if (!isControl)
    dataIn = moduleOp.getArgument(portIdx++);
  auto readyIn = moduleOp.getArgument(portIdx++);
  auto validIn = moduleOp.getArgument(portIdx++);

  if (!isControl)
    dataOut = moduleOp.getArgument(portIdx++);
  auto readyOut = moduleOp.getArgument(portIdx++);
  auto validOut = moduleOp.getArgument(portIdx++);

  auto clk = moduleOp.getArgument(portIdx++);
  auto rst = moduleOp.getArgument(portIdx++);

  auto depthPtrType =
      UIntType::get(builder.getContext(), llvm::Log2_64_Ceil(depth));
  // Depth bit width. Defined as log2ceil(depth+1) to be able to represent the
  // actual depth count. I.e. if depth=2 , we need at least 2 bits, to count up
  // to 2, from 0.
  auto depthType =
      UIntType::get(builder.getContext(), llvm::Log2_64_Ceil(depth + 1));

  /// Returns a constant value 'value' width a width equal to that of
  /// 'refValue'.
  auto getConstantOfEqWidth = [&](uint64_t value, Value refValue) {
    FIRRTLBaseType type = refValue.getType().cast<FIRRTLBaseType>();
    return createConstantOp(type, APInt(type.getBitWidthOrSentinel(), value),
                            loc, builder);
  };

  // Signal declarations
  auto zeroConst =
      createConstantOp(bitType, APInt(1, 0), builder.getLoc(), builder);
  auto oneConst =
      createConstantOp(bitType, APInt(1, 1), builder.getLoc(), builder);

  auto readEn = builder.create<WireOp>(bitType, "read_en");
  auto writeEn = builder.create<WireOp>(bitType, "write_en");
  auto tail =
      builder.create<RegResetOp>(depthPtrType, clk, rst, zeroConst, "tail_reg");
  auto head =
      builder.create<RegResetOp>(depthPtrType, clk, rst, zeroConst, "head_reg");
  auto full = builder.create<WireOp>(bitType, "full");
  auto empty = builder.create<WireOp>(bitType, "empty");
  auto notEmpty = builder.create<NotPrimOp>(empty);
  auto count =
      builder.create<RegResetOp>(depthType, clk, rst, zeroConst, "count_reg");

  // Function for truncating results to a given types' width.
  auto trunc = [&](Value v, FIRRTLBaseType toType) {
    unsigned truncBits =
        v.getType().cast<FIRRTLBaseType>().getBitWidthOrSentinel() -
        toType.getBitWidthOrSentinel();
    return builder.create<TailPrimOp>(v, truncBits);
  };

  // Full when number of elements in fifo is == depth
  builder.create<ConnectOp>(
      full,
      builder.create<EQPrimOp>(count, getConstantOfEqWidth(depth, count)));

  // Empty when number of elements in fifo is == 0
  builder.create<ConnectOp>(
      empty, builder.create<EQPrimOp>(count, getConstantOfEqWidth(0, count)));

  // Ready if there is space in the FIFO.
  builder.create<ConnectOp>(
      readyOut,
      builder.create<OrPrimOp>(builder.create<NotPrimOp>(full), readyIn));

  // Ready if next can accept and there is something in the FIFO to read.
  builder.create<ConnectOp>(readEn,
                            builder.create<AndPrimOp>(notEmpty, readyIn));

  // Valid when not empty
  builder.create<ConnectOp>(validOut, notEmpty);

  // Writing when input is valid and is not full or input ready.
  builder.create<ConnectOp>(
      writeEn, builder.create<AndPrimOp>(
                   validIn, builder.create<OrPrimOp>(
                                builder.create<NotPrimOp>(full), readyIn)));

  // Memory declaration, data writing and reading.
  if (!isControl) {
    SmallVector<std::pair<StringAttr, MemOp::PortKind>, 8> memPorts;
    memPorts.push_back({strAttr("read"), MemOp::PortKind::Read});
    memPorts.push_back({strAttr("write"), MemOp::PortKind::Write});
    llvm::SmallVector<Type> memTypes;
    llvm::SmallVector<Attribute> memNames;
    for (auto p : memPorts) {
      memTypes.push_back(MemOp::getTypeForPort(depth, dataType, p.second));
      memNames.push_back(strAttr(p.first.str()));
    }

    // Build a combinational read, synchronous write memory. We set the read
    // under write attribute to new for transparency.
    auto memOp =
        builder.create<MemOp>(memTypes, /*readLatency=*/0, /*writeLatency=*/1,
                              depth, RUWAttr::New, memNames, "mem");

    // Extract the port bundles.
    auto readBundle = memOp.getPortNamed("read");
    auto readType =
        readBundle.getType().cast<FIRRTLBaseType>().cast<BundleType>();
    auto writeBundle = memOp.getPortNamed("write");
    auto writeType =
        writeBundle.getType().cast<FIRRTLBaseType>().cast<BundleType>();

    // Get the clock out of the bundle and connect them.
    auto readClock = builder.create<SubfieldOp>(
        readBundle, *readType.getElementIndex("clk"));
    builder.create<ConnectOp>(readClock, clk);
    auto writeClock = builder.create<SubfieldOp>(
        writeBundle, *writeType.getElementIndex("clk"));
    builder.create<ConnectOp>(writeClock, clk);

    // Get the addresses out of the bundle
    auto readAddr = builder.create<SubfieldOp>(
        readBundle, *readType.getElementIndex("addr"));
    auto writeAddr = builder.create<SubfieldOp>(
        writeBundle, *readType.getElementIndex("addr"));

    // Connect read and write to head and tail registers.
    builder.create<ConnectOp>(readAddr, head);
    builder.create<ConnectOp>(writeAddr, tail);

    // Get the memory enable out of the bundles.
    auto memReadEn =
        builder.create<SubfieldOp>(readBundle, *readType.getElementIndex("en"));
    auto memWriteEn = builder.create<SubfieldOp>(
        writeBundle, *writeType.getElementIndex("en"));
    // Always read
    builder.create<ConnectOp>(memReadEn, oneConst);
    // Write on writeEn
    builder.create<ConnectOp>(memWriteEn, writeEn);

    // Connect read and write data.
    auto readData = builder.create<SubfieldOp>(
        readBundle, *readType.getElementIndex("data"));
    auto writeData = builder.create<SubfieldOp>(
        writeBundle, *writeType.getElementIndex("data"));
    builder.create<ConnectOp>(dataOut, readData);
    builder.create<ConnectOp>(writeData, dataIn);

    // Get the store mask out of the bundle.
    auto writeMask = builder.create<SubfieldOp>(
        writeBundle, *writeType.getElementIndex("mask"));

    // We might be storing bundles. Therefore, we have to ensure that writeEn is
    // connected to all elements of the mask.
    connectSrcToAllElements(builder, writeMask, writeEn);
  }

  // Next-state tail register; tail <- writeEn ? tail + 1 % depth : tail
  // (tail + 1 % depth) may be wider than tail, so also add truncation.
  auto tail1 = builder.create<AddPrimOp>(tail, oneConst);
  builder.create<ConnectOp>(
      tail, builder.create<MuxPrimOp>(
                writeEn,
                trunc(builder.create<RemPrimOp>(
                          tail1, getConstantOfEqWidth(depth, tail1)),
                      depthPtrType),
                tail));

  // Next-state head register; head <- readEn ? head + 1 % depth : head
  // (head + 1 % depth) may be wider than tail, so also add truncation.
  auto head1 = builder.create<AddPrimOp>(head, oneConst);
  builder.create<ConnectOp>(
      head, builder.create<MuxPrimOp>(
                readEn,
                trunc(builder.create<RemPrimOp>(
                          head1, getConstantOfEqWidth(depth, head1)),
                      depthPtrType),
                head));

  // Next-state count. Update whenever filling xor emptying. In other cases,
  // nothing happens to the total number of elements in the fifo.
  auto countp1 =
      builder.create<AddPrimOp>(count, getConstantOfEqWidth(1, count));
  auto countn1 =
      builder.create<SubPrimOp>(count, getConstantOfEqWidth(1, count));
  auto readXorWrite = builder.create<XorPrimOp>(readEn, writeEn);
  auto nsCountMux = builder.create<MuxPrimOp>(
      readXorWrite,
      trunc(builder.create<MuxPrimOp>(writeEn, countp1, countn1), depthType),
      count);
  builder.create<ConnectOp>(count, nsCountMux);
  return moduleOp;
}

bool HandshakeBuilder::buildFIFOBufferLogic(int64_t numStage,
                                            ValueVector *input,
                                            ValueVector *output, Value clock,
                                            Value reset, bool isControl) {
  ImplicitLocOpBuilder builder(insertLoc, rewriter.getContext());
  builder.setInsertionPoint(rewriter.getInsertionBlock(),
                            rewriter.getInsertionPoint());
  auto inputSubfields = *input;
  auto inputValid = inputSubfields[0];
  auto inputReady = inputSubfields[1];
  Value inputData = nullptr;
  FIRRTLBaseType dataType = nullptr;
  if (!isControl) {
    inputData = inputSubfields[2];
    dataType = inputData.getType().cast<FIRRTLBaseType>();
  }

  auto outputSubfields = *output;
  auto outputValid = outputSubfields[0];
  auto outputReady = outputSubfields[1];

  auto bitType = UIntType::get(builder.getContext(), 1);
  auto muxSelWire = builder.create<WireOp>(insertLoc, bitType, "muxSelWire");
  auto fifoValid = builder.create<WireOp>(insertLoc, bitType, "fifoValid");
  auto fifoPValid = builder.create<WireOp>(insertLoc, bitType, "fifoPValid");
  auto fifoReady = builder.create<WireOp>(insertLoc, bitType, "fifoReady");
  auto fifoNReady = builder.create<WireOp>(insertLoc, bitType, "fifoNReady");
  Value fifoIn = nullptr;
  Value fifoOut = nullptr;
  if (!isControl) {
    fifoIn = builder.create<WireOp>(insertLoc, dataType, "fifoIn");
    fifoOut = builder.create<WireOp>(insertLoc, dataType, "fifoOut");
  }

  // Connect output valid and ready signals.
  builder.create<ConnectOp>(outputValid,
                            builder.create<OrPrimOp>(inputValid, fifoValid));
  builder.create<ConnectOp>(inputReady,
                            builder.create<OrPrimOp>(fifoReady, outputReady));

  builder.create<ConnectOp>(
      fifoPValid,
      builder.create<AndPrimOp>(
          inputValid, builder.create<OrPrimOp>(
                          builder.create<NotPrimOp>(outputReady), fifoValid)));

  builder.create<ConnectOp>(muxSelWire, fifoValid);
  builder.create<ConnectOp>(fifoNReady, outputReady);
  if (!isControl)
    builder.create<ConnectOp>(fifoIn, inputData);

  std::string innerFifoModName = "innerFIFO_" + std::to_string(numStage);
  if (!isControl)
    innerFifoModName += getTypeName(insertLoc, inputData.getType());
  else
    innerFifoModName += "_ctrl";

  // Instantiate the inner FIFO. Check if we already have one of the
  // appropriate type, else, generate it.
  FModuleLike innerFifoModule = checkSubModuleOp(circuit, innerFifoModName);
  if (!innerFifoModule)
    innerFifoModule = buildInnerFIFO(circuit, innerFifoModName, numStage,
                                     isControl, dataType);

  auto innerFIFOInst =
      builder.create<firrtl::InstanceOp>(innerFifoModule, "innerFIFO");

  // Unpack inner fifo ports
  int portIdx = 0;
  Value fifoDataIn = nullptr;
  Value fifoDataOut = nullptr;
  if (!isControl)
    fifoDataIn = innerFIFOInst.getResult(portIdx++);
  auto fifoReadyIn = innerFIFOInst.getResult(portIdx++);
  auto fifoValidIn = innerFIFOInst.getResult(portIdx++);

  if (!isControl)
    fifoDataOut = innerFIFOInst.getResult(portIdx++);
  auto fifoReadyOut = innerFIFOInst.getResult(portIdx++);
  auto fifoValidOut = innerFIFOInst.getResult(portIdx++);

  auto fifoClk = innerFIFOInst.getResult(portIdx++);
  auto fifoRst = innerFIFOInst.getResult(portIdx++);

  builder.create<ConnectOp>(fifoClk, clock);
  builder.create<ConnectOp>(fifoRst, reset);
  builder.create<ConnectOp>(fifoValidIn, fifoPValid);
  builder.create<ConnectOp>(fifoReadyIn, fifoNReady);
  builder.create<ConnectOp>(fifoValid, fifoValidOut);
  builder.create<ConnectOp>(fifoReady, fifoReadyOut);
  if (!isControl) {
    builder.create<ConnectOp>(fifoDataIn, fifoIn);
    builder.create<ConnectOp>(fifoOut, fifoDataOut);
  }

  // Select fifo or bypass input based on mux selection.
  if (!isControl) {
    auto outputData = outputSubfields[2];
    auto muxOut = builder.create<MuxPrimOp>(muxSelWire, fifoOut, inputData);
    builder.create<ConnectOp>(outputData, muxOut);
  }

  return true;
}

bool HandshakeBuilder::buildSeqBufferLogic(int64_t numStage, ValueVector *input,
                                           ValueVector *output, Value clock,
                                           Value reset, bool isControl,
                                           ArrayRef<int64_t> initValues) {
  if (input == nullptr || output == nullptr)
    return false;

  auto inputSubfields = *input;
  auto inputValid = inputSubfields[0];
  auto inputReady = inputSubfields[1];

  auto outputSubfields = *output;
  auto outputValid = outputSubfields[0];
  auto outputReady = outputSubfields[1];

  // Create useful value and type for valid/ready signal.
  auto bitType = UIntType::get(rewriter.getContext(), 1);
  auto falseConst = createConstantOp(bitType, APInt(1, 0), insertLoc, rewriter);
  auto trueConst = createConstantOp(bitType, APInt(1, 1), insertLoc, rewriter);

  // Create useful value and type for data signal.
  FIRRTLBaseType dataType = nullptr;
  Value zeroDataConst = nullptr;

  // Temporary values for storing the valid, ready, and data signals in the
  // procedure of constructing the multi-stages buffer.
  Value currentValid = inputValid;
  Value currentReady = inputReady;
  Value currentData = nullptr;

  // If is not a control buffer, fill in corresponding values and type.
  if (!isControl) {
    auto inputData = inputSubfields[2];

    dataType = inputData.getType().cast<FIRRTLBaseType>();

    zeroDataConst = createZeroDataConst(dataType, insertLoc, rewriter);
    currentData = inputData;
  }

  // Create multiple stages buffer logic.
  for (unsigned i = 0; i < numStage; ++i) {
    bool isInitialized = initValues.size() > i;

    // Create wires for ready signal from the success buffer stage.
    auto readyWire = rewriter.create<WireOp>(insertLoc, bitType,
                                             "readyWire" + std::to_string(i));

    // Create a register for valid signal.
    auto validReg = rewriter.create<RegResetOp>(
        insertLoc, bitType, clock, reset,
        isInitialized ? trueConst : falseConst, "validReg" + std::to_string(i));

    // Create registers for data signal.
    Value dataReg = nullptr;
    Value initValue = zeroDataConst;
    if (!isControl) {
      if (isInitialized) {
        assert(dataType.isa<IntType>() &&
               "initial values are only supported for integer buffers");
        initValue = createConstantOp(
            dataType, APInt(dataType.getBitWidthOrSentinel(), initValues[i]),
            insertLoc, rewriter);
      }
      dataReg =
          rewriter.create<RegResetOp>(insertLoc, dataType, clock, reset,
                                      initValue, "dataReg" + std::to_string(i));
    }

    // Create wires for valid, ready and data signal coming from the control
    // buffer stage.
    auto ctrlValidWire = rewriter.create<WireOp>(
        insertLoc, bitType, "ctrlValidWire" + std::to_string(i));

    auto ctrlReadyWire = rewriter.create<WireOp>(
        insertLoc, bitType, "ctrlReadyWire" + std::to_string(i));

    Value ctrlDataWire;
    if (!isControl)
      ctrlDataWire = rewriter.create<WireOp>(
          insertLoc, dataType, "ctrlDataWire" + std::to_string(i));

    // Build the current stage of the buffer.
    buildDataBufferLogic(currentValid, validReg, currentReady, readyWire,
                         currentData, dataReg);

    buildControlBufferLogic(validReg, readyWire, ctrlValidWire, ctrlReadyWire,
                            clock, reset, dataReg, ctrlDataWire);

    // Update the current valid, ready, and data.
    currentValid = ctrlValidWire;
    currentReady = ctrlReadyWire;
    currentData = ctrlDataWire;
  }

  // Connect to the output ports.
  rewriter.create<ConnectOp>(insertLoc, outputValid, currentValid);
  rewriter.create<ConnectOp>(insertLoc, currentReady, outputReady);
  if (!isControl) {
    auto outputData = outputSubfields[2];
    rewriter.create<ConnectOp>(insertLoc, outputData, currentData);
  }

  return true;
}

bool HandshakeBuilder::visitHandshake(BufferOp op) {
  ValueVector input = portList[0];
  ValueVector output = portList[1];

  Value clock = portList[2][0];
  Value reset = portList[3][0];

  // For now, we only support sequential buffers.
  if (op.isSequential()) {
    SmallVector<int64_t> initValues = {};
    if (op.getInitValues())
      initValues = op.getInitValueArray();
    return buildSeqBufferLogic(op.getNumSlots(), &input, &output, clock, reset,
                               isControlOp(op), initValues);
  }

  return buildFIFOBufferLogic(op.getNumSlots(), &input, &output, clock, reset,
                              isControlOp(op));
}

bool HandshakeBuilder::visitHandshake(ExternalMemoryOp op) {

  // The external memory input is a bundle containing equivalent bundles to the
  // remainder of inputs to this component. Due to this, we simply need to
  // connect everything.

  // Port list format:
  // [0]: external memory bundle { like everything below, but inside a bundle }
  // [...]: [{store data, store address}, ...]
  // [...]: [load address, ...]
  // [...]: [load data, ...]
  // [...]: [control output, ...]
  auto inBundle = op.getOperand(0).getType().cast<BundleType>();
  unsigned numElements = inBundle.getNumElements();
  auto loc = op.getLoc();

  auto &inPort = portList[0];
  for (unsigned i = 0; i < numElements; ++i) {
    // the inPortBundle will be a handshake bundle for all inputs apart from
    // clock and reset - these are non-bundled.
    auto inPortBundle = inPort[i];
    const bool outerFlip = inBundle.getElement(i).isFlip;

    for (auto field : enumerate(portList[1 + i])) {
      Value extInputSubfield;
      bool innerFlip;

      // Extract the bundle field and flip state.
      if (inPortBundle.getType().isa<BundleType>()) {
        extInputSubfield =
            rewriter.create<SubfieldOp>(insertLoc, inPortBundle, field.index());
        innerFlip = inBundle.getElement(i)
                        .type.cast<BundleType>()
                        .getElement(field.index())
                        .isFlip;
      } else {
        extInputSubfield = inPortBundle;
        innerFlip = inBundle.getElement(i).isFlip;
      }

      if (outerFlip ^ innerFlip)
        rewriter.create<ConnectOp>(loc, extInputSubfield, field.value());
      else
        rewriter.create<ConnectOp>(loc, field.value(), extInputSubfield);
    }
  }

  return true;
}

bool HandshakeBuilder::visitHandshake(MemoryOp op) {
  // Get the memory type and element type.
  MemRefType type = op.getMemRefType();
  Type elementType = type.getElementType();
  if (!elementType.isSignlessInteger()) {
    op.emitError("only memrefs of signless ints are supported");
    return false;
  }

  // Set up FIRRTL memory attributes. This circuit relies on a read latency of 0
  // and a write latency of 1, but this could be generalized.
  uint32_t readLatency = 0;
  uint32_t writeLatency = 1;
  RUWAttr ruw = RUWAttr::Old;
  uint64_t depth = type.getNumElements();
  FIRRTLBaseType dataType = getFIRRTLType(elementType);
  auto name = "mem" + std::to_string(op.getId());

  // Helpers to get port identifiers.
  auto loadIdentifier = [&](size_t i) {
    return rewriter.getStringAttr("load" + std::to_string(i));
  };

  auto storeIdentifier = [&](size_t i) {
    return rewriter.getStringAttr("store" + std::to_string(i));
  };

  // Collect the port info for each port.
  uint64_t numLoads = op.getLdCount();
  uint64_t numStores = op.getStCount();
  SmallVector<std::pair<StringAttr, MemOp::PortKind>, 8> ports;
  for (size_t i = 0; i < numLoads; ++i) {
    auto portName = loadIdentifier(i);
    auto portKind = MemOp::PortKind::Read;
    ports.push_back({portName, portKind});
  }
  for (size_t i = 0; i < numStores; ++i) {
    auto portName = storeIdentifier(i);
    auto portKind = MemOp::PortKind::Write;
    ports.push_back({portName, portKind});
  }

  llvm::SmallVector<Type> resultTypes;
  llvm::SmallVector<Attribute> resultNames;
  for (auto p : ports) {
    resultTypes.push_back(MemOp::getTypeForPort(depth, dataType, p.second));
    resultNames.push_back(rewriter.getStringAttr(p.first.str()));
  }

  auto memOp =
      rewriter.create<MemOp>(insertLoc, resultTypes, readLatency, writeLatency,
                             depth, ruw, resultNames, name);

  // Prepare to create each load and store port logic.
  auto bitType = UIntType::get(rewriter.getContext(), 1);
  auto numPorts = portList.size();
  auto clock = portList[numPorts - 2][0];
  auto reset = portList[numPorts - 1][0];

  // Collect load arguments.
  for (size_t i = 0; i < numLoads; ++i) {
    // Extract load ports from the port list.
    auto loadAddr = portList[2 * numStores + i];
    auto loadData = portList[2 * numStores + numLoads + i];
    auto loadControl = portList[3 * numStores + 2 * numLoads + i];

    assert(loadAddr.size() == 3 && loadData.size() == 3 &&
           loadControl.size() == 2 && "incorrect load port number");

    // Unpack load address.
    auto loadAddrValid = loadAddr[0];
    auto loadAddrData = loadAddr[2];

    // Unpack load data.
    auto loadDataData = loadData[2];

    // Create a subfield op to access this port in the memory.
    auto fieldName = loadIdentifier(i);
    auto memBundle = memOp.getPortNamed(fieldName);
    auto memType = memBundle.getType().cast<BundleType>();

    // Get the clock out of the bundle and connect it.
    auto memClock = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("clk"));
    rewriter.create<ConnectOp>(insertLoc, memClock, clock);

    // Get the load address out of the bundle.
    auto memAddr = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("addr"));

    // Since addresses coming from Handshake are IndexType and have a hardcoded
    // 64-bit width in this pass, we may need to truncate down to the actual
    // size of the address port used by the FIRRTL memory.
    auto memAddrType = memAddr.getType().cast<FIRRTLBaseType>();
    auto loadAddrType = loadAddrData.getType().cast<FIRRTLBaseType>();
    if (memAddrType != loadAddrType) {
      auto memAddrPassiveType = memAddrType.getPassiveType();
      auto tailAmount = loadAddrType.getBitWidthOrSentinel() -
                        memAddrPassiveType.getBitWidthOrSentinel();
      loadAddrData = rewriter.create<TailPrimOp>(insertLoc, memAddrPassiveType,
                                                 loadAddrData, tailAmount);
    }

    // Connect the load address to the memory.
    rewriter.create<ConnectOp>(insertLoc, memAddr, loadAddrData);

    // Get the load data out of the bundle.
    auto memData = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("data"));

    // Connect the memory to the load data.
    rewriter.create<ConnectOp>(insertLoc, loadDataData, memData);

    // Get the load enable out of the bundle.
    auto memEnable = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("en"));

    // Connect the address valid signal to the memory enable.
    rewriter.create<ConnectOp>(insertLoc, memEnable, loadAddrValid);

    // Create control-only fork for the load address valid and ready signal.
    buildForkLogic(&loadAddr, {&loadData, &loadControl}, clock, reset, true);
  }

  // Collect store arguments.
  for (size_t i = 0; i < numStores; ++i) {
    // Extract store ports from the port list.
    auto storeData = portList[2 * i];
    auto storeAddr = portList[2 * i + 1];
    auto storeControl = portList[2 * numStores + 2 * numLoads + i];

    assert(storeAddr.size() == 3 && storeData.size() == 3 &&
           storeControl.size() == 2 && "incorrect store port number");

    // Unpack store data.
    auto storeDataReady = storeData[1];
    auto storeDataData = storeData[2];

    // Unpack store address.
    auto storeAddrReady = storeAddr[1];
    auto storeAddrData = storeAddr[2];

    // Unpack store control.
    auto storeControlValid = storeControl[0];

    auto fieldName = storeIdentifier(i);
    auto memBundle = memOp.getPortNamed(fieldName);
    auto memType = memBundle.getType().cast<BundleType>();

    // Get the clock out of the bundle and connect it.
    auto memClock = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("clk"));
    rewriter.create<ConnectOp>(insertLoc, memClock, clock);

    // Get the store address out of the bundle.
    auto memAddr = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("addr"));

    // Since addresses coming from Handshake are IndexType and have a hardcoded
    // 64-bit width in this pass, we may need to truncate down to the actual
    // size of the address port used by the FIRRTL memory.
    auto memAddrType = memAddr.getType();
    auto storeAddrType = storeAddrData.getType().cast<FIRRTLBaseType>();
    if (memAddrType != storeAddrType) {
      auto memAddrPassiveType = memAddrType.getPassiveType();
      auto tailAmount = storeAddrType.getBitWidthOrSentinel() -
                        memAddrPassiveType.getBitWidthOrSentinel();
      storeAddrData = rewriter.create<TailPrimOp>(insertLoc, memAddrPassiveType,
                                                  storeAddrData, tailAmount);
    }

    // Connect the store address to the memory.
    rewriter.create<ConnectOp>(insertLoc, memAddr, storeAddrData);

    // Get the store data out of the bundle.
    auto memData = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("data"));

    // Connect the store data to the memory.
    rewriter.create<ConnectOp>(insertLoc, memData, storeDataData);

    // Create a register to buffer the valid path by 1 cycle, to match the write
    // latency of 1.
    auto falseConst =
        createConstantOp(bitType, APInt(1, 0), insertLoc, rewriter);
    auto writeValidBuffer = rewriter.create<RegResetOp>(
        insertLoc, bitType, clock, reset, falseConst, "writeValidBuffer");

    // Connect the write valid buffer to the store control valid.
    rewriter.create<ConnectOp>(insertLoc, storeControlValid, writeValidBuffer);

    // Create the logic for when both the buffered write valid signal and the
    // store complete ready signal are asserted.
    Value storeCompleted =
        rewriter.create<WireOp>(insertLoc, bitType, "storeCompleted");
    ValueVector storeCompletedVector({Value(), storeCompleted});
    buildAllReadyLogic({&storeCompletedVector}, &storeControl,
                       writeValidBuffer);

    // Create a signal for when the write valid buffer is empty or the output is
    // ready.
    auto notWriteValidBuffer =
        rewriter.create<NotPrimOp>(insertLoc, bitType, writeValidBuffer);

    auto emptyOrComplete = rewriter.create<OrPrimOp>(
        insertLoc, bitType, notWriteValidBuffer, storeCompleted);

    // Connect the gate to both the store address ready and store data ready.
    rewriter.create<ConnectOp>(insertLoc, storeAddrReady, emptyOrComplete);
    rewriter.create<ConnectOp>(insertLoc, storeDataReady, emptyOrComplete);

    // Create a wire for when both the store address and data are valid.
    SmallVector<Value, 2> storeValids;
    extractValues({&storeAddr, &storeData}, 0, storeValids);
    Value writeValid =
        rewriter.create<WireOp>(insertLoc, bitType, "writeValid");
    buildReductionTree<AndPrimOp>(storeValids, writeValid);

    // Create a mux that drives the buffer input. If the emptyOrComplete signal
    // is asserted, the mux selects the writeValid signal. Otherwise, it selects
    // the buffer output, keeping the output registered until the
    // emptyOrComplete signal is asserted.
    auto writeValidBufferMux = rewriter.create<MuxPrimOp>(
        insertLoc, bitType, emptyOrComplete, writeValid, writeValidBuffer);

    rewriter.create<ConnectOp>(insertLoc, writeValidBuffer,
                               writeValidBufferMux);

    // Get the store enable out of the bundle.
    auto memEnable = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("en"));

    // Connect the write valid signal to the memory enable.
    rewriter.create<ConnectOp>(insertLoc, memEnable, writeValid);

    // Get the store mask out of the bundle.
    auto memMask = rewriter.create<SubfieldOp>(
        insertLoc, memBundle, *memType.getElementIndex("mask"));

    // Since we are not storing bundles in the memory, we can assume the mask is
    // a single bit.
    rewriter.create<ConnectOp>(insertLoc, memMask, writeValid);
  }

  return true;
}

bool HandshakeBuilder::visitHandshake(handshake::StoreOp op) {
  // Input address accepted from the predecessor.
  ValueVector inputAddr = portList[0];
  Value inputAddrData = inputAddr[2];

  // Input data accepted from the predecessor.
  ValueVector inputData = portList[1];
  Value inputDataData = inputData[2];

  // Control channel.
  ValueVector control = portList[2];

  // Data sending to the MemoryOp.
  ValueVector outputData = portList[3];
  Value outputDataValid = outputData[0];
  Value outputDataReady = outputData[1];
  Value outputDataData = outputData[2];

  // Address sending to the MemoryOp.
  ValueVector outputAddr = portList[4];
  Value outputAddrValid = outputAddr[0];
  Value outputAddrReady = outputAddr[1];
  Value outputAddrData = outputAddr[2];

  auto bitType = UIntType::get(rewriter.getContext(), 1);

  // Create a wire that will be asserted when all inputs are valid.
  auto inputsValid = rewriter.create<WireOp>(insertLoc, bitType, "inputsValid");

  // Create a gate that will be asserted when all outputs are ready.
  auto outputsReady = rewriter.create<AndPrimOp>(
      insertLoc, bitType, outputDataReady, outputAddrReady);

  // Build the standard join logic from the inputs to the inputsValid and
  // outputsReady signals.
  ValueVector joinLogicOutput({inputsValid, outputsReady});
  buildJoinLogic({&inputData, &inputAddr, &control}, &joinLogicOutput);

  // Output address and data signals are connected directly.
  rewriter.create<ConnectOp>(insertLoc, outputAddrData, inputAddrData);
  rewriter.create<ConnectOp>(insertLoc, outputDataData, inputDataData);

  // Output valid signals are connected from the inputsValid wire.
  rewriter.create<ConnectOp>(insertLoc, outputDataValid, inputsValid);
  rewriter.create<ConnectOp>(insertLoc, outputAddrValid, inputsValid);

  return true;
}

bool HandshakeBuilder::visitHandshake(handshake::LoadOp op) {
  // Input address accepted from the predecessor.
  ValueVector inputAddr = portList[0];
  Value inputAddrValid = inputAddr[0];
  Value inputAddrReady = inputAddr[1];
  Value inputAddrData = inputAddr[2];

  // Data accepted from the MemoryOp.
  ValueVector memoryData = portList[1];
  Value memoryDataValid = memoryData[0];
  Value memoryDataReady = memoryData[1];
  Value memoryDataData = memoryData[2];

  // Control channel.
  ValueVector control = portList[2];
  Value controlValid = control[0];
  Value controlReady = control[1];

  // Output data sending to the successor.
  ValueVector outputData = portList[3];
  Value outputDataValid = outputData[0];
  Value outputDataReady = outputData[1];
  Value outputDataData = outputData[2];

  // Address sending to the MemoryOp.
  ValueVector memoryAddr = portList[4];
  Value memoryAddrValid = memoryAddr[0];
  Value memoryAddrReady = memoryAddr[1];
  Value memoryAddrData = memoryAddr[2];

  auto bitType = UIntType::get(rewriter.getContext(), 1);

  // Address and data are connected accordingly.
  rewriter.create<ConnectOp>(insertLoc, memoryAddrData, inputAddrData);
  rewriter.create<ConnectOp>(insertLoc, outputDataData, memoryDataData);

  // The valid/ready logic between inputAddr, control, and memoryAddr is similar
  // to a JoinOp logic.
  auto addrValid = rewriter.create<AndPrimOp>(insertLoc, bitType,
                                              inputAddrValid, controlValid);
  rewriter.create<ConnectOp>(insertLoc, memoryAddrValid, addrValid);

  auto addrCompleted = rewriter.create<AndPrimOp>(insertLoc, bitType, addrValid,
                                                  memoryAddrReady);
  rewriter.create<ConnectOp>(insertLoc, inputAddrReady, addrCompleted);
  rewriter.create<ConnectOp>(insertLoc, controlReady, addrCompleted);

  // The valid/ready logic between memoryData and outputData is a direct
  // connection.
  rewriter.create<ConnectOp>(insertLoc, outputDataValid, memoryDataValid);
  rewriter.create<ConnectOp>(insertLoc, memoryDataReady, outputDataReady);

  return true;
}

/// Please refer to test_pack_unpack.mlir test case.
bool HandshakeBuilder::visitHandshake(PackOp op) {
  ValueVector tuple = portList.back();
  Value tupleData = tuple[2];

  auto bundleType = tupleData.getType().dyn_cast<BundleType>();

  // Create subfields for each bundle element
  ValueVector elements;
  for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i)
    elements.push_back(rewriter.create<SubfieldOp>(insertLoc, tupleData, i));

  // Collect all input ports.
  SmallVector<ValueVector *, 4> inputs;
  for (unsigned i = 0, e = portList.size() - 1; i < e; ++i)
    inputs.push_back(&portList[i]);

  // Connect each input to the corresponding part of the output bundle
  for (auto [element, input] : llvm::zip(elements, inputs))
    rewriter.create<ConnectOp>(insertLoc, element, (*input)[2]);

  return buildJoinLogic(inputs, &tuple);
}

/// Please refer to test_pack_unpack.mlir test case.
bool HandshakeBuilder::visitHandshake(UnpackOp op) {
  ValueVector tuple = portList[0];
  Value tupleData = tuple[2];
  unsigned portNum = portList.size();

  auto bundleType = tupleData.getType().dyn_cast<BundleType>();

  // Create subfields for each bundle element
  ValueVector elements;
  for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i)
    elements.push_back(rewriter.create<SubfieldOp>(insertLoc, tupleData, i));

  // Collect all output ports.
  SmallVector<ValueVector *, 4> outputs;
  for (int i = 1, e = portNum - 2; i < e; ++i)
    outputs.push_back(&portList[i]);

  // Connect each bundle element to the corresponding output
  for (auto &&[element, output] : llvm::zip(elements, outputs))
    rewriter.create<ConnectOp>(insertLoc, (*output)[2], element);

  auto clock = portList[portNum - 2][0];
  auto reset = portList[portNum - 1][0];

  return buildForkLogic(&tuple, outputs, clock, reset, true);
}

//===----------------------------------------------------------------------===//
// Old Operation Conversion Functions
//===----------------------------------------------------------------------===//

/// Create InstanceOp in the top-module. This will be called after the
/// corresponding sub-module and combinational logic are created.
static void createInstOp(Operation *oldOp, FModuleLike subModuleOp,
                         FModuleOp topModuleOp, unsigned clockDomain,
                         ConversionPatternRewriter &rewriter,
                         NameUniquer &instanceNameGen) {
  rewriter.setInsertionPointAfter(oldOp);

  // Create a instance operation.
  auto instanceOp = rewriter.create<firrtl::InstanceOp>(
      oldOp->getLoc(), subModuleOp, instanceNameGen(oldOp));

  // Connect the new created instance with its predecessors and successors in
  // the top-module.
  unsigned portIndex = 0;
  for (auto result : instanceOp.getResults()) {
    unsigned numIns = oldOp->getNumOperands();
    unsigned numArgs = numIns + oldOp->getNumResults();

    auto topArgs = topModuleOp.getBodyBlock()->getArguments();
    auto firstClock = std::find_if(topArgs.begin(), topArgs.end(),
                                   [](BlockArgument &arg) -> bool {
                                     return arg.getType().isa<ClockType>();
                                   });
    assert(firstClock != topArgs.end() && "Expected a clock signal");
    unsigned firstClkIdx = std::distance(topArgs.begin(), firstClock);

    if (portIndex < numIns) {
      // Connect input ports.
      rewriter.create<ConnectOp>(oldOp->getLoc(), result,
                                 oldOp->getOperand(portIndex));
    } else if (portIndex < numArgs) {
      // Connect output ports.
      Value newResult = oldOp->getResult(portIndex - numIns);
      newResult.replaceAllUsesWith(result);
    } else {
      // Connect clock or reset signal(s).
      unsigned clkOrResetIdx =
          firstClkIdx + 2 * clockDomain + portIndex - numArgs;
      assert(topArgs.size() > clkOrResetIdx);
      auto signal = topArgs[clkOrResetIdx];
      rewriter.create<ConnectOp>(oldOp->getLoc(), result, signal);
    }
    ++portIndex;
  }
  rewriter.eraseOp(oldOp);
}

static void convertReturnOp(Operation *oldOp, FModuleOp topModuleOp,
                            handshake::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(oldOp);
  unsigned numIns = funcOp.getNumArguments();

  // Connect each operand of the old return operation with the corresponding
  // output ports.
  unsigned argIndex = 0;
  for (auto result : oldOp->getOperands()) {
    rewriter.create<ConnectOp>(
        oldOp->getLoc(), topModuleOp.getArgument(numIns + argIndex), result);
    ++argIndex;
  }

  rewriter.eraseOp(oldOp);
}

static std::string getInstanceName(Operation *op) {
  auto instOp = dyn_cast<handshake::InstanceOp>(op);
  return instOp ? instOp.getModule().str() : getBareSubModuleName(op);
}

//===----------------------------------------------------------------------===//
// HandshakeToFIRRTL lowering Pass
//===----------------------------------------------------------------------===//

/// Process of lowering:
///
/// 0)  Create and go into a new FIRRTL circuit;
/// 1)  Create and go into a new FIRRTL top-module;
/// 2)  Inline Handshake FuncOp region into the FIRRTL top-module;
/// 3)  Traverse and convert each Standard or Handshake operation:
///   i)    Check if an identical sub-module exists. If so, skip to vi);
///   ii)   Create and go into a new FIRRTL sub-module;
///   iii)  Extract data (if applied), valid, and ready subfield from each port
///         of the sub-module;
///   iv)   Build combinational logic;
///   v)    Exit the sub-module and go back to the top-module;
///   vi)   Create an new instance for the sub-module;
///   vii)  Connect the instance with its predecessors and successors;
/// 4)  Erase the Handshake FuncOp.
///
/// createTopModuleOp():  1) and 2)
/// checkSubModuleOp():   3.i)
/// createSubModuleOp():  3.ii)
/// extractSubfields():   3.iii)
/// build*Logic():        3.iv)
/// createInstOp():       3.v), 3.vi), and 3.vii)
///
/// Please refer to test_addi.mlir test case.
struct HandshakeFuncOpLowering : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;
  HandshakeFuncOpLowering(MLIRContext *context, CircuitOp circuitOp,
                          bool enableFlattening)
      : OpConversionPattern<handshake::FuncOp>(context), circuitOp(circuitOp),
        setFlattenAttr(enableFlattening) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointToStart(circuitOp.getBodyBlock());
    if (funcOp.isExternal()) {
      if (failed(createTopModuleOp<FExtModuleOp>(funcOp, /*numClocks=*/1,
                                                 rewriter, setFlattenAttr)))
        return failure();
      rewriter.eraseOp(funcOp);
      return success();
    }

    // Adds a boolean "control" attribute for all Handshake operations
    // As handshake operations get lowered to FIRRTL (in particular, as
    // NonType's get lowered), the logic that determines whether an operation
    // is a control operation may no longer give the right answer. We therefore
    // cache the "control-ness" of each operation before any modification to the
    // operation and then refer to that attribute instead during lowering.
    for (auto &op : funcOp.getOps()) {
      auto ctrl = dyn_cast<handshake::ControlInterface>(op);
      op.setAttr("control", BoolAttr::get(rewriter.getContext(),
                                          ctrl && ctrl.isControl()));
    }

    auto maybeTopModuleOp = createTopModuleOp<FModuleOp>(
        funcOp, /*numClocks=*/1, rewriter, setFlattenAttr);
    if (failed(maybeTopModuleOp))
      return failure();
    auto topModuleOp = *maybeTopModuleOp;
    inlineFuncRegion(funcOp, topModuleOp, rewriter);

    NameUniquer instanceUniquer = [&](Operation *op) {
      std::string instName = getInstanceName(op);

      if (auto idAttr = op->getAttrOfType<IntegerAttr>("handshake_id");
          idAttr) {
        // We use a special naming convention for operations which have a
        // 'handshake_id' attribute.
        instName += "_id" + std::to_string(idAttr.getValue().getZExtValue());
      } else {
        // Fallback to just prefixing with an integer.
        instName += std::to_string(instanceNameCntr[instName]++);
      }

      return instName;
    };

    // Traverse and convert each operation in funcOp.
    for (Operation &op : *topModuleOp.getBodyBlock()) {
      if (isa<handshake::ReturnOp>(op))
        convertReturnOp(&op, topModuleOp, funcOp, rewriter);

      // This branch takes care of all non-timing operations that require to
      // be instantiated in the top-module.
      else if (op.getDialect()->getNamespace() != "firrtl") {
        FModuleLike subModuleOp = checkSubModuleOp(circuitOp, &op);

        // Check if the sub-module already exists.
        if (!subModuleOp) {
          FModuleOp newSubModuleOp =
              createSubModuleOp(topModuleOp, &op, rewriter);
          subModuleOp = newSubModuleOp;

          Location insertLoc = newSubModuleOp.getLoc();
          auto *bodyBlock = newSubModuleOp.getBodyBlock();
          rewriter.setInsertionPoint(bodyBlock, bodyBlock->end());

          ValueVectorList portList =
              extractSubfields(newSubModuleOp, insertLoc, rewriter);

          if (HandshakeBuilder(circuitOp, portList, insertLoc, rewriter)
                  .dispatchHandshakeVisitor(&op)) {
          } else if (StdExprBuilder(portList, insertLoc, rewriter)
                         .dispatchStdExprVisitor(&op)) {
          } else
            return op.emitError("unsupported operation type");
        }

        // Instantiate the new created sub-module.
        createInstOp(&op, subModuleOp, topModuleOp, /*clockDomain=*/0, rewriter,
                     instanceUniquer);
      }
    }
    rewriter.eraseOp(funcOp);

    legalizeFModule(topModuleOp);

    return success();
  }

private:
  /// Maintain a map from module names to the # of times the module has been
  /// instantiated inside this module. This is used to generate unique names for
  /// each instance.
  mutable std::map<std::string, unsigned> instanceNameCntr;

  /// Top level FIRRTL circuit operation, which we'll emit into. Marked as
  /// mutable due to circuitOp.getBody() being non-const.
  mutable CircuitOp circuitOp;

  /// If true, the top-level module will have the FIRRTL inlining attribute set.
  /// All module instances will be recursively inlined into the top module.
  bool setFlattenAttr;
};

namespace {
class HandshakeToFIRRTLPass
    : public HandshakeToFIRRTLBase<HandshakeToFIRRTLPass> {
public:
  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = op.getContext();

    // Lowering to FIRRTL requires that every value is used exactly once. Check
    // whether this precondition is met, and if not, exit.
    if (llvm::any_of(op.getOps<handshake::FuncOp>(), [](auto f) {
          return failed(verifyAllValuesHasOneUse(f));
        })) {
      signalPassFailure();
      return;
    }

    // Resolve the instance graph to get a top-level module.
    std::string topLevel;
    handshake::InstanceGraph uses;
    SmallVector<std::string> sortedFuncs;
    if (resolveInstanceGraph(op, uses, topLevel, sortedFuncs).failed()) {
      signalPassFailure();
      return;
    }

    // Create FIRRTL circuit op.
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(op.getBody());
    auto circuitOp =
        builder.create<CircuitOp>(op.getLoc(), builder.getStringAttr(topLevel));

    ConversionTarget target(getContext());
    target.addLegalDialect<FIRRTLDialect>();
    target.addIllegalDialect<handshake::HandshakeDialect>();

    // Convert the handshake.func operations in post-order wrt. the instance
    // graph. This ensures that any referenced submodules (through
    // handshake.instance) has already been lowered, and their FIRRTL module
    // equivalents are available.
    for (auto funcName : llvm::reverse(sortedFuncs)) {
      RewritePatternSet patterns(op.getContext());
      patterns.insert<HandshakeFuncOpLowering>(op.getContext(), circuitOp,
                                               enableFlattening);
      auto funcOp = op.lookupSymbol(funcName);
      assert(funcOp && "Symbol not found in module!");
      if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
        funcOp->emitOpError() << "error during conversion";
        return;
      }
    }
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createHandshakeToFIRRTLPass() {
  return std::make_unique<HandshakeToFIRRTLPass>();
}
