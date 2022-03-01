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
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <variant>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

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
  if (auto channelType = opType.dyn_cast<esi::ChannelPort>())
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
        return DiscriminatingTypes{{}, {memOp.memRefType().getElementType()}};
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

// Wraps a type into an ESI ChannelPort type. The inner type is converted to
// ensure comprehensability by the RTL dialects.
static Type esiWrapper(Type t) {
  // Translate none- and index types to something HW understands.
  if (t.isa<NoneType>())
    t = IntegerType::get(t.getContext(), 0);
  else if (t.isa<IndexType>())
    t = IntegerType::get(t.getContext(), 64);

  return esi::ChannelPort::get(t.getContext(), t);
};

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
        llvm::enumerate(op.getType().getInputs()), std::back_inserter(inputs),
        [&](auto it) { return builder.getStringAttr(inF(it.index())); });
    llvm::transform(
        llvm::enumerate(op.getType().getResults()), std::back_inserter(outputs),
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
    subModuleName += "_id" + std::to_string(memOp.id());

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

static llvm::SmallVector<PortInfo>
getPortInfoForOp(ConversionPatternRewriter &rewriter, Operation *op,
                 TypeRange inputs, TypeRange outputs) {
  llvm::SmallVector<PortInfo> ports;
  HandshakePortNameGenerator portNames(op);

  // Add all inputs of funcOp.
  unsigned inIdx = 0;
  for (auto &arg : llvm::enumerate(inputs)) {
    ports.push_back({portNames.inputName(arg.index()), PortDirection::INPUT,
                     esiWrapper(arg.value()), arg.index(), StringAttr{}});
    inIdx++;
  }

  // Add all outputs of funcOp.
  for (auto &res : llvm::enumerate(outputs)) {
    ports.push_back({portNames.outputName(res.index()), PortDirection::OUTPUT,
                     esiWrapper(res.value()), res.index(), StringAttr{}});
  }

  // Add clock and reset signals.
  if (op->hasTrait<mlir::OpTrait::HasClock>()) {
    ports.push_back({rewriter.getStringAttr("clock"), PortDirection::INPUT,
                     rewriter.getI1Type(), inIdx++, StringAttr{}});
    ports.push_back({rewriter.getStringAttr("reset"), PortDirection::INPUT,
                     rewriter.getI1Type(), inIdx, StringAttr{}});
  }

  return ports;
}

/// Returns a vector of PortInfo's which defines the FIRRTL interface of the
/// to-be-converted op.
static llvm::SmallVector<PortInfo>
getPortInfoForOp(ConversionPatternRewriter &rewriter, Operation *op) {
  return getPortInfoForOp(rewriter, op, op->getOperandTypes(),
                          op->getResultTypes());
}

/// All standard expressions and handshake elastic components will be converted
/// to a HW sub-module and be instantiated in the top-module.
static HWModuleExternOp createSubModuleOp(ModuleOp parentModule,
                                          Operation *oldOp,
                                          ConversionPatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(parentModule.getBody());
  auto ports = getPortInfoForOp(rewriter, oldOp);
  // todo: HWModuleOp; for this initial commit we'll leave this as an extern op.
  return rewriter.create<HWModuleExternOp>(
      parentModule.getLoc(), rewriter.getStringAttr(getSubModuleName(oldOp)),
      ports);
}

//===----------------------------------------------------------------------===//
// HW Top-module Related Functions
//===----------------------------------------------------------------------===//

static hw::HWModuleOp createTopModuleOp(handshake::FuncOp funcOp,
                                        ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<PortInfo, 8> ports =
      getPortInfoForOp(rewriter, funcOp, funcOp.getType().getInputs(),
                       funcOp.getType().getResults());

  // Create a HW module.
  auto hwModuleOp = rewriter.create<hw::HWModuleOp>(
      funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()), ports);

  // Remove the default created hw_output operation.
  auto outputOps = hwModuleOp.getOps<hw::OutputOp>();
  assert(std::distance(outputOps.begin(), outputOps.end()) == 1 &&
         "Expected exactly 1 default created hw_output operation");
  rewriter.eraseOp(*outputOps.begin());

  return hwModuleOp;
}

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

namespace {

using TypeTransformer = llvm::function_ref<Type(Type)>;
static Type defaultTypeTransformer(Type t) { return t; }

/// The ValueMapping class facilitates the definition and connection of SSA
/// def-use chains between two separate regions - a 'from' region (defining
/// use-def chains) and a 'to' region (where new operations are created based on
/// the 'from' region).´
class ValueMapping {
public:
  explicit ValueMapping(BackedgeBuilder &bb) : bb(bb) {}

  // Get the mapped value of value 'from'. If no mapping has been registered, a
  // new backedge is created. The type of the mapped value may optionally be
  // modified through the 'typeTransformer'.
  Value get(Value from,
            TypeTransformer typeTransformer = defaultTypeTransformer) {
    if (mapping.count(from) == 0) {
      // Create a backedge which will be resolved at a later time once all
      // operands are created.
      mapping[from] = bb.get(typeTransformer(from.getType()));
    }
    auto operandMapping = mapping[from];
    Value mappedOperand;
    if (auto *v = std::get_if<Value>(&operandMapping))
      mappedOperand = *v;
    else
      mappedOperand = std::get<Backedge>(operandMapping);
    return mappedOperand;
  }

  llvm::SmallVector<Value>
  get(ValueRange from,
      TypeTransformer typeTransformer = defaultTypeTransformer) {
    llvm::SmallVector<Value> to;
    for (auto f : from)
      to.push_back(get(f, typeTransformer));
    return to;
  }

  // Set the mapped value of 'from' to 'to'. If 'from' is already mapped to a
  // backedge, replaces that backedge with 'to'.
  void set(Value from, Value to) {
    auto it = mapping.find(from);
    if (it != mapping.end()) {
      if (auto *backedge = std::get_if<Backedge>(&it->second)) {
        backedge->setValue(to);
      } else {
        assert(false && "'from' was already mapped to a final value!");
      }
    }
    // Register the new mapping
    mapping[from] = to;
  }

  void set(ValueRange from, ValueRange to) {
    assert(from.size() == to.size() &&
           "Expected # of 'from' values and # of 'to' values to be identical.");
    for (auto [f, t] : llvm::zip(from, to))
      set(f, t);
  }

private:
  BackedgeBuilder &bb;
  DenseMap<Value, std::variant<Value, Backedge>> mapping;
};

// Shared state used by various functions; captured in a struct to reduce the
// number of arguments that we have to pass around.
struct HandshakeLoweringState {
  ValueMapping &mapping;
  ModuleOp parentModule;
  hw::HWModuleOp hwModuleOp;
  NameUniquer nameUniquer;
  Value clock;
  Value reset;
};

static Operation *createOpInHWModule(ConversionPatternRewriter &rewriter,
                                     HandshakeLoweringState &ls,
                                     Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToEnd(ls.hwModuleOp.getBodyBlock());

  // Create a mapping between the operands of 'op' and the replacement operands
  // in the target hwModule. For any missing operands, we create a new backedge.
  llvm::SmallVector<Value> hwOperands =
      ls.mapping.get(op->getOperands(), esiWrapper);

  // Add clock and reset if needed.
  if (op->hasTrait<mlir::OpTrait::HasClock>())
    hwOperands.append({ls.clock, ls.reset});

  // Check if a sub-module for the operation already exists.
  Operation *subModuleSymOp = checkSubModuleOp(ls.parentModule, op);
  if (!subModuleSymOp) {
    subModuleSymOp = createSubModuleOp(ls.parentModule, op, rewriter);
    // TODO: fill the subModuleSymOp with the meat of the handshake
    // operations.
  }

  // Instantiate the new created sub-module.
  rewriter.setInsertionPointToEnd(ls.hwModuleOp.getBodyBlock());
  auto submoduleInstanceOp = rewriter.create<hw::InstanceOp>(
      op->getLoc(), subModuleSymOp, rewriter.getStringAttr(ls.nameUniquer(op)),
      hwOperands);

  // Resolve any previously created backedges that referred to the results of
  // 'op'.
  ls.mapping.set(op->getResults(), submoduleInstanceOp.getResults());

  return submoduleInstanceOp;
}

static void convertReturnOp(handshake::ReturnOp op, HandshakeLoweringState &ls,
                            ConversionPatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToEnd(ls.hwModuleOp.getBodyBlock());
  rewriter.create<hw::OutputOp>(op.getLoc(), ls.mapping.get(op.getOperands()));
}

struct HandshakeFuncOpLowering : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;
  HandshakeFuncOpLowering(MLIRContext *context)
      : OpConversionPattern<handshake::FuncOp>(context) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyHandshakeFuncOp(funcOp)))
      return failure();

    ModuleOp parentModule = funcOp->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(funcOp->getBlock());
    HWModuleOp topModuleOp = createTopModuleOp(funcOp, rewriter);
    Value clockPort =
        topModuleOp.getArgument(topModuleOp.getNumArguments() - 2);
    Value resetPort =
        topModuleOp.getArgument(topModuleOp.getNumArguments() - 1);

    // Create a uniquer function for creating instance names.
    NameUniquer instanceUniquer = [&](Operation *op) {
      std::string instName = getCallName(op);
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

    // Initialize the Handshake lowering state.
    BackedgeBuilder bb(rewriter, funcOp.getLoc());
    ValueMapping valueMapping(bb);
    auto ls = HandshakeLoweringState{valueMapping,    parentModule, topModuleOp,
                                     instanceUniquer, clockPort,    resetPort};

    // Extend value mapping with input arguments. Drop the 2 last inputs from
    // the HW module (clock and reset).
    valueMapping.set(funcOp.getArguments(),
                     topModuleOp.getArguments().drop_back(2));

    // Traverse and convert each operation in funcOp.
    for (Operation &op : funcOp.front()) {
      if (isa<hw::OutputOp>(op)) {
        // Skip the default created HWModule terminator, for now.
        continue;
      }
      if (auto returnOp = dyn_cast<handshake::ReturnOp>(op)) {
        convertReturnOp(returnOp, ls, rewriter);
      } else {
        // Regular operation.
        createOpInHWModule(rewriter, ls, &op);
      }
    }
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  /// Maintain a map from module names to the # of times the module has been
  /// instantiated inside this module. This is used to generate unique names for
  /// each instance.
  mutable std::map<std::string, unsigned> instanceNameCntr;
};

class HandshakeToHWPass : public HandshakeToHWBase<HandshakeToHWPass> {
public:
  void runOnOperation() override {
    auto op = getOperation();

    // Lowering to HW requires that every value is used exactly once. Check
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

    ConversionTarget target(getContext());
    target.addLegalDialect<HWDialect>();
    target.addIllegalDialect<handshake::HandshakeDialect>();

    // Convert the handshake.func operations in post-order wrt. the instance
    // graph. This ensures that any referenced submodules (through
    // handshake.instance) has already been lowered, and their HW module
    // equivalents are available.
    for (auto &funcName : llvm::reverse(sortedFuncs)) {
      RewritePatternSet patterns(op.getContext());
      patterns.insert<HandshakeFuncOpLowering>(op.getContext());
      auto *funcOp = op.lookupSymbol(funcName);
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

std::unique_ptr<mlir::Pass> circt::createHandshakeToHWPass() {
  return std::make_unique<HandshakeToHWPass>();
}
