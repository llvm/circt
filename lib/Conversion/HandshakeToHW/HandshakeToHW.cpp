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

namespace {
using NameUniquer = std::function<std::string(Operation *)>;
using DiscriminatingTypes = std::pair<SmallVector<Type>, SmallVector<Type>>;
} // namespace

static bool isMemrefType(Type t) { return t.isa<mlir::MemRefType>(); }

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

// Wraps a type into an ESI ChannelType type. The inner type is converted to
// ensure comprehensability by the RTL dialects.
static Type esiWrapper(Type t) {
  // Translate index types to something HW understands.
  if (t.isa<IndexType>())
    t = IntegerType::get(t.getContext(), 64);

  return esi::ChannelType::get(t.getContext(), t);
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
} // namespace

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

static llvm::SmallVector<PortInfo> getPortInfoForOp(OpBuilder &b, Operation *op,
                                                    TypeRange inputs,
                                                    TypeRange outputs) {
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
    ports.push_back({b.getStringAttr("clock"), PortDirection::INPUT,
                     b.getI1Type(), inIdx++, StringAttr{}});
    ports.push_back({b.getStringAttr("reset"), PortDirection::INPUT,
                     b.getI1Type(), inIdx, StringAttr{}});
  }

  return ports;
}

static llvm::SmallVector<PortInfo> getPortInfoForOp(OpBuilder &b,
                                                    Operation *op) {
  return getPortInfoForOp(b, op, op->getOperandTypes(), op->getResultTypes());
}

/// All standard expressions and handshake elastic components will be converted
/// to a HW sub-module and be instantiated in the top-module.
static HWModuleExternOp
createSubModuleOp(ModuleOp parentModule, Operation *oldOp, OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(parentModule.getBody());
  auto ports = getPortInfoForOp(builder, oldOp);
  // todo: HWModuleOp; for this initial commit we'll leave this as an extern op.
  return builder.create<HWModuleExternOp>(
      parentModule.getLoc(), builder.getStringAttr(getSubModuleName(oldOp)),
      ports);
}

/// Check whether a submodule with the same name has been created elsewhere in
/// the top level module. Return the matched module operation if true, else,
/// create the module, and return the created module.
static hw::HWModuleLike getOrCreateSubModuleForOp(OpBuilder &builder,
                                                  mlir::ModuleOp parentModule,
                                                  Operation *op) {
  auto submoduleName = getSubModuleName(op);
  if (auto mod = parentModule.lookupSymbol<HWModuleOp>(submoduleName))
    return mod;
  if (auto mod = parentModule.lookupSymbol<HWModuleExternOp>(submoduleName))
    return mod;

  assert(!isa<handshake::InstanceOp>(op) &&
         "handshake.instance target modules should always have been lowered "
         "before the modules that reference them!");

  return createSubModuleOp(parentModule, op, builder);
}

//===----------------------------------------------------------------------===//
// HW top-module related functions
//===----------------------------------------------------------------------===//

static hw::HWModuleOp createTopModuleOp(handshake::FuncOp funcOp,
                                        OpBuilder &b) {
  llvm::SmallVector<PortInfo, 8> ports = getPortInfoForOp(
      b, funcOp, funcOp.getArgumentTypes(), funcOp.getResultTypes());

  // Create a HW module.
  auto hwModuleOp = b.create<hw::HWModuleOp>(
      funcOp.getLoc(), b.getStringAttr(funcOp.getName()), ports);

  // Remove the default created hw_output operation.
  auto outputOps = hwModuleOp.getOps<hw::OutputOp>();
  assert(std::distance(outputOps.begin(), outputOps.end()) == 1 &&
         "Expected exactly 1 default created hw_output operation");
  (*outputOps.begin()).erase();

  return hwModuleOp;
}

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

// Shared state used by various functions; captured in a struct to reduce the
// number of arguments that we have to pass around.
struct HandshakeLoweringState {
  ValueMapper &mapper;
  ModuleOp parentModule;
  hw::HWModuleOp hwModuleOp;
  NameUniquer nameUniquer;
  Value clock;
  Value reset;
};

} // namespace

static Operation *createOpInHWModule(OpBuilder &b, HandshakeLoweringState &ls,
                                     Operation *op) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(ls.hwModuleOp.getBodyBlock());

  // Create a mapper between the operands of 'op' and the replacement operands
  // in the target hwModule. For any missing operands, we create a new backedge.
  llvm::SmallVector<Value> hwOperands =
      ls.mapper.get(op->getOperands(), esiWrapper);

  // Add clock and reset if needed.
  if (op->hasTrait<mlir::OpTrait::HasClock>())
    hwOperands.append({ls.clock, ls.reset});

  // Instantiate the sub-module.
  hw::HWModuleLike subModuleOp =
      getOrCreateSubModuleForOp(b, ls.parentModule, op);
  b.setInsertionPointToEnd(ls.hwModuleOp.getBodyBlock());
  auto submoduleInstanceOp =
      b.create<hw::InstanceOp>(op->getLoc(), subModuleOp,
                               b.getStringAttr(ls.nameUniquer(op)), hwOperands);

  // Resolve any previously created backedges that referred to the results of
  // 'op'.
  ls.mapper.set(op->getResults(), submoduleInstanceOp.getResults());

  return submoduleInstanceOp;
}

static void convertReturnOp(handshake::ReturnOp op, HandshakeLoweringState &ls,
                            OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(ls.hwModuleOp.getBodyBlock());
  b.create<hw::OutputOp>(op.getLoc(), ls.mapper.get(op.getOperands()));
}

static LogicalResult convertHandshakeFuncOp(handshake::FuncOp funcOp,
                                            OpBuilder &b) {
  if (failed(verifyHandshakeFuncOp(funcOp)))
    return failure();
  OpBuilder::InsertionGuard g(b);

  ModuleOp parentModule = funcOp->getParentOfType<ModuleOp>();
  b.setInsertionPointToStart(parentModule.getBody());
  HWModuleOp topModuleOp = createTopModuleOp(funcOp, b);
  Value clockPort = topModuleOp.getArgument(topModuleOp.getNumArguments() - 2);
  Value resetPort = topModuleOp.getArgument(topModuleOp.getNumArguments() - 1);

  /// Maintain a map from module names to the # of times the module has been
  /// instantiated inside this module. This is used to generate unique names for
  /// each instance.
  std::map<std::string, unsigned> instanceNameCntr;

  // Create a uniquer function for creating instance names.
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

  // Initialize the Handshake lowering state.
  BackedgeBuilder bb(b, funcOp.getLoc());
  ValueMapper valuemapper(&bb);
  auto ls = HandshakeLoweringState{valuemapper,     parentModule, topModuleOp,
                                   instanceUniquer, clockPort,    resetPort};

  // Extend value mapper with input arguments. Drop the 2 last inputs from
  // the HW module (clock and reset).
  valuemapper.set(funcOp.getArguments(),
                  topModuleOp.getArguments().drop_back(2));

  // Traverse and convert each operation in funcOp.
  for (Operation &op : funcOp.front()) {
    if (isa<hw::OutputOp>(op)) {
      // Skip the default created HWModule terminator, for now.
      continue;
    }
    if (auto returnOp = dyn_cast<handshake::ReturnOp>(op)) {
      convertReturnOp(returnOp, ls, b);
    } else {
      // Regular operation.
      createOpInHWModule(b, ls, &op);
    }
  }
  funcOp->erase();
  return success();
}

namespace {

class HandshakeToHWPass : public HandshakeToHWBase<HandshakeToHWPass> {
public:
  void runOnOperation() override;
};

void HandshakeToHWPass::runOnOperation() {
  auto op = getOperation();
  OpBuilder builder(op->getContext());

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

  // Convert the handshake.func operations in post-order wrt. the instance
  // graph. This ensures that any referenced submodules (through
  // handshake.instance) has already been lowered, and their HW module
  // equivalents are available.
  std::map<std::string, hw::HWModuleLike> opNameToModule;
  for (auto &funcName : llvm::reverse(sortedFuncs)) {
    auto funcOp = op.lookupSymbol<handshake::FuncOp>(funcName);
    assert(funcOp && "Symbol not found in module!");
    if (failed(convertHandshakeFuncOp(funcOp, builder))) {
      signalPassFailure();
      funcOp->emitOpError() << "error during conversion";
      return;
    }
  }
}

} // namespace
std::unique_ptr<mlir::Pass> circt::createHandshakeToHWPass() {
  return std::make_unique<HandshakeToHWPass>();
}
