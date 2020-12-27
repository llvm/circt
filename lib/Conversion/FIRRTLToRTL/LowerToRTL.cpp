//===- LowerToRTL.cpp - FIRRTL to RTL Lowering Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main FIRRTL to RTL Lowering Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/FIRRTLToRTL/FIRRTLToRTL.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/Visitors.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Types.h"
#include "circt/Dialect/SV/Ops.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TinyPtrVector.h"

using namespace circt;
using namespace firrtl;

/// Return the type of the specified value, casted to the template type.
template <typename T = FIRRTLType>
static T getTypeOf(Value v) {
  return v.getType().cast<T>();
}

/// Given a FIRRTL type, return the corresponding type for the RTL dialect.
/// This returns a null type if it cannot be lowered.
static Type lowerType(Type type) {
  auto firType = type.dyn_cast<FIRRTLType>();
  if (!firType)
    return {};

  // Ignore flip types.
  firType = firType.getPassiveType();

  auto width = firType.getBitWidthOrSentinel();
  if (width >= 0) // IntType, analog with known width, clock, etc.
    return IntegerType::get(type.getContext(), width);

  return {};
}

/// Given two FIRRTL integer types, return the widest one.
static IntType getWidestIntType(Type t1, Type t2) {
  auto t1c = t1.cast<IntType>(), t2c = t2.cast<IntType>();
  return t2c.getWidth() > t1c.getWidth() ? t2c : t1c;
}

/// Cast from a standard type to a FIRRTL type, potentially with a flip.
static Value castToFIRRTLType(Value val, Type type,
                              ImplicitLocOpBuilder &builder) {
  auto firType = type.cast<FIRRTLType>();

  // If this was an Analog type, it will be converted to an InOut type.
  if (type.isa<AnalogType>())
    return builder.createOrFold<AnalogInOutCastOp>(firType, val);

  val = builder.createOrFold<StdIntCastOp>(firType.getPassiveType(), val);

  // Handle the flip type if needed.
  if (type != val.getType())
    val = builder.createOrFold<AsNonPassivePrimOp>(firType, val);
  return val;
}

/// Cast from a FIRRTL type (potentially with a flip) to a standard type.
static Value castFromFIRRTLType(Value val, Type type,
                                ImplicitLocOpBuilder &builder) {
  // Strip off Flip type if needed.
  val = builder.createOrFold<AsPassivePrimOp>(val);
  return builder.createOrFold<StdIntCastOp>(type, val);
}

static Value extendOrTruncateFIRRTL(Value val, IntType destTy,
                                    ImplicitLocOpBuilder &builder) {
  auto srcTy = val.getType().cast<IntType>();
  assert(srcTy.hasWidth() && destTy.hasWidth() &&
         "only works with width-inferred integer values");

  if (srcTy.getWidthOrSentinel() == destTy.getWidthOrSentinel())
    return val;

  if (srcTy.getWidthOrSentinel() > destTy.getWidthOrSentinel())
    return builder.createOrFold<TailPrimOp>(destTy, val,
                                            destTy.getWidthOrSentinel());
  return builder.createOrFold<PadPrimOp>(destTy, val,
                                         destTy.getWidthOrSentinel());
}

//===----------------------------------------------------------------------===//
// firrtl.module Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLModuleLowering
    : public LowerFIRRTLToRTLModuleBase<FIRRTLModuleLowering> {

  void runOnOperation() override;

private:
  LogicalResult lowerPorts(ArrayRef<ModulePortInfo> firrtlPorts,
                           SmallVectorImpl<rtl::ModulePortInfo> &ports,
                           Operation *moduleOp);
  rtl::RTLModuleOp lowerModule(FModuleOp oldModule, Block *topLevelModule);
  rtl::RTLExternModuleOp lowerExtModule(FExtModuleOp oldModule,
                                        Block *topLevelModule);

  void lowerModuleBody(FModuleOp oldModule,
                       DenseMap<Operation *, Operation *> &oldToNewModuleMap);

  void lowerInstance(InstanceOp instance,
                     DenseMap<Operation *, Operation *> &oldToNewModuleMap);
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLToRTLModulePass() {
  return std::make_unique<FIRRTLModuleLowering>();
}

/// Run on the firrtl.circuit operation, lowering any firrtl.module operations
/// it contains.
void FIRRTLModuleLowering::runOnOperation() {
  // We run on the top level modules in the IR blob.  Start by finding the
  // firrtl.circuit within it.  If there is none, then there is nothing to do.
  auto *moduleBody = getOperation().getBody();

  // Find the single firrtl.circuit in the module.
  CircuitOp circuit;
  for (auto &op : *moduleBody) {
    if ((circuit = dyn_cast<CircuitOp>(&op)))
      break;
  }

  if (!circuit)
    return;

  auto *circuitBody = circuit.getBody();

  // Keep track of the mapping from old to new modules.  The result may be null
  // if lowering failed.
  DenseMap<Operation *, Operation *> oldToNewModuleMap;

  // Iterate through each operation in the circuit body, transforming any
  // FModule's we come across.
  for (auto &op : circuitBody->getOperations()) {
    if (auto module = dyn_cast<FModuleOp>(op)) {
      oldToNewModuleMap[&op] = lowerModule(module, moduleBody);
      continue;
    }

    if (auto extModule = dyn_cast<FExtModuleOp>(op)) {
      oldToNewModuleMap[&op] = lowerExtModule(extModule, moduleBody);
      continue;
    }

    if (isa<DoneOp>(op))
      continue;

    // Otherwise we don't know what this is.  We are just going to drop it,
    // but emit an error so the client has some chance to know that this is
    // going to happen.
    op.emitError("unexpected operation '")
        << op.getName() << "' in a firrtl.circuit";
  }

  // Now that we've lowered all of the modules, move the bodies over and update
  // any instances that refer to the old modules.  Only rtl.instance can refer
  // to an rtl.module, not a firrtl.instance.
  //
  // TODO: This is a trivially parallelizable for loop.  We should be able to
  // process each module in parallel.
  for (auto &op : circuitBody->getOperations()) {
    if (auto module = dyn_cast<FModuleOp>(op))
      lowerModuleBody(module, oldToNewModuleMap);
  }

  // Finally delete all the old modules.
  for (auto oldNew : oldToNewModuleMap)
    oldNew.first->erase();

  // Now that the modules are moved over, remove the Circuit.  We pop the 'main
  // module' specified in the Circuit into an attribute on the top level module.
  getOperation().setAttr("firrtl.mainModule",
                         StringAttr::get(circuit.name(), circuit.getContext()));
  circuit.erase();
}

LogicalResult
FIRRTLModuleLowering::lowerPorts(ArrayRef<ModulePortInfo> firrtlPorts,
                                 SmallVectorImpl<rtl::ModulePortInfo> &ports,
                                 Operation *moduleOp) {

  ports.reserve(firrtlPorts.size());
  size_t numArgs = 0;
  size_t numResults = 0;
  for (auto firrtlPort : firrtlPorts) {
    rtl::ModulePortInfo rtlPort;

    rtlPort.name = firrtlPort.name;
    rtlPort.type = lowerType(firrtlPort.type);

    // We can't lower all types, so make sure to cleanly reject them.
    if (!rtlPort.type) {
      moduleOp->emitError("cannot lower this port type to RTL");
      return failure();
    }

    // Figure out the direction of the port.
    if (firrtlPort.type.isa<AnalogType>()) {
      // If the port is analog, then it is implicitly inout.
      rtlPort.type = rtl::InOutType::get(rtlPort.type);
      rtlPort.direction = rtl::PortDirection::INOUT;
      rtlPort.argNum = numArgs++;
    } else if (firrtlPort.isOutput()) {
      rtlPort.direction = rtl::PortDirection::OUTPUT;
      rtlPort.argNum = numResults++;
    } else if (firrtlPort.isInput()) {
      rtlPort.direction = rtl::PortDirection::INPUT;
      rtlPort.argNum = numArgs++;
    } else {
      // This isn't currently expressible in low-firrtl, due to bundle types
      // being lowered.
      rtlPort.direction = rtl::PortDirection::INOUT;
      rtlPort.argNum = numArgs++;
    }
    ports.push_back(rtlPort);
  }
  return success();
}

rtl::RTLExternModuleOp
FIRRTLModuleLowering::lowerExtModule(FExtModuleOp oldModule,
                                     Block *topLevelModule) {
  // Map the ports over, lowering their types as we go.
  SmallVector<ModulePortInfo, 8> firrtlPorts;
  oldModule.getPortInfo(firrtlPorts);
  SmallVector<rtl::ModulePortInfo, 8> ports;
  if (failed(lowerPorts(firrtlPorts, ports, oldModule)))
    return {};

  StringRef verilogName;
  if (auto defName = oldModule.defname())
    verilogName = defName.getValue();

  // Build the new rtl.module op.
  OpBuilder builder(topLevelModule->getTerminator());
  auto nameAttr = builder.getStringAttr(oldModule.getName());
  return builder.create<rtl::RTLExternModuleOp>(oldModule.getLoc(), nameAttr,
                                                ports, verilogName);
}

/// Run on each firrtl.module, transforming it from an firrtl.module into an
/// rtl.module, then deleting the old one.
rtl::RTLModuleOp FIRRTLModuleLowering::lowerModule(FModuleOp oldModule,
                                                   Block *topLevelModule) {
  // Map the ports over, lowering their types as we go.
  SmallVector<ModulePortInfo, 8> firrtlPorts;
  oldModule.getPortInfo(firrtlPorts);
  SmallVector<rtl::ModulePortInfo, 8> ports;
  if (failed(lowerPorts(firrtlPorts, ports, oldModule)))
    return {};

  // Build the new rtl.module op.
  OpBuilder builder(topLevelModule->getTerminator());
  auto nameAttr = builder.getStringAttr(oldModule.getName());
  return builder.create<rtl::RTLModuleOp>(oldModule.getLoc(), nameAttr, ports);
}

/// Given a value of flip type, check to see if all of the uses of it are
/// connects.  If so, remove the connects and return the value being connected
/// to it.  If this isn't a situation we can handle, just return null.
///
/// This can happen when there are no connects to the value, or if
/// firrtl.invalid is used.  The 'mergePoint' location is where a 'rtl.merge'
/// operation should be inserted if needed.
static Value tryToConnectsToFlipValue(Value portValue, Operation *insertPoint) {
  SmallVector<ConnectOp, 2> connects;
  for (auto *use : portValue.getUsers()) {
    // We only know about 'connect' uses.
    auto connect = dyn_cast<ConnectOp>(use);
    if (!connect)
      return {};

    connects.push_back(connect);
  }

  // We don't have an RTL equivalent of "poison" so just don't special case the
  // case where there are no connects other uses of an output.
  if (connects.empty())
    return {};

  // Convert each connect into an extended version of its operand being output.
  SmallVector<Value, 2> results;

  for (auto connect : connects) {
    ImplicitLocOpBuilder builder(connect.getLoc(), connect);
    auto connectSrc = connect.src();

    // Convert fliped sources to passive sources.
    if (!connectSrc.getType().cast<FIRRTLType>().isPassive())
      connectSrc = builder.createOrFold<AsPassivePrimOp>(connectSrc);

    // We know it must be the destination operand due to the types, but the
    // source may not match the destination width.
    auto destTy = portValue.getType().cast<FIRRTLType>().getPassiveType();
    if (destTy != connectSrc.getType()) {
      // The only type mismatch we can have is due to integer width differences.
      // FIXME: connects shouldn't allow truncates, so this should just be an
      // extension.
      connectSrc =
          extendOrTruncateFIRRTL(connectSrc, destTy.cast<IntType>(), builder);
    }

    // Remove the connect and use its source as the value for the output.
    connect.erase();
    results.push_back(connectSrc);
  }

  // If there was only one source, just return it.  Otherwise emit an rtl.merge
  // right before the output.
  if (connects.size() == 1)
    return results.back();

  auto builder = ImplicitLocOpBuilder(insertPoint->getLoc(), insertPoint);

  // Annoyingly, we need to convert from FIRRTL type to builtin type to do the
  // merge, then back.
  auto loweredType = lowerType(portValue.getType());
  for (auto &result : results) {
    result = castFromFIRRTLType(result, loweredType, builder);
  }

  auto merge = builder.createOrFold<rtl::MergeOp>(results);
  return castToFIRRTLType(merge, portValue.getType(), builder);
}

/// Now that we have the operations for the rtl.module's corresponding to the
/// firrtl.module's, we can go through and move the bodies over, updating the
/// ports and instances.
void FIRRTLModuleLowering::lowerModuleBody(
    FModuleOp oldModule,
    DenseMap<Operation *, Operation *> &oldToNewModuleMap) {
  auto newModule =
      dyn_cast_or_null<rtl::RTLModuleOp>(oldToNewModuleMap[oldModule]);
  // Don't touch modules if we failed to lower ports.
  if (!newModule)
    return;

  // Start by updating all the firrtl.instance's to be rtl.instance's.
  // Lowering an instance will also delete a bunch of firrtl.subfield
  // operations, so we have to be careful about iterator invalidation.
  for (auto opIt = oldModule.getBodyBlock()->begin(),
            opEnd = oldModule.getBodyBlock()->end();
       opIt != opEnd;) {
    auto instance = dyn_cast<InstanceOp>(&*opIt);
    if (!instance) {
      ++opIt;
      continue;
    }

    // We found an instance - lower it.  On successful return there will be
    // zero uses and we can remove the operation.
    lowerInstance(instance, oldToNewModuleMap);
    ++opIt;
    if (instance.use_empty())
      instance.erase();
  }

  ImplicitLocOpBuilder bodyBuilder(oldModule.getLoc(), newModule.body());

  // Use a placeholder instruction be a cursor that indicates where we want to
  // move the new function body to.  This is important because we insert some
  // ops at the start of the function and some at the end, and the body is
  // currently empty to avoid iterator invalidation.
  auto cursor = bodyBuilder.create<rtl::ConstantOp>(APInt(1, 1));
  bodyBuilder.setInsertionPoint(cursor);

  // Insert argument casts, and re-vector users in the old body to use them.
  SmallVector<ModulePortInfo, 8> ports;
  oldModule.getPortInfo(ports);

  size_t nextNewArg = 0;
  size_t firrtlArg = 0;
  SmallVector<Value, 4> outputs;

  // This is the terminator in the new module.
  auto outputOp = newModule.getBodyBlock()->getTerminator();
  ImplicitLocOpBuilder outputBuilder(oldModule.getLoc(), outputOp);

  for (auto &port : ports) {
    // Inputs and outputs are both modeled as arguments in the FIRRTL level.
    auto oldArg = oldModule.body().getArgument(firrtlArg++);

    Value newArg;
    if (!port.isOutput()) {
      // Inputs and InOuts are modeled as arguments in the result, so we can
      // just map them over.
      newArg = newModule.body().getArgument(nextNewArg++);

      // Cast the argument to the old type, reintroducing sign information in
      // the rtl.module body.
      newArg = castToFIRRTLType(newArg, oldArg.getType(), bodyBuilder);
    } else if (auto value = tryToConnectsToFlipValue(oldArg, outputOp)) {
      // If we were able to find the value being connected to the output,
      // directly use it!
      newArg = value;
      newArg = castFromFIRRTLType(newArg, lowerType(port.type), outputBuilder);
      outputs.push_back(newArg);
      continue;
    } else {
      // Outputs need a temporary wire so they can be connect'd to, which we
      // then return.
      newArg = bodyBuilder.create<WireOp>(port.type, /*name=*/StringAttr());
      auto output =
          castFromFIRRTLType(newArg, lowerType(port.type), outputBuilder);
      outputs.push_back(output);
    }

    // Switch all uses of the old operands to the new ones.
    oldArg.replaceAllUsesWith(newArg);
  }

  // Update the rtl.output terminator with the list of outputs we have.
  outputOp->setOperands(outputs);

  // Finally splice the body over, don't move the old terminator over though.
  auto &oldBlockInstList = oldModule.getBodyBlock()->getOperations();
  auto &newBlockInstList = newModule.getBodyBlock()->getOperations();
  newBlockInstList.splice(Block::iterator(cursor), oldBlockInstList,
                          oldBlockInstList.begin(),
                          std::prev(oldBlockInstList.end()));

  cursor.erase();
}

/// Lower a firrtl.instance operation to an rtl.instance operation.  This is a
/// bit more involved than it sounds because we have to clean up the subfield
/// operations that are hanging off of it, handle the differences between FIRRTL
/// and RTL approaches to module parameterization and output ports.
///
/// On success, this returns with the firrtl.instance op having no users,
/// letting the caller erase it.
void FIRRTLModuleLowering::lowerInstance(
    InstanceOp oldInstance,
    DenseMap<Operation *, Operation *> &oldToNewModuleMap) {

  auto *oldModule = oldInstance.getReferencedModule();
  auto newModule = oldToNewModuleMap[oldModule];
  if (!newModule)
    return;

  // If this is a referenced to a parameterized extmodule, then bring the
  // parameters over to this instance.
  DictionaryAttr parameters;
  if (auto oldExtModule = dyn_cast<FExtModuleOp>(oldModule))
    if (auto paramsOptional = oldExtModule.parameters())
      parameters = paramsOptional.getValue();

  // Decode information about the input and output ports on the referenced
  // module.
  SmallVector<ModulePortInfo, 8> portInfo;
  getModulePortInfo(oldModule, portInfo);

  // Build an index from the name attribute to an index into portInfo, so we can
  // do efficient lookups.
  llvm::SmallDenseMap<Attribute, unsigned> portIndicesByName;
  for (unsigned portIdx = 0, e = portInfo.size(); portIdx != e; ++portIdx)
    portIndicesByName[portInfo[portIdx].name] = portIdx;

  // Find all the subfield ops hanging off of this instance, indexed by
  // portRecord.  Typically there is exactly one subfield for every port, but
  // there can be more.
  SmallVector<TinyPtrVector<Operation *>, 8> subfieldsByPortIndex;
  subfieldsByPortIndex.resize(portInfo.size());
  for (auto *user : Value(oldInstance).getUsers()) {
    auto subfield = dyn_cast<SubfieldOp>(user);
    if (!subfield) {
      user->emitOpError("unexpected user of firrtl.instance operation");
      return;
    }

    // Find the port record for this port.
    assert(portIndicesByName.count(subfield.fieldnameAttr()) &&
           "invalid subfield for instance");
    unsigned portIndex = portIndicesByName[subfield.fieldnameAttr()];
    subfieldsByPortIndex[portIndex].push_back(subfield);
  }

  // Ok, get ready to create the new instance operation.  We need to prepare
  // input operands and results.
  ImplicitLocOpBuilder builder(oldInstance.getLoc(), oldInstance);
  SmallVector<Type, 8> resultTypes;
  SmallVector<Value, 8> operands;
  for (size_t portIndex = 0, e = portInfo.size(); portIndex != e; ++portIndex) {
    auto &port = portInfo[portIndex];
    auto portType = lowerType(port.type);
    if (!portType)
      return;

    if (port.isOutput()) {
      // outputs become results.
      resultTypes.push_back(portType);
      continue;
    }

    assert(port.isInput() &&
           "TODO: Handle inout ports when we can lower mid FIRRTL bundles");

    // If there is a single subfield projection for this input, and if we can
    // find the connects to it, then we can directly materialize it.
    auto &subfields = subfieldsByPortIndex[portIndex];
    if (subfields.size() == 1) {
      if (auto value = tryToConnectsToFlipValue(subfields[0]->getResult(0),
                                                oldInstance)) {
        // If we got a value connecting to the input port, then we can cast it
        // and pass it into the RTL instance without a temporary wire.
        operands.push_back(castFromFIRRTLType(value, portType, builder));
        // Remove the subfield itself.
        subfields.back()->erase();
        subfields.clear();
        continue;
      }
    }

    // Otherwise, create a wire for each input/inout operand, so there is
    // something to connect to.
    auto name = builder.getStringAttr(port.getName().str() + ".wire");
    auto wire = builder.create<WireOp>(port.type, name);
    operands.push_back(castFromFIRRTLType(wire, portType, builder));

    // Replace all the uses of the subfields with the wire we just created.
    for (auto *subfield : subfields) {
      subfield->getResult(0).replaceAllUsesWith(wire);
      subfield->erase();
    }
    subfields.clear();
  }

  // Use the symbol from the module we are referencing.
  FlatSymbolRefAttr symbolAttr = builder.getSymbolRefAttr(newModule);

  // Create the new rtl.instance operation.
  StringAttr instanceName;
  if (oldInstance.name().hasValue())
    instanceName = oldInstance.nameAttr();

  auto newInst = builder.create<rtl::InstanceOp>(
      resultTypes, instanceName, symbolAttr, operands, parameters);

  // Now that we have the new rtl.instance, we need to remap all of the users
  // of the outputs/results to the values returned by the instance.
  unsigned resultNo = 0;
  for (size_t portIndex = 0, e = portInfo.size(); portIndex != e; ++portIndex) {
    auto &port = portInfo[portIndex];
    if (!port.isOutput())
      continue;

    // Replace any subfield uses of this output port with the returned value
    // directly.
    auto &subfields = subfieldsByPortIndex[portIndex];
    for (auto *subfield : subfields) {
      auto resultVal = newInst.getResult(resultNo);
      // Cast the value to the right signedness and flippedness.
      resultVal =
          castToFIRRTLType(resultVal, FlipType::get(port.type), builder);
      subfield->getResult(0).replaceAllUsesWith(resultVal);
      subfield->erase();
    }
    subfields.clear();
    ++resultNo;
  }
}

//===----------------------------------------------------------------------===//
// Module Body Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLLowering : public LowerFIRRTLToRTLBase<FIRRTLLowering>,
                        public FIRRTLVisitor<FIRRTLLowering, LogicalResult> {

  void runOnOperation() override;

  // Helpers.
  Value getPossiblyInoutLoweredValue(Value value);
  Value getLoweredValue(Value value);
  Value getLoweredAndExtendedValue(Value value, Type destType);
  LogicalResult setLowering(Value orig, Value result);
  template <typename ResultOpType, typename... CtorArgTypes>
  LogicalResult setLoweringTo(Operation *orig, CtorArgTypes... args);

  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitExpr;
  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitDecl;
  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitStmt;

  // Lowering hooks.
  void handleUnloweredOp(Operation *op);
  LogicalResult visitExpr(ConstantOp op);
  LogicalResult visitDecl(WireOp op);
  LogicalResult visitDecl(NodeOp op);
  LogicalResult visitUnhandledOp(Operation *op) { return failure(); }
  LogicalResult visitInvalidOp(Operation *op) { return failure(); }

  // Unary Ops.
  LogicalResult lowerNoopCast(Operation *op);
  LogicalResult visitExpr(AsSIntPrimOp op) { return lowerNoopCast(op); }
  LogicalResult visitExpr(AsUIntPrimOp op) { return lowerNoopCast(op); }
  LogicalResult visitExpr(AsPassivePrimOp op) { return lowerNoopCast(op); }
  LogicalResult visitExpr(AsNonPassivePrimOp op) { return lowerNoopCast(op); }
  LogicalResult visitExpr(AsClockPrimOp op) { return lowerNoopCast(op); }
  LogicalResult visitExpr(AsAsyncResetPrimOp op) { return lowerNoopCast(op); }

  LogicalResult visitExpr(StdIntCastOp op);
  LogicalResult visitExpr(AnalogInOutCastOp op);
  LogicalResult visitExpr(CvtPrimOp op);
  LogicalResult visitExpr(NotPrimOp op);
  LogicalResult visitExpr(NegPrimOp op);
  LogicalResult visitExpr(PadPrimOp op);
  LogicalResult visitExpr(XorRPrimOp op);
  LogicalResult visitExpr(AndRPrimOp op);
  LogicalResult visitExpr(OrRPrimOp op);

  // Binary Ops.
  template <typename ResultUnsignedOpType,
            typename ResultSignedOpType = ResultUnsignedOpType>
  LogicalResult lowerBinOp(Operation *op);
  template <typename ResultOpType>
  LogicalResult lowerBinOpToVariadic(Operation *op);
  LogicalResult lowerCmpOp(Operation *op, ICmpPredicate signedOp,
                           ICmpPredicate unsignedOp);
  template <typename SignedOp, typename UnsignedOp>
  LogicalResult lowerDivLikeOp(Operation *op);

  LogicalResult visitExpr(CatPrimOp op);

  LogicalResult visitExpr(AndPrimOp op) {
    return lowerBinOpToVariadic<rtl::AndOp>(op);
  }
  LogicalResult visitExpr(OrPrimOp op) {
    return lowerBinOpToVariadic<rtl::OrOp>(op);
  }
  LogicalResult visitExpr(XorPrimOp op) {
    return lowerBinOpToVariadic<rtl::XorOp>(op);
  }
  LogicalResult visitExpr(AddPrimOp op) {
    return lowerBinOpToVariadic<rtl::AddOp>(op);
  }
  LogicalResult visitExpr(EQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::eq, ICmpPredicate::eq);
  }
  LogicalResult visitExpr(NEQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::ne, ICmpPredicate::ne);
  }
  LogicalResult visitExpr(LTPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::slt, ICmpPredicate::ult);
  }
  LogicalResult visitExpr(LEQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::sle, ICmpPredicate::ule);
  }
  LogicalResult visitExpr(GTPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::sgt, ICmpPredicate::ugt);
  }
  LogicalResult visitExpr(GEQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::sge, ICmpPredicate::uge);
  }

  LogicalResult visitExpr(SubPrimOp op) { return lowerBinOp<rtl::SubOp>(op); }
  LogicalResult visitExpr(MulPrimOp op) {
    return lowerBinOpToVariadic<rtl::MulOp>(op);
  }
  LogicalResult visitExpr(DivPrimOp op) {
    return lowerDivLikeOp<rtl::DivSOp, rtl::DivUOp>(op);
  }
  LogicalResult visitExpr(RemPrimOp op);

  // Other Operations
  LogicalResult visitExpr(BitsPrimOp op);
  LogicalResult visitExpr(HeadPrimOp op);
  LogicalResult visitExpr(ShlPrimOp op);
  LogicalResult visitExpr(ShrPrimOp op);
  LogicalResult visitExpr(DShlPrimOp op) {
    return lowerDivLikeOp<rtl::ShlOp, rtl::ShlOp>(op);
  }
  LogicalResult visitExpr(DShrPrimOp op) {
    return lowerDivLikeOp<rtl::ShrSOp, rtl::ShrUOp>(op);
  }
  LogicalResult visitExpr(TailPrimOp op);
  LogicalResult visitExpr(MuxPrimOp op);
  LogicalResult visitExpr(ValidIfPrimOp op);

  // Statements
  LogicalResult visitStmt(ConnectOp op);
  LogicalResult visitStmt(InvalidOp op);
  LogicalResult visitStmt(PrintFOp op);
  LogicalResult visitStmt(StopOp op);
  LogicalResult visitStmt(AssertOp op);
  LogicalResult visitStmt(AssumeOp op);
  LogicalResult visitStmt(CoverOp op);
  LogicalResult visitStmt(AttachOp op);

private:
  /// This builder is set to the right location for each visit call.
  ImplicitLocOpBuilder *builder = nullptr;

  /// Each value lowered (e.g. operation result) is kept track in this map.  The
  /// key should have a FIRRTL type, the result will have an RTL dialect type.
  DenseMap<Value, Value> valueMapping;

  template <typename AOpTy, typename BOpTy>
  LogicalResult lowerVerificationStatement(AOpTy op);
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLToRTLPass() {
  return std::make_unique<FIRRTLLowering>();
}

// This is the main entrypoint for the lowering pass.
void FIRRTLLowering::runOnOperation() {
  // FIRRTL FModule is a single block because FIRRTL ops are a DAG.  Walk
  // through each operation lowering each in turn if we can, introducing casts
  // if we cannot.
  auto *body = getOperation().getBodyBlock();

  SmallVector<Operation *, 16> opsToRemove;

  // Iterate through each operation in the module body, attempting to lower
  // each of them.  We maintain 'builder' for each invocation.
  ImplicitLocOpBuilder theBuilder(getOperation().getLoc(), &getContext());
  builder = &theBuilder;
  for (auto &op : body->getOperations()) {
    builder->setInsertionPoint(&op);
    builder->setLoc(op.getLoc());
    if (succeeded(dispatchVisitor(&op))) {
      opsToRemove.push_back(&op);
    } else {
      // If lowering didn't succeed, then make sure to rewrite operands that
      // refer to lowered values.
      handleUnloweredOp(&op);
    }
  }
  builder = nullptr;

  // Now that all of the operations that can be lowered are, remove the original
  // values.  We know that any lowered operations will be dead (if removed in
  // reverse order) at this point - any users of them from unremapped operations
  // will be changed to use the newly lowered ops.
  while (!opsToRemove.empty())
    opsToRemove.pop_back_val()->erase();

  // Clear out the value mapping for next time, so we don't have dangling keys.
  valueMapping.clear();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return the lowered RTL value corresponding to the specified original value.
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.  This returns rtl::inout type values if present, it
/// does not implicitly read from them.
Value FIRRTLLowering::getPossiblyInoutLoweredValue(Value value) {
  // All FIRRTL dialect values have FIRRTL types, so if we see something else
  // mixed in, it must be something we can't lower.  Just return it directly.
  auto firType = value.getType().dyn_cast<FIRRTLType>();
  if (!firType)
    return value;

  // If we lowered this value, then return the lowered value.
  auto it = valueMapping.find(value);
  if (it != valueMapping.end())
    return it->second;

  // Otherwise, we need to introduce (or look through) a cast to the right
  // FIRRTL type.
  auto resultType = lowerType(firType);
  if (!resultType)
    return value;

  if (!resultType.isa<IntegerType>())
    return {};

  // Cast FIRRTL -> standard type.
  value = builder->createOrFold<AsPassivePrimOp>(value);
  return builder->createOrFold<StdIntCastOp>(resultType, value);
}

/// Return the lowered value corresponding to the specified original value.
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredValue(Value value) {
  auto result = getPossiblyInoutLoweredValue(value);
  if (!result)
    return result;

  // If we got an inout value, implicitly read it.  FIRRTL allows direct use
  // of wires and other things that lower to inout type.
  if (result.getType().isa<rtl::InOutType>())
    return builder->createOrFold<rtl::ReadInOutOp>(result);

  return result;
}

/// Return the lowered value corresponding to the specified original value and
/// then extend it to match the width of destType if needed.
///
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredAndExtendedValue(Value value, Type destType) {
  assert(value.getType().isa<FIRRTLType>() && destType.isa<FIRRTLType>() &&
         "input/output value should be FIRRTL");
  auto destFIRType = destType.cast<FIRRTLType>();

  auto result = getLoweredValue(value);
  if (!result)
    return {};

  // We only know how to extend integer types with known width.
  auto destIntType = destFIRType.dyn_cast<IntType>();
  if (!destIntType || !destIntType.hasWidth())
    return {};

  auto destWidth = unsigned(destIntType.getWidthOrSentinel());
  auto srcWidth = result.getType().cast<IntegerType>().getWidth();
  if (srcWidth == destWidth)
    return result;

  if (srcWidth > destWidth) {
    builder->emitError("operand should not be a truncation");
    return {};
  }

  auto resultType = builder->getIntegerType(destWidth);

  // Extension follows the sign of the source value, not the destination.
  auto valueFIRType = value.getType().cast<FIRRTLType>().getPassiveType();
  if (valueFIRType.cast<IntType>().isSigned())
    return builder->createOrFold<rtl::SExtOp>(resultType, result);

  return builder->createOrFold<rtl::ZExtOp>(resultType, result);
}

/// Set the lowered value of 'orig' to 'result', remembering this in a map.
/// This always returns success() to make it more convenient in lowering code.
LogicalResult FIRRTLLowering::setLowering(Value orig, Value result) {
  assert(orig.getType().isa<FIRRTLType>() &&
         !result.getType().isa<FIRRTLType>() &&
         "Lowering didn't turn a FIRRTL value into a non-FIRRTL value");

  assert(!valueMapping.count(orig) && "value lowered multiple times");
  valueMapping[orig] = result;
  return success();
}

/// Create a new operation with type ResultOpType and arguments CtorArgTypes,
/// then call setLowering with its result.
template <typename ResultOpType, typename... CtorArgTypes>
LogicalResult FIRRTLLowering::setLoweringTo(Operation *orig,
                                            CtorArgTypes... args) {
  auto result = builder->createOrFold<ResultOpType>(args...);
  return setLowering(orig->getResult(0), result);
}

//===----------------------------------------------------------------------===//
// Special Operations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitDecl(WireOp op) {
  auto resType = op.result().getType().cast<FIRRTLType>();
  auto resultType = lowerType(resType);
  if (!resultType)
    return failure();

  // Convert the inout to a non-inout type.
  return setLoweringTo<rtl::WireOp>(op, resultType, op.nameAttr());
}

LogicalResult FIRRTLLowering::visitDecl(NodeOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();

  // Node operations are logical noops, but can carry a name.  If a name is
  // present then we lower this into a wire and a connect, otherwise we just
  // drop it.
  if (auto name = op.getAttrOfType<StringAttr>("name")) {
    auto wire = builder->create<rtl::WireOp>(operand.getType(), name);
    builder->create<rtl::ConnectOp>(wire, operand);
  }

  // TODO(clattner): This is dropping the location information from unnamed node
  // ops.  I suspect that this falls below the fold in terms of things we care
  // about given how Chisel works, but we should reevaluate with more
  // information.
  return setLowering(op, operand);
}

/// Handle the case where an operation wasn't lowered.  When this happens, the
/// operands may be a mix of lowered and unlowered values.  If the operand was
/// not lowered then leave it alone, otherwise insert a cast from the lowered
/// value.
void FIRRTLLowering::handleUnloweredOp(Operation *op) {
  for (auto &operand : op->getOpOperands()) {
    Value origValue = operand.get();
    auto it = valueMapping.find(origValue);
    // If the operand wasn't lowered, then leave it alone.
    if (it == valueMapping.end())
      continue;

    // Otherwise, insert a cast from the lowered value.
    Value mapped = castToFIRRTLType(it->second, origValue.getType(), *builder);
    operand.set(mapped);
  }
}

LogicalResult FIRRTLLowering::visitExpr(ConstantOp op) {
  return setLoweringTo<rtl::ConstantOp>(op, op.value());
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Lower a cast that is a noop at the RTL level.
LogicalResult FIRRTLLowering::lowerNoopCast(Operation *op) {
  auto operand = getLoweredValue(op->getOperand(0));
  if (!operand)
    return failure();

  // Noop cast.
  return setLowering(op->getResult(0), operand);
}

LogicalResult FIRRTLLowering::visitExpr(StdIntCastOp op) {
  auto result = getLoweredValue(op.getOperand());
  if (!result)
    return failure();

  // Conversions from standard integer types to FIRRTL types are lowered as the
  // input operand.
  if (!op.getType().isa<IntegerType>())
    return setLowering(op, result);

  // We lower firrtl.stdIntCast converting from a firrtl type to a standard type
  // into the lowered operand.
  op.replaceAllUsesWith(result);
  return success();
}

LogicalResult FIRRTLLowering::visitExpr(AnalogInOutCastOp op) {
  auto result = getPossiblyInoutLoweredValue(op.getOperand());
  if (!result)
    return failure();

  return setLowering(op->getResult(0), result);
}

LogicalResult FIRRTLLowering::visitExpr(CvtPrimOp op) {
  auto operand = getLoweredValue(op.getOperand());
  if (!operand)
    return failure();

  // Signed to signed is a noop.
  if (getTypeOf<IntType>(op.getOperand()).isSigned())
    return setLowering(op, operand);

  // Otherwise prepend a zero bit.
  auto zero = builder->create<rtl::ConstantOp>(APInt(1, 0));
  return setLoweringTo<rtl::ConcatOp>(op, ValueRange({zero, operand}));
}

LogicalResult FIRRTLLowering::visitExpr(NotPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();
  // ~x  ---> x ^ 0xFF
  auto type = operand.getType().cast<IntegerType>();
  auto allOnes = builder->create<rtl::ConstantOp>(-1, type);
  return setLoweringTo<rtl::XorOp>(op, ValueRange({operand, allOnes}),
                                   ArrayRef<NamedAttribute>{});
}

LogicalResult FIRRTLLowering::visitExpr(NegPrimOp op) {
  // FIRRTL negate always adds a bit, and always does a sign extension even if
  // the input is unsigned.
  // -x  ---> 0-sext(x)
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();
  auto resultType = lowerType(op.getType());
  operand = builder->createOrFold<rtl::SExtOp>(resultType, operand);

  auto zero =
      builder->create<rtl::ConstantOp>(0, resultType.cast<IntegerType>());
  return setLoweringTo<rtl::SubOp>(op, zero, operand);
}

// Pad is a noop or extension operation.
LogicalResult FIRRTLLowering::visitExpr(PadPrimOp op) {
  auto operand = getLoweredAndExtendedValue(op.input(), op.getType());
  if (!operand)
    return failure();
  return setLowering(op, operand);
}

LogicalResult FIRRTLLowering::visitExpr(XorRPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();

  return setLoweringTo<rtl::XorROp>(op, builder->getIntegerType(1), operand);
}

LogicalResult FIRRTLLowering::visitExpr(AndRPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();

  return setLoweringTo<rtl::AndROp>(op, builder->getIntegerType(1), operand);
}

LogicalResult FIRRTLLowering::visitExpr(OrRPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();

  return setLoweringTo<rtl::OrROp>(op, builder->getIntegerType(1), operand);
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

template <typename ResultOpType>
LogicalResult FIRRTLLowering::lowerBinOpToVariadic(Operation *op) {
  auto resultType = op->getResult(0).getType();
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);
  if (!lhs || !rhs)
    return failure();

  return setLoweringTo<ResultOpType>(op, ValueRange({lhs, rhs}),
                                     ArrayRef<NamedAttribute>{});
}

/// lowerBinOp extends each operand to the destination type, then performs the
/// specified binary operator.
template <typename ResultUnsignedOpType, typename ResultSignedOpType>
LogicalResult FIRRTLLowering::lowerBinOp(Operation *op) {
  // Extend the two operands to match the destination type.
  auto resultType = op->getResult(0).getType();
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);
  if (!lhs || !rhs)
    return failure();

  // Emit the result operation.
  if (resultType.cast<IntType>().isSigned())
    return setLoweringTo<ResultSignedOpType>(op, lhs, rhs);
  return setLoweringTo<ResultUnsignedOpType>(op, lhs, rhs);
}

/// lowerCmpOp extends each operand to the longest type, then performs the
/// specified binary operator.
LogicalResult FIRRTLLowering::lowerCmpOp(Operation *op, ICmpPredicate signedOp,
                                         ICmpPredicate unsignedOp) {
  // Extend the two operands to match the longest type.
  auto lhsIntType = op->getOperand(0).getType().cast<IntType>();
  auto rhsIntType = op->getOperand(1).getType().cast<IntType>();
  if (!lhsIntType.hasWidth() || !rhsIntType.hasWidth())
    return failure();

  Type cmpType =
      *lhsIntType.getWidth() < *rhsIntType.getWidth() ? rhsIntType : lhsIntType;

  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), cmpType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), cmpType);
  if (!lhs || !rhs)
    return failure();

  // Emit the result operation.
  Type resultType = builder->getIntegerType(1);
  return setLoweringTo<rtl::ICmpOp>(
      op, resultType, lhsIntType.isSigned() ? signedOp : unsignedOp, lhs, rhs);
}

/// Lower a divide or dynamic shift, where the operation has to be performed in
/// the widest type of the result and two inputs then truncated down.
template <typename SignedOp, typename UnsignedOp>
LogicalResult FIRRTLLowering::lowerDivLikeOp(Operation *op) {
  // rtl has equal types for these, firrtl doesn't.  The type of the firrtl RHS
  // may be wider than the LHS, and we cannot truncate off the high bits
  // (because an overlarge amount is supposed to shift in sign or zero bits).
  auto opType = op->getResult(0).getType().cast<IntType>();
  auto resultType = getWidestIntType(opType, op->getOperand(1).getType());
  resultType = getWidestIntType(resultType, op->getOperand(0).getType());
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);
  if (!lhs || !rhs)
    return failure();

  Value result;
  if (opType.isSigned())
    result = builder->createOrFold<SignedOp>(lhs, rhs);
  else
    result = builder->createOrFold<UnsignedOp>(lhs, rhs);

  if (resultType == opType)
    return setLowering(op->getResult(0), result);
  return setLoweringTo<rtl::ExtractOp>(op, lowerType(opType), result, 0);
}

LogicalResult FIRRTLLowering::visitExpr(CatPrimOp op) {
  auto lhs = getLoweredValue(op.lhs());
  auto rhs = getLoweredValue(op.rhs());
  if (!lhs || !rhs)
    return failure();

  return setLoweringTo<rtl::ConcatOp>(op, ValueRange({lhs, rhs}));
}

LogicalResult FIRRTLLowering::visitExpr(RemPrimOp op) {
  // FIRRTL has the width of (a % b) = Min(W(a), W(b)), but the operation is
  // done at max(W(a), W(b))) so we need to extend one operand, then truncate
  // the result.
  auto lhsFirTy = op.lhs().getType().dyn_cast<IntType>();
  auto rhsFirTy = op.rhs().getType().dyn_cast<IntType>();
  if (!lhsFirTy || !rhsFirTy || !lhsFirTy.hasWidth() || !rhsFirTy.hasWidth())
    return failure();
  auto opType = lhsFirTy.getWidth() > rhsFirTy.getWidth() ? lhsFirTy : rhsFirTy;

  auto lhs = getLoweredAndExtendedValue(op.lhs(), opType);
  auto rhs = getLoweredAndExtendedValue(op.rhs(), opType);
  if (!lhs || !rhs)
    return failure();

  auto resultFirType = op.getType().cast<IntType>();
  if (!resultFirType.hasWidth())
    return failure();
  auto destWidth = unsigned(resultFirType.getWidthOrSentinel());
  auto resultType = builder->getIntegerType(destWidth);

  Value modInst;
  if (resultFirType.isUnsigned()) {
    modInst = builder->createOrFold<rtl::ModUOp>(ValueRange({lhs, rhs}));
  } else {
    modInst = builder->createOrFold<rtl::ModSOp>(ValueRange({lhs, rhs}));
  }

  return setLoweringTo<rtl::ExtractOp>(op, resultType, modInst, 0);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitExpr(BitsPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  Type resultType = builder->getIntegerType(op.hi() - op.lo() + 1);
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input, op.lo());
}

LogicalResult FIRRTLLowering::visitExpr(HeadPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  Type resultType = builder->getIntegerType(op.amount());
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input,
                                       inWidth - op.amount());
}

LogicalResult FIRRTLLowering::visitExpr(ShlPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  // Handle the degenerate case.
  if (op.amount() == 0)
    return setLowering(op, input);

  // TODO: We could keep track of zeros and implicitly CSE them.
  auto zero = builder->create<rtl::ConstantOp>(APInt(op.amount(), 0));
  return setLoweringTo<rtl::ConcatOp>(op, ValueRange({input, zero}));
}

LogicalResult FIRRTLLowering::visitExpr(ShrPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  // Handle the special degenerate cases.
  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  auto shiftAmount = op.amount();
  if (shiftAmount == inWidth) {
    // Unsigned shift by full width returns a single-bit zero.
    if (op.input().getType().cast<IntType>().isUnsigned())
      return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 0));

    // Signed shift by full width is equivalent to extracting the sign bit.
    --shiftAmount;
  }

  Type resultType = builder->getIntegerType(inWidth - shiftAmount);
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input, shiftAmount);
}

LogicalResult FIRRTLLowering::visitExpr(TailPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  Type resultType = builder->getIntegerType(inWidth - op.amount());
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input, 0);
}

LogicalResult FIRRTLLowering::visitExpr(MuxPrimOp op) {
  auto cond = getLoweredValue(op.sel());
  auto ifTrue = getLoweredAndExtendedValue(op.high(), op.getType());
  auto ifFalse = getLoweredAndExtendedValue(op.low(), op.getType());
  if (!cond || !ifTrue || !ifFalse)
    return failure();

  return setLoweringTo<rtl::MuxOp>(op, ifTrue.getType(), cond, ifTrue, ifFalse);
}

LogicalResult FIRRTLLowering::visitExpr(ValidIfPrimOp op) {
  // It isn't clear to me why it it is ok to ignore the binding condition,
  // but this is what the existing FIRRTL verilog emitter does.
  auto val = getLoweredValue(op.rhs());
  if (!val)
    return failure();

  return setLowering(op, val);
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitStmt(InvalidOp op) {
  auto dest = getPossiblyInoutLoweredValue(op.operand());
  if (!dest)
    return failure();

  auto inoutTy = dest.getType().dyn_cast<rtl::InOutType>();
  if (!inoutTy)
    return op.emitError("destination isn't an inout type");

  auto zero = builder->create<rtl::ConstantOp>(
      0, inoutTy.getElementType().cast<IntegerType>());

  builder->create<rtl::ConnectOp>(dest, zero);
  return success();
}

LogicalResult FIRRTLLowering::visitStmt(ConnectOp op) {
  auto dest = getPossiblyInoutLoweredValue(op.dest());

  // The source can be a smaller integer, extend it as appropriate if so.
  auto destType = op.dest().getType().cast<FIRRTLType>().getPassiveType();
  Value src = getLoweredAndExtendedValue(op.src(), destType);

  if (!dest || !src)
    return failure();

  if (!dest.getType().isa<rtl::InOutType>())
    return op.emitError("destination isn't an inout type");

  builder->create<rtl::ConnectOp>(dest, src);
  return success();
}

// Printf is a macro op that lowers to an sv.ifdef, an sv.if, and an sv.fwrite
// all nested together.
LogicalResult FIRRTLLowering::visitStmt(PrintFOp op) {
  auto clock = getLoweredValue(op.clock());
  auto cond = getLoweredValue(op.cond());
  if (!clock || !cond)
    return failure();

  SmallVector<Value, 4> operands;
  operands.reserve(op.operands().size());
  for (auto operand : op.operands()) {
    operands.push_back(getLoweredValue(operand));
    if (!operands.back())
      return failure();
  }

  // Emit this into an "sv.alwaysat_posedge" body.
  builder->create<sv::AlwaysAtPosEdgeOp>(clock, [&]() {
    // Emit an "#ifndef SYNTHESIS" guard into the always block.
    builder->create<sv::IfDefOp>("!SYNTHESIS", [&]() {
      // Emit an "sv.if '`PRINTF_COND_ & cond' into the #ifndef.
      Value ifCond =
          builder->create<sv::TextualValueOp>(cond.getType(), "`PRINTF_COND_");
      ifCond = builder->createOrFold<rtl::AndOp>(ValueRange{ifCond, cond},
                                                 ArrayRef<NamedAttribute>{});
      builder->create<sv::IfOp>(ifCond, [&]() {
        // Emit the sv.fwrite.
        builder->create<sv::FWriteOp>(op.formatString(), operands);
      });
    });
  });

  return success();
}

// Stop lowers into a nested series of behavioral statements plus $fatal or
// $finish.
LogicalResult FIRRTLLowering::visitStmt(StopOp op) {
  // Emit this into an "sv.alwaysat_posedge" body.
  auto clock = getLoweredValue(op.clock());
  auto cond = getLoweredValue(op.cond());
  if (!clock || !cond)
    return failure();

  builder->create<sv::AlwaysAtPosEdgeOp>(clock, [&]() {
    // Emit an "#ifndef SYNTHESIS" guard into the always block.
    builder->create<sv::IfDefOp>("!SYNTHESIS", [&]() {
      // Emit an "sv.if '`STOP_COND_ & cond' into the #ifndef.
      Value ifCond =
          builder->create<sv::TextualValueOp>(cond.getType(), "`STOP_COND_");
      ifCond = builder->createOrFold<rtl::AndOp>(ValueRange{ifCond, cond},
                                                 ArrayRef<NamedAttribute>{});
      builder->create<sv::IfOp>(ifCond, [&]() {
        // Emit the sv.fatal or sv.finish.
        if (op.exitCode())
          builder->create<sv::FatalOp>();
        else
          builder->create<sv::FinishOp>();
      });
    });
  });

  return success();
}

/// Template for lowering verification statements from type A to
/// type B.
///
/// For example, lowering the "foo" op to the "bar" op would start
/// with:
///
///     foo(clock, condition, enable, "message")
///
/// This becomes a Verilog clocking block with the "bar" op guarded
/// by an if enable:
///
///     always @(posedge clock) begin
///       if (enable) begin
///         bar(condition);
///       end
///     end
template <typename AOpTy, typename BOpTy>
LogicalResult FIRRTLLowering::lowerVerificationStatement(AOpTy op) {
  auto clock = getLoweredValue(op.clock());
  auto enable = getLoweredValue(op.enable());
  auto predicate = getLoweredValue(op.predicate());
  if (!clock || !enable || !predicate)
    return failure();

  builder->create<sv::AlwaysAtPosEdgeOp>(clock, [&]() {
    builder->create<sv::IfOp>(enable, [&]() {
      // Create BOpTy inside the always/if.
      builder->createOrFold<BOpTy>(predicate);
    });
  });

  return success();
}

// Lower an assert to SystemVerilog.
LogicalResult FIRRTLLowering::visitStmt(AssertOp op) {
  return lowerVerificationStatement<AssertOp, sv::AssertOp>(op);
}

// Lower an assume to SystemVerilog.
LogicalResult FIRRTLLowering::visitStmt(AssumeOp op) {
  return lowerVerificationStatement<AssumeOp, sv::AssumeOp>(op);
}

// Lower a cover to SystemVerilog.
LogicalResult FIRRTLLowering::visitStmt(CoverOp op) {
  return lowerVerificationStatement<CoverOp, sv::CoverOp>(op);
}

LogicalResult FIRRTLLowering::visitStmt(AttachOp op) {
  // Don't emit anything for a zero or one operand attach.
  if (op.operands().size() < 2)
    return success();

  SmallVector<Value, 4> inoutValues;
  for (auto v : op.operands()) {
    inoutValues.push_back(getPossiblyInoutLoweredValue(v));
    if (!inoutValues.back())
      return failure();

    if (!inoutValues.back().getType().isa<rtl::InOutType>())
      return op.emitError("operand isn't an inout type");
  }

  // In the non-synthesis case, we emit a SystemVerilog alias statement.
  builder->create<sv::IfDefOp>(
      "!SYNTHESIS", [&]() { builder->create<sv::AliasOp>(inoutValues); });

  // If we're doing synthesis, we emit an all-pairs assign complex.
  builder->create<sv::IfDefOp>("SYNTHESIS", [&]() {
    // Lower the
    SmallVector<Value, 4> values;
    for (size_t i = 0, e = inoutValues.size(); i != e; ++i)
      values.push_back(builder->createOrFold<rtl::ReadInOutOp>(inoutValues[i]));

    for (size_t i1 = 0, e = inoutValues.size(); i1 != e; ++i1) {
      for (size_t i2 = 0; i2 != e; ++i2)
        if (i1 != i2)
          builder->create<rtl::ConnectOp>(inoutValues[i1], values[i2]);
    }
  });

  return success();
}
