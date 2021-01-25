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
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TinyPtrVector.h"
using namespace circt;
using namespace firrtl;

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

/// Return true if the specified FIRRTL type is a sized type (Int or Analog)
/// with zero bits.
static bool isZeroBitFIRRTLType(Type type) {
  return type.cast<FIRRTLType>().getPassiveType().getBitWidthOrSentinel() == 0;
}

//===----------------------------------------------------------------------===//
// firrtl.module Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLModuleLowering
    : public LowerFIRRTLToRTLModuleBase<FIRRTLModuleLowering> {

  void runOnOperation() override;

private:
  void lowerFileHeader(CircuitOp op);
  LogicalResult lowerPorts(ArrayRef<ModulePortInfo> firrtlPorts,
                           SmallVectorImpl<rtl::ModulePortInfo> &ports,
                           Operation *moduleOp);
  rtl::RTLModuleOp lowerModule(FModuleOp oldModule, Block *topLevelModule);
  rtl::RTLExternModuleOp lowerExtModule(FExtModuleOp oldModule,
                                        Block *topLevelModule);

  void lowerModuleBody(FModuleOp oldModule,
                       DenseMap<Operation *, Operation *> &oldToNewModuleMap);

  void lowerInstance(InstanceOp instance, CircuitOp circuitOp,
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

  // Emit all the macros and preprocessor gunk at the start of the file.
  lowerFileHeader(circuit);

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
  getOperation()->setAttr(
      "firrtl.mainModule",
      StringAttr::get(circuit.name(), circuit.getContext()));
  circuit.erase();
}

/// Emit the file header that defines a bunch of macros.
void FIRRTLModuleLowering::lowerFileHeader(CircuitOp op) {
  // Intentionally pass an UnknownLoc here so we don't get line number comments
  // on the output of this boilerplate in generated Verilog.
  ImplicitLocOpBuilder b(UnknownLoc::get(&getContext()), op);

  // TODO: We could have an operation for macros and uses of them, and
  // even turn them into symbols so we can DCE unused macro definitions.
  auto emitString = [&](StringRef verilogString) {
    b.create<sv::VerbatimOp>(verilogString);
  };

  // Helper function to emit a "#ifdef guard" with a `define in the then and
  // optionally in the else branch.
  auto emitGuardedDefine = [&](const char *guard, const char *defineTrue,
                               const char *defineFalse = nullptr) {
    std::string define = "`define ";
    if (!defineFalse) {
      b.create<sv::IfDefOp>(guard, [&]() { emitString(define + defineTrue); });
    } else {
      b.create<sv::IfDefOp>(
          guard, [&]() { emitString(define + defineTrue); },
          [&]() { emitString(define + defineFalse); });
    }
  };

  emitString("// Standard header to adapt well known macros to our needs.");
  emitGuardedDefine("RANDOMIZE_GARBAGE_ASSIGN", "RANDOMIZE");
  emitGuardedDefine("RANDOMIZE_INVALID_ASSIGN", "RANDOMIZE");
  emitGuardedDefine("RANDOMIZE_REG_INIT", "RANDOMIZE");
  emitGuardedDefine("RANDOMIZE_MEM_INIT", "RANDOMIZE");
  emitGuardedDefine("!RANDOM", "RANDOM $random");

  emitString(
      "\n// Users can define 'PRINTF_COND' to add an extra gate to prints.");

  emitGuardedDefine("PRINTF_COND", "PRINTF_COND_ (`PRINTF_COND)",
                    "PRINTF_COND_ 1");

  emitString("\n// Users can define 'STOP_COND' to add an extra gate "
             "to stop conditions.");
  emitGuardedDefine("STOP_COND", "STOP_COND_ (`STOP_COND)", "STOP_COND_ 1");

  emitString(
      "\n// Users can define INIT_RANDOM as general code that gets injected "
      "into the\n// initializer block for modules with registers.");
  emitGuardedDefine("!INIT_RANDOM", "INIT_RANDOM");

  emitString("\n// If using random initialization, you can also define "
             "RANDOMIZE_DELAY to\n// customize the delay used, otherwise 0.002 "
             "is used.");
  emitGuardedDefine("!RANDOMIZE_DELAY", "RANDOMIZE_DELAY 0.002");

  emitString("\n// Define INIT_RANDOM_PROLOG_ for use in our modules below.");

  b.create<sv::IfDefOp>(
      "RANDOMIZE",
      [&]() {
        emitGuardedDefine(
            "!VERILATOR",
            "INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end",
            "INIT_RANDOM_PROLOG_ `INIT_RANDOM");
      },
      [&]() { emitString("`define INIT_RANDOM_PROLOG_"); });

  // Blank line to separate the header from the modules.
  emitString("");
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

    // If this is a zero bit port, just drop it.  It doesn't matter if it is
    // input, output, or inout.  We don't want these at the RTL level.
    if (rtlPort.type.isInteger(0))
      continue;

    // Figure out the direction of the port.
    if (firrtlPort.isOutput()) {
      rtlPort.direction = rtl::PortDirection::OUTPUT;
      rtlPort.argNum = numResults++;
    } else if (firrtlPort.isInput()) {
      rtlPort.direction = rtl::PortDirection::INPUT;
      rtlPort.argNum = numArgs++;
    } else {
      // If the port is an inout bundle or contains an analog type, then it is
      // implicitly inout.
      rtlPort.type = rtl::InOutType::get(rtlPort.type);
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

/// If the value dominates the marker, just return.  Otherwise add it to ops and
/// recursively process its operands.
///
/// Return true if we can't move an operation, false otherwise.
static bool
collectOperationTreeBelowMarker(Value value, Operation *marker,
                                SmallVector<Operation *, 8> &ops,
                                SmallPtrSet<Operation *, 8> &visited) {
  // If the value is a BB argument or if the op is in a containing block, then
  // it must dominate the marker.
  auto *op = value.getDefiningOp();
  if (!op || op->getBlock() != marker->getBlock())
    return false;

  // We can't move the marker itself.
  if (op == marker)
    return true;

  // Otherwise if it is an op in the same block as the marker, see if it is
  // already at or above the marker.
  if (op->isBeforeInBlock(marker))
    return false;

  // Pull the computation tree into the set.  If it was already added, then
  // don't reprocess it.
  if (!visited.insert(op).second)
    return false;

  // Otherwise recursively process the operands.
  for (auto operand : op->getOperands())
    if (collectOperationTreeBelowMarker(operand, marker, ops, visited))
      return true;

  // Add ops in post order so we make sure they get moved in a coherent order.
  ops.push_back(op);
  return false;
}

/// Given a value of flip type, check to see if all of the uses of it are
/// connects.  If so, remove the connects and return the value being connected
/// to it, converted to an RTL type.  If this isn't a situation we can handle,
/// just return null.
///
/// This can happen when there are no connects to the value, or if
/// firrtl.invalid is used.  The 'mergePoint' location is where a 'rtl.merge'
/// operation should be inserted if needed.
static Value tryEliminatingConnectsToValue(Value flipValue,
                                           Operation *insertPoint) {
  SmallVector<ConnectOp, 2> connects;
  for (auto *use : flipValue.getUsers()) {
    // We only know about 'connect' uses, where this is the destination.
    auto connect = dyn_cast<ConnectOp>(use);
    if (!connect || connect.src() == flipValue)
      return {};

    connects.push_back(connect);
  }

  // We don't have an RTL equivalent of "poison" so just don't special case the
  // case where there are no connects other uses of an output.
  if (connects.empty())
    return {};

  // Don't special case zero-bit results.
  auto loweredType = lowerType(flipValue.getType());
  if (loweredType.isInteger(0))
    return {};

  // We need to see if we can move all of the computation that feeds the
  // connects to be "above" the insertion point to avoid introducing cycles that
  // will break LowerToRTL.  Consider optimizing away a wire for inputs on an
  // instance like this:
  //
  //    %input1, %input2, %output = firrtl.instance (...)
  //    %value1 = computation1()
  //    firrtl.connect %input1, %value1
  //
  //    %value2 = computation2(%output)
  //    firrtl.connect %input2, %value2
  //
  // We can elide the wire for %input1, but have to move the computation1 ops
  // above the firrtl.instance.   However, there are cases like the second one
  // where we *cannot* move the computation.  In these sorts of cases, we just
  // fall back to inserting a wire conservatively, which breaks the cycle.
  //
  // We don't have to do this check for insertion points that are at the
  // terminator in the module, because we know that everything is above it by
  // definition.
  if (!insertPoint->isKnownTerminator()) {
    // On success, these are the ops that we need to move up above the insertion
    // point.  We keep track of a visited set because each compute subgraph is
    // a dag (not a tree), and we want to only want to visit each subnode once.
    SmallVector<Operation *, 8> opsToMove;
    SmallPtrSet<Operation *, 8> visited;

    // Collect the computation tree feeding the source operations.  We build the
    // list of ops to move in post-order to ensure that we provide a valid DAG
    // ordering of the result.
    for (auto connect : connects) {
      if (collectOperationTreeBelowMarker(connect.src(), insertPoint, opsToMove,
                                          visited))
        return {};
    }

    // Since it looks like all the operations can be moved, actually do it.
    for (auto *op : opsToMove)
      op->moveBefore(insertPoint);
  }

  // Convert each connect into an extended version of its operand being output.
  SmallVector<Value, 2> results;
  ImplicitLocOpBuilder builder(insertPoint->getLoc(), insertPoint);

  for (auto connect : connects) {
    auto connectSrc = connect.src();

    // Convert fliped sources to passive sources.
    if (!connectSrc.getType().cast<FIRRTLType>().isPassive())
      connectSrc = builder.createOrFold<AsPassivePrimOp>(connectSrc);

    // We know it must be the destination operand due to the types, but the
    // source may not match the destination width.
    auto destTy = flipValue.getType().cast<FIRRTLType>().getPassiveType();
    if (destTy != connectSrc.getType()) {
      // The only type mismatch we can have is due to integer width differences.
      auto destWidth = destTy.getBitWidthOrSentinel();
      assert(destWidth != -1 && "must know integer widths");
      connectSrc =
          builder.createOrFold<PadPrimOp>(destTy, connectSrc, destWidth);
    }

    // Remove the connect and use its source as the value for the output.
    connect.erase();
    results.push_back(connectSrc);
  }

  // Convert from FIRRTL type to builtin type to do the merge.
  for (auto &result : results)
    result = castFromFIRRTLType(result, loweredType, builder);

  // Folding merge of one value just returns the value.
  return builder.createOrFold<rtl::MergeOp>(results);
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
  assert(oldModule.body().getNumArguments() == ports.size() &&
         "port count mismatch");

  size_t nextNewArg = 0;
  size_t firrtlArg = 0;
  SmallVector<Value, 4> outputs;

  // This is the terminator in the new module.
  auto outputOp = newModule.getBodyBlock()->getTerminator();
  ImplicitLocOpBuilder outputBuilder(oldModule.getLoc(), outputOp);

  for (auto &port : ports) {
    // Inputs and outputs are both modeled as arguments in the FIRRTL level.
    auto oldArg = oldModule.body().getArgument(firrtlArg++);

    bool isZeroWidth =
        port.type.cast<FIRRTLType>().getBitWidthOrSentinel() == 0;

    if (!port.isOutput() && !isZeroWidth) {
      // Inputs and InOuts are modeled as arguments in the result, so we can
      // just map them over.  We model zero bit outputs as inouts.
      Value newArg = newModule.body().getArgument(nextNewArg++);

      // Cast the argument to the old type, reintroducing sign information in
      // the rtl.module body.
      newArg = castToFIRRTLType(newArg, oldArg.getType(), bodyBuilder);
      // Switch all uses of the old operands to the new ones.
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }

    // We lower zero width inout and outputs to a wire that isn't connected to
    // anything outside the module.  Inputs are lowered to zero.
    if (isZeroWidth && port.isInput()) {
      Value newArg = bodyBuilder.create<WireOp>(FlipType::get(port.type),
                                                "." + port.getName().str() +
                                                    ".0width_input");

      newArg = bodyBuilder.create<AsPassivePrimOp>(newArg);
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }

    if (auto value = tryEliminatingConnectsToValue(oldArg, outputOp)) {
      // If we were able to find the value being connected to the output,
      // directly use it!
      outputs.push_back(value);
      assert(oldArg.use_empty() && "should have removed all uses of oldArg");
      continue;
    }

    // Outputs need a temporary wire so they can be connect'd to, which we
    // then return.
    Value newArg = bodyBuilder.create<WireOp>(
        port.type, "." + port.getName().str() + ".output");
    // Switch all uses of the old operands to the new ones.
    oldArg.replaceAllUsesWith(newArg);

    // Don't output zero bit results or inouts.
    auto resultRTLType = lowerType(port.type);
    if (!resultRTLType.isInteger(0)) {
      auto output = castFromFIRRTLType(newArg, resultRTLType, outputBuilder);
      outputs.push_back(output);
    }
  }

  // Update the rtl.output terminator with the list of outputs we have.
  outputOp->setOperands(outputs);

  // Finally splice the body over, don't move the old terminator over though.
  auto &oldBlockInstList = oldModule.getBodyBlock()->getOperations();
  auto &newBlockInstList = newModule.getBodyBlock()->getOperations();
  newBlockInstList.splice(Block::iterator(cursor), oldBlockInstList,
                          oldBlockInstList.begin(),
                          std::prev(oldBlockInstList.end()));

  // Now that we're all over into the new module, update all the
  // firrtl.instance's to be rtl.instance's.  Lowering an instance will also
  // delete a bunch of firrtl.subfield and firrtl.connect operations, so we have
  // to be careful about iterator invalidation.
  for (auto opIt = newBlockInstList.begin(), opEnd = newBlockInstList.end();
       opIt != opEnd;) {
    auto instance = dyn_cast<InstanceOp>(&*opIt);
    if (!instance) {
      ++opIt;
      continue;
    }

    // Remember a position above the current op.  New things will get put before
    // the current op (including other instances!) and we want to make sure to
    // revisit them.
    cursor->moveBefore(instance);

    // We found an instance - lower it.  On successful return there will be
    // zero uses and we can remove the operation.
    lowerInstance(instance, oldModule->getParentOfType<CircuitOp>(),
                  oldToNewModuleMap);
    opIt = Block::iterator(cursor);
  }

  // We are done with our cursor op.
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
    InstanceOp oldInstance, CircuitOp circuitOp,
    DenseMap<Operation *, Operation *> &oldToNewModuleMap) {

  auto *oldModule = circuitOp.lookupSymbol(oldInstance.moduleName());
  auto newModule = oldToNewModuleMap[oldModule];
  if (!newModule) {
    oldInstance->emitOpError("could not find module referenced by instance");
    return;
  }

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

  // Ok, get ready to create the new instance operation.  We need to prepare
  // input operands and results.
  ImplicitLocOpBuilder builder(oldInstance.getLoc(), oldInstance);
  SmallVector<Type, 8> resultTypes;
  SmallVector<Value, 8> operands;
  for (size_t portIndex = 0, e = portInfo.size(); portIndex != e; ++portIndex) {
    auto &port = portInfo[portIndex];
    auto portType = lowerType(port.type);
    if (!portType) {
      oldInstance->emitOpError("could not lower type of port ") << port.name;
      return;
    }

    if (port.isOutput()) {
      // Drop zero bit results.
      if (!portType.isInteger(0))
        resultTypes.push_back(portType);
      continue;
    }

    // If we can find the connects to this port, then we can directly
    // materialize it.
    auto portResult = oldInstance.getPortNamed(port.name);
    assert(portResult && "invalid IR, couldn't find port");
    if (auto value = tryEliminatingConnectsToValue(portResult, oldInstance)) {
      // If we got a value connecting to the input port, then we can pass it
      // into the RTL instance without a temporary wire.
      operands.push_back(value);
      continue;
    }

    // Otherwise, create a wire for each input/inout operand, so there is
    // something to connect to.
    auto name = builder.getStringAttr("." + port.getName().str() + ".wire");
    auto wire = builder.create<WireOp>(port.type, name);

    // Drop zero bit input/inout ports.
    if (!portType.isInteger(0))
      operands.push_back(castFromFIRRTLType(wire, portType, builder));

    portResult.replaceAllUsesWith(wire);
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

    auto resultType = FlipType::get(port.type);
    Value resultVal;
    if (port.type.getPassiveType().getBitWidthOrSentinel() != 0) {
      // Cast the value to the right signedness and flippedness.
      resultVal = newInst.getResult(resultNo++);
      resultVal = castToFIRRTLType(resultVal, resultType, builder);
    } else {
      // Zero bit results are just replaced with a wire.
      resultVal = builder.create<WireOp>(
          resultType, "." + port.getName().str() + ".0width_result");
    }

    // Replace uses of the old output port with the returned value directly.
    auto portResult = oldInstance.getPortNamed(port.name);
    assert(portResult && "invalid IR, couldn't find port");
    portResult.replaceAllUsesWith(resultVal);
  }

  // Done with the oldInstance!
  oldInstance.erase();
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
  Value getLoweredAndExtOrTruncValue(Value value, Type destType);
  LogicalResult setLowering(Value orig, Value result);
  template <typename ResultOpType, typename... CtorArgTypes>
  LogicalResult setLoweringTo(Operation *orig, CtorArgTypes... args);
  void emitRandomizePrologIfNeeded();

  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitExpr;
  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitDecl;
  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitStmt;

  // Lowering hooks.
  enum UnloweredOpResult { AlreadyLowered, NowLowered, LoweringFailure };
  UnloweredOpResult handleUnloweredOp(Operation *op);
  LogicalResult visitExpr(ConstantOp op);
  LogicalResult visitExpr(SubfieldOp op);
  LogicalResult visitUnhandledOp(Operation *op) { return failure(); }
  LogicalResult visitInvalidOp(Operation *op) { return failure(); }

  // Declarations.
  LogicalResult visitDecl(WireOp op);
  LogicalResult visitDecl(NodeOp op);
  LogicalResult visitDecl(RegOp op);
  LogicalResult visitDecl(RegResetOp op);
  LogicalResult visitDecl(MemOp op);

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
  template <typename AOpTy, typename BOpTy>
  LogicalResult lowerVerificationStatement(AOpTy op);

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
  LogicalResult visitExpr(InvalidValuePrimOp op);
  LogicalResult visitExpr(HeadPrimOp op);
  LogicalResult visitExpr(ShlPrimOp op);
  LogicalResult visitExpr(ShrPrimOp op);
  LogicalResult visitExpr(DShlPrimOp op) {
    return lowerDivLikeOp<rtl::ShlOp, rtl::ShlOp>(op);
  }
  LogicalResult visitExpr(DShrPrimOp op) {
    return lowerDivLikeOp<rtl::ShrSOp, rtl::ShrUOp>(op);
  }
  LogicalResult visitExpr(DShlwPrimOp op) {
    return lowerDivLikeOp<rtl::ShrSOp, rtl::ShrUOp>(op);
  }
  LogicalResult visitExpr(TailPrimOp op);
  LogicalResult visitExpr(MuxPrimOp op);
  LogicalResult visitExpr(ValidIfPrimOp op);

  // Statements
  LogicalResult visitStmt(SkipOp op);
  LogicalResult visitStmt(ConnectOp op);
  LogicalResult visitStmt(PartialConnectOp op);
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

  /// This is true if we've emitted `INIT_RANDOM_PROLOG_ into an initial block
  /// in this module already.
  bool randomizePrologEmitted;
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLToRTLPass() {
  return std::make_unique<FIRRTLLowering>();
}

// This is the main entrypoint for the lowering pass.
void FIRRTLLowering::runOnOperation() {
  // FIRRTL FModule is a single block because FIRRTL ops are a DAG.  Walk
  // through each operation, lowering each in turn if we can, introducing casts
  // if we cannot.
  auto *body = getOperation().getBodyBlock();

  randomizePrologEmitted = false;

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
      switch (handleUnloweredOp(&op)) {
      case AlreadyLowered:
        break;         // Something like rtl.output, which is already lowered.
      case NowLowered: // Something handleUnloweredOp removed.
        opsToRemove.push_back(&op);
        break;
      case LoweringFailure:
        // If lowering failed, don't remove *anything* we've lowered so far,
        // there may be uses, and the pass will fail anyway.
        opsToRemove.clear();
      }
    }
  }
  builder = nullptr;

  // Now that all of the operations that can be lowered are, remove the original
  // values.  We know that any lowered operations will be dead (if removed in
  // reverse order) at this point - any users of them from unremapped operations
  // will be changed to use the newly lowered ops.
  while (!opsToRemove.empty()) {
    assert(opsToRemove.back()->use_empty() &&
           "Should remove ops in reverse order of visitation");
    opsToRemove.pop_back_val()->erase();
  }

  // Clear out the value mapping for next time, so we don't have dangling keys.
  valueMapping.clear();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Zero bit operands end up looking like failures from getLoweredValue.  This
/// helper function invokes the closure specified if the operand was actually
/// zero bit, or returns failure() if it was some other kind of failure.
static LogicalResult handleZeroBit(Value failedOperand,
                                   std::function<LogicalResult()> fn) {
  assert(failedOperand && failedOperand.getType().isa<FIRRTLType>() &&
         "Should be called on the failed FIRRTL operand");
  if (!isZeroBitFIRRTLType(failedOperand.getType()))
    return failure();
  return fn();
}

/// Return the lowered RTL value corresponding to the specified original value.
/// This returns a null value for FIRRTL values that haven't be lowered, e.g.
/// unknown width integers.  This returns rtl::inout type values if present, it
/// does not implicitly read from them.
Value FIRRTLLowering::getPossiblyInoutLoweredValue(Value value) {
  assert(value.getType().isa<FIRRTLType>() &&
         "Should only lower FIRRTL operands");
  // If we lowered this value, then return the lowered value, otherwise fail.
  auto it = valueMapping.find(value);
  return it != valueMapping.end() ? it->second : Value();
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

  // We only know how to extend integer types with known width.
  auto destWidth = destType.cast<FIRRTLType>().getBitWidthOrSentinel();
  if (destWidth == -1)
    return {};

  auto result = getLoweredValue(value);
  if (!result) {
    // If this was a zero bit operand being extended, then produce a zero of the
    // right result type.  If it is just a failure, fail.
    if (!isZeroBitFIRRTLType(value.getType()))
      return {};
    // Zero bit results have to be returned as null.  The caller can handle
    // this if they want to.
    if (destWidth == 0)
      return {};
    // Otherwise, FIRRTL semantics is that an extension from a zero bit value
    // always produces a zero value in the destination width.
    return builder->create<rtl::ConstantOp>(APInt(destWidth, 0));
  }

  auto srcWidth = result.getType().cast<IntegerType>().getWidth();
  if (srcWidth == unsigned(destWidth))
    return result;

  if (srcWidth > unsigned(destWidth)) {
    builder->emitError("operand should not be a truncation");
    return {};
  }

  auto resultType = builder->getIntegerType(destWidth);

  // Extension follows the sign of the source value, not the destination.
  auto valueFIRType = value.getType().cast<FIRRTLType>().getPassiveType();
  if (valueFIRType.cast<IntType>().isSigned())
    return builder->createOrFold<rtl::SExtOp>(resultType, result);

  auto zero = builder->create<rtl::ConstantOp>(APInt(destWidth - srcWidth, 0));
  return builder->createOrFold<rtl::ConcatOp>(zero, result);
}

/// Return the lowered value corresponding to the specified original value and
/// then extended or truncated to match the width of destType if needed.
///
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredAndExtOrTruncValue(Value value, Type destType) {
  assert(value.getType().isa<FIRRTLType>() && destType.isa<FIRRTLType>() &&
         "input/output value should be FIRRTL");

  // We only know how to adjust integer types with known width.
  auto destWidth = destType.cast<FIRRTLType>().getBitWidthOrSentinel();
  if (destWidth == -1)
    return {};

  auto result = getLoweredValue(value);
  if (!result) {
    // If this was a zero bit operand being extended, then produce a zero of the
    // right result type.  If it is just a failure, fail.
    if (!isZeroBitFIRRTLType(value.getType()))
      return {};
    // Zero bit results have to be returned as null.  The caller can handle
    // this if they want to.
    if (destWidth == 0)
      return {};
    // Otherwise, FIRRTL semantics is that an extension from a zero bit value
    // always produces a zero value in the destination width.
    return builder->create<rtl::ConstantOp>(APInt(destWidth, 0));
  }

  auto srcWidth = result.getType().cast<IntegerType>().getWidth();
  if (srcWidth == unsigned(destWidth))
    return result;

  if (srcWidth > unsigned(destWidth)) {
    auto resultType = builder->getIntegerType(destWidth);
    return builder->createOrFold<rtl::ExtractOp>(resultType, result, 0);
  } else {
    auto resultType = builder->getIntegerType(destWidth);

    // Extension follows the sign of the source value, not the destination.
    auto valueFIRType = value.getType().cast<FIRRTLType>().getPassiveType();
    if (valueFIRType.cast<IntType>().isSigned())
      return builder->createOrFold<rtl::SExtOp>(resultType, result);

    auto zero =
        builder->create<rtl::ConstantOp>(APInt(destWidth - srcWidth, 0));
    return builder->createOrFold<rtl::ConcatOp>(zero, result);
  }
}

/// Set the lowered value of 'orig' to 'result', remembering this in a map.
/// This always returns success() to make it more convenient in lowering code.
///
/// Note that result may be null here if we're lowering orig to a zero-bit
/// value.
///
LogicalResult FIRRTLLowering::setLowering(Value orig, Value result) {
  assert(orig.getType().isa<FIRRTLType>() &&
         (!result || !result.getType().isa<FIRRTLType>()) &&
         "Lowering didn't turn a FIRRTL value into a non-FIRRTL value");

#ifndef NDEBUG
  auto srcWidth = orig.getType()
                      .cast<FIRRTLType>()
                      .getPassiveType()
                      .getBitWidthOrSentinel();

  // Caller should pass null value iff this was a zero bit value.
  if (srcWidth != -1) {
    if (result)
      assert((srcWidth != 0) &&
             "Lowering produced value for zero width source");
    else
      assert((srcWidth == 0) &&
             "Lowering produced null value but source wasn't zero width");
  }
#endif

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

/// Handle the case where an operation wasn't lowered.  When this happens, the
/// operands should just be unlowered non-FIRRTL values.  If the operand was
/// not lowered then leave it alone, otherwise we have a problem with lowering.
///
FIRRTLLowering::UnloweredOpResult
FIRRTLLowering::handleUnloweredOp(Operation *op) {
  // Scan the operand list for the operation to see if none were lowered.  In
  // that case the operation must be something lowered to RTL already, e.g. the
  // rtl.output operation.  This is success for us because it is already
  // lowered.
  if (llvm::all_of(op->getOpOperands(), [&](auto &operand) -> bool {
        return !valueMapping.count(operand.get());
      })) {
    return AlreadyLowered;
  }

  // Ok, at least one operand got lowered, so this operation is using a FIRRTL
  // value, but wasn't itself lowered.  This is because the lowering is
  // incomplete. This is either a bug or incomplete implementation.
  //
  // There is one aspect of incompleteness we intentionally expect: we allow
  // primitive operations that produce a zero bit result to be ignored by the
  // lowering logic.  They don't have side effects, and handling this corner
  // case just complicates each of the lowering hooks. Instead, we just handle
  // them all right here.
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    if (resultType.isa<FIRRTLType>() && isZeroBitFIRRTLType(resultType) &&
        (isExpression(op) || isa<AsPassivePrimOp>(op) ||
         isa<AsNonPassivePrimOp>(op))) {
      // Zero bit values lower to the null Value.
      setLowering(op->getResult(0), Value());
      return NowLowered;
    }
  }
  op->emitOpError("LowerToRTL couldn't handle this operation");
  return LoweringFailure;
}

LogicalResult FIRRTLLowering::visitExpr(ConstantOp op) {
  return setLoweringTo<rtl::ConstantOp>(op, op.value());
}

LogicalResult FIRRTLLowering::visitExpr(SubfieldOp op) {
  // Subfield operations should either be dead or have a lowering installed
  // already.  They only come up with mems.
  if (op.use_empty() || valueMapping.count(op))
    return success();

  op.emitOpError("operand should have lowered its subfields");
  return failure();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitDecl(WireOp op) {
  auto resultType = lowerType(op.result().getType());
  if (!resultType)
    return failure();

  if (resultType.isInteger(0))
    return setLowering(op, Value());

  // Convert the inout to a non-inout type.
  return setLoweringTo<rtl::WireOp>(op, resultType, op.nameAttr());
}

LogicalResult FIRRTLLowering::visitDecl(NodeOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return handleZeroBit(op.input(),
                         [&]() { return setLowering(op, Value()); });

  // Node operations are logical noops, but can carry a name.  If a name is
  // present then we lower this into a wire and a connect, otherwise we just
  // drop it.
  if (auto name = op->getAttrOfType<StringAttr>("name")) {
    if (!name.getValue().empty()) {
      auto wire = builder->create<rtl::WireOp>(operand.getType(), name);
      builder->create<rtl::ConnectOp>(wire, operand);
    }
  }

  // TODO(clattner): This is dropping the location information from unnamed node
  // ops.  I suspect that this falls below the fold in terms of things we care
  // about given how Chisel works, but we should reevaluate with more
  // information.
  return setLowering(op, operand);
}

/// Emit a `INIT_RANDOM_PROLOG_ statement into the current block.  This should
/// already be within an `ifndef SYNTHESIS + initial block.
void FIRRTLLowering::emitRandomizePrologIfNeeded() {
  if (randomizePrologEmitted)
    return;

  builder->create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");
  randomizePrologEmitted = true;
}

LogicalResult FIRRTLLowering::visitDecl(RegOp op) {
  auto resultType = lowerType(op.result().getType());
  if (!resultType)
    return failure();
  if (resultType.isInteger(0))
    return setLowering(op, Value());

  auto regResult = builder->create<sv::RegOp>(resultType, op.nameAttr());
  setLowering(op, regResult);

  // Emit the initializer expression for simulation that fills it with random
  // value.
  builder->create<sv::IfDefOp>("!SYNTHESIS", [&]() {
    builder->create<sv::InitialOp>([&]() {
      emitRandomizePrologIfNeeded();

      builder->create<sv::IfDefOp>("RANDOMIZE_REG_INIT", [&]() {
        auto type = regResult.getType().getElementType();
        auto randomVal = builder->create<sv::TextualValueOp>(type, "`RANDOM");
        builder->create<sv::BPAssignOp>(regResult, randomVal);
      });
    });
  });

  return success();
}

LogicalResult FIRRTLLowering::visitDecl(RegResetOp op) {
  auto resultType = lowerType(op.result().getType());
  if (!resultType)
    return failure();
  if (resultType.isInteger(0))
    return setLowering(op, Value());

  Value clockVal = getLoweredValue(op.clockVal());
  Value resetSignal = getLoweredValue(op.resetSignal());
  Value resetValue = getLoweredValue(op.resetValue());

  if (!clockVal || !resetSignal || !resetValue)
    return failure();

  auto regResult = builder->create<sv::RegOp>(resultType, op.nameAttr());
  setLowering(op, regResult);

  auto resetFn = [&]() {
    builder->create<sv::IfOp>(resetSignal, [&]() {
      builder->create<sv::PAssignOp>(regResult, resetValue);
    });
  };

  if (op.resetSignal().getType().isa<AsyncResetType>()) {
    builder->create<sv::AlwaysOp>(
        ArrayRef<EventControl>{EventControl::AtPosEdge,
                               EventControl::AtPosEdge},
        ArrayRef<Value>{clockVal, resetSignal}, resetFn);
  } else { // sync reset
    builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clockVal, resetFn);
  }

  // Emit the initializer expression for simulation that fills it with random
  // value.
  builder->create<sv::IfDefOp>("!SYNTHESIS", [&]() {
    builder->create<sv::InitialOp>([&]() {
      emitRandomizePrologIfNeeded();

      // When RANDOMIZE_REG_INIT is enabled, we assign a random value to the reg
      // if the reset line is low at start.
      builder->create<sv::IfDefOp>("RANDOMIZE_REG_INIT", [&]() {
        auto one = builder->create<rtl::ConstantOp>(APInt(1, 1));
        auto notResetValue = builder->create<rtl::XorOp>(resetSignal, one);
        builder->create<sv::IfOp>(notResetValue, [&]() {
          auto type = regResult.getType().getElementType();
          auto randomVal = builder->create<sv::TextualValueOp>(type, "`RANDOM");
          builder->create<sv::BPAssignOp>(regResult, randomVal);
        });
      });
    });
  });

  return success();
}

namespace {
/// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  /// This is the underlying ground type of the field.
  Type type;
  /// This is a field name with any suffixes to make it unique.
  std::string name;
};
} // end anonymous namespace.

/// Convert a nested bundle of fields into a flat list of fields.  This is used
/// when working with mems to flatten them.  Return true if we fail to lower
/// any of the element types, or false if successul.
static bool flattenBundleTypes(Type type, StringRef nameSoFar,
                               SmallVectorImpl<FlatBundleFieldEntry> &results) {
  // Ignore flips.
  if (auto flip = type.dyn_cast<FlipType>())
    return flattenBundleTypes(flip.getElementType(), nameSoFar, results);

  // In the base case we record this field.
  auto bundle = type.dyn_cast<BundleType>();
  if (!bundle) {
    auto rtlType = lowerType(type);
    if (rtlType && !rtlType.isInteger(0))
      results.push_back({rtlType, nameSoFar.str()});
    return rtlType == Type();
  }

  SmallString<16> tmpName(nameSoFar);

  // Otherwise, we have a bundle type.  Break it down.
  for (auto &elt : bundle.getElements()) {
    // Construct the suffix to pass down.
    tmpName.resize(nameSoFar.size());
    tmpName.push_back('_');
    tmpName.append(elt.name.strref());
    // Recursively process subelements.
    if (flattenBundleTypes(elt.type, tmpName, results))
      return true;
  }
  return false;
}

LogicalResult FIRRTLLowering::visitDecl(MemOp op) {
  if (op.readLatency() != 0 || op.writeLatency() != 1) {
    // FIXME: This should be an error.
    op.emitWarning("FIXME: need to support mem read/write latency correctly");
  }

  StringRef memName = "mem";
  if (op.name().hasValue())
    memName = op.name().getValue();

  // Aggregate mems may declare multiple reg's.  We need to declare and random
  // initialize them all.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  if (auto dataType = op.getDataTypeOrNull()) {
    if (flattenBundleTypes(dataType, memName, fieldTypes))
      return op.emitError("could not lower mem element type");
  }

  uint64_t depth = op.depth();

  // Add one reg declaration for each field of the mem.
  SmallVector<Value> regs;
  for (const auto &field : fieldTypes) {
    auto resultType = rtl::UnpackedArrayType::get(field.type, depth);
    auto name = builder->getStringAttr(field.name);
    regs.push_back(builder->create<sv::RegOp>(resultType, name));
  }

  // Emit the initializer expression for simulation that fills it with random
  // value.
  builder->create<sv::IfDefOp>("!SYNTHESIS", [&]() {
    builder->create<sv::InitialOp>([&]() {
      emitRandomizePrologIfNeeded();

      builder->create<sv::IfDefOp>("RANDOMIZE_MEM_INIT", [&]() {
        if (depth == 1) { // Don't emit a for loop for one element.
          for (Value reg : regs) {
            auto type = rtl::getInOutElementType(reg.getType());
            type = rtl::getAnyRTLArrayElementType(type);
            auto randomVal =
                builder->create<sv::TextualValueOp>(type, "`RANDOM");
            auto zero = builder->create<rtl::ConstantOp>(APInt(1, 0));
            auto subscript = builder->create<rtl::ArrayIndexOp>(reg, zero);
            builder->create<sv::BPAssignOp>(subscript, randomVal);
          }
        } else if (!regs.empty()) {
          assert(depth < (1ULL << 31) && "FIXME: Our initialization logic uses "
                                         "'integer' which doesn't support "
                                         "mems greater than 2^32");

          std::string action = "integer {{0}}_initvar;\n";
          action += "for ({{0}}_initvar = 0; {{0}}_initvar < " +
                    llvm::utostr(depth) + "; {{0}}_initvar = {{0}}_initvar+1)";
          if (regs.size() != 1)
            action += " begin";

          for (size_t i = 0, e = regs.size(); i != e; ++i)
            action +=
                "\n  {{" + llvm::utostr(i) + "}}[{{0}}_initvar] = `RANDOM;";

          if (regs.size() != 1)
            action += "\nend";
          builder->create<sv::VerbatimOp>(action, regs);
        }
      });
    });
  });

  // Keep track of whether this mem is an even power of two or not.
  bool isPowerOfTwo = llvm::isPowerOf2_64(depth);

  // Lower all of the read/write ports.  Each port is a separate
  // return value of the memory.
  auto namesArray = op.portNames();
  for (size_t i = 0, e = namesArray.size(); i != e; ++i) {

    auto portName = namesArray[i].cast<StringAttr>().getValue();
    auto port = op.getPortNamed(portName);
    auto portBundleType =
        port.getType().cast<FIRRTLType>().getPassiveType().cast<BundleType>();

    SmallVector<std::pair<Identifier, Value>> portWires;
    for (BundleType::BundleElement elt : portBundleType.getElements()) {
      auto fieldType = lowerType(elt.type);
      if (!fieldType)
        return op.emitOpError("port " + elt.name.str() +
                              " has unexpected field");

      if (fieldType.isInteger(0)) {
        portWires.push_back({elt.name, Value()});
        continue;
      }
      auto name =
          (Twine(memName) + "_" + portName + "_" + elt.name.str()).str();
      auto fieldWire = builder->create<rtl::WireOp>(fieldType, name);
      portWires.push_back({elt.name, fieldWire});
    }

    // Return the inout wire corresponding to a port field.
    auto getPortFieldWire = [&](StringRef portName) -> Value {
      for (auto entry : portWires) {
        if (entry.first.strref() == portName)
          return entry.second;
      }
      llvm_unreachable("unknown port wire!");
    };

    // Now that we have the wires for each element, rewrite any subfields to use
    // them instead of the subfields.
    while (!port.use_empty()) {
      auto portField = cast<SubfieldOp>(*port.user_begin());
      portField->dropAllReferences();
      setLowering(portField, getPortFieldWire(portField.fieldname()));
    }

    // Return the value corresponding to a port field.
    auto getPortFieldValue = [&](StringRef portName) -> Value {
      return builder->create<rtl::ReadInOutOp>(getPortFieldWire(portName));
    };

    switch (op.getPortKind(portName).getValue()) {
    case MemOp::PortKind::ReadWrite:
      op.emitOpError("readwrite ports should be lowered into separate read and "
                     "write ports by previous passes");
      continue;
    case MemOp::PortKind::Read: {
      auto emitReads = [&](bool masked) {
        // TODO: not handling bundle elements correctly yet.
        if (regs.size() > 1)
          op.emitOpError("don't support bundle elements yet");

        // Emit an assign to the read port, using the address.
        // TODO(firrtl-spec): It appears that the clock signal on the read port
        // is ignored, why does it exist?
        for (auto reg : regs) {
          auto addr = getPortFieldValue("addr");
          Value value = builder->create<rtl::ArrayIndexOp>(reg, addr);
          value = builder->create<rtl::ReadInOutOp>(value);

          // If we're masking, emit "addr < Depth ? mem[addr] : `RANDOM".
          if (masked) {
            auto addrWidth = addr.getType().getIntOrFloatBitWidth();
            auto depthCst =
                builder->create<rtl::ConstantOp>(APInt(addrWidth, depth));
            auto cmp = builder->create<rtl::ICmpOp>(ICmpPredicate::ult, addr,
                                                    depthCst);
            auto randomVal =
                builder->create<sv::TextualValueOp>(value.getType(), "`RANDOM");
            value = builder->create<rtl::MuxOp>(cmp, value, randomVal);
          }

          // FIXME: This isn't right for multi-slot data's.
          builder->create<rtl::ConnectOp>(getPortFieldWire("data"), value);
        }
      };

      // If the memory size is a power of two, then we can just unconditionally
      // read from it, otherwise we emit #ifdef'd masking logic.
      if (isPowerOfTwo) {
        emitReads(false);
      } else {
        builder->create<sv::IfDefOp>(
            "!RANDOMIZE_GARBAGE_ASSIGN", [&]() { emitReads(false); },
            [&]() { emitReads(true); });
      }
      break;
    }

    case MemOp::PortKind::Write: {
      // TODO: not handling bundle elements correctly yet.
      if (regs.size() > 1)
        op.emitOpError("don't support bundle elements yet");

      // Emit something like:
      // always @(posedge _M_write_clk) begin
      //   if (_M_write_en & _M_write_mask)
      //     _M[_M_write_addr] <= _M_write_data;
      // end
      auto clock = getPortFieldValue("clk");
      builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clock, [&]() {
        auto enable = getPortFieldValue("en");
        auto mask = getPortFieldValue("mask");
        auto cond = builder->create<rtl::AndOp>(enable, mask);
        builder->create<sv::IfOp>(cond, [&]() {
          // FIXME: This isn't right for multi-slot data mems.
          auto data = getPortFieldValue("data");
          auto addr = getPortFieldValue("addr");

          for (auto reg : regs) {
            auto slot = builder->create<rtl::ArrayIndexOp>(reg, addr);
            builder->create<sv::BPAssignOp>(slot, data);
          }
        });
      });

      break;
    }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Lower a cast that is a noop at the RTL level.
LogicalResult FIRRTLLowering::lowerNoopCast(Operation *op) {
  auto operand = getPossiblyInoutLoweredValue(op->getOperand(0));
  if (!operand)
    return failure();

  // Noop cast.
  return setLowering(op->getResult(0), operand);
}

LogicalResult FIRRTLLowering::visitExpr(StdIntCastOp op) {
  // Conversions from standard integer types to FIRRTL types are lowered as
  // the input operand.
  if (auto opIntType = op.getOperand().getType().dyn_cast<IntegerType>()) {
    if (opIntType.getWidth() != 0)
      return setLowering(op, op.getOperand());
    else
      return setLowering(op, Value());
  }

  // Otherwise must be a conversion from FIRRTL type to standard int type.
  auto result = getLoweredValue(op.getOperand());
  if (!result) {
    // If this is a conversion from a zero bit RTL type to firrtl value, then
    // we want to successfully lower this to a null Value.
    if (op.getOperand().getType().isSignlessInteger(0)) {
      return setLowering(op, Value());
    }
    return failure();
  }

  // We lower firrtl.stdIntCast converting from a firrtl type to a standard
  // type into the lowered operand.
  op.replaceAllUsesWith(result);
  return success();
}

LogicalResult FIRRTLLowering::visitExpr(AnalogInOutCastOp op) {
  // Standard -> FIRRTL.
  if (!op.getOperand().getType().isa<FIRRTLType>())
    return setLowering(op, op.getOperand());

  // FIRRTL -> Standard.
  auto result = getPossiblyInoutLoweredValue(op.getOperand());
  if (!result)
    return failure();

  return setLowering(op, result);
}

LogicalResult FIRRTLLowering::visitExpr(CvtPrimOp op) {
  auto operand = getLoweredValue(op.getOperand());
  if (!operand) {
    return handleZeroBit(op.getOperand(), [&]() {
      // Unsigned zero bit to Signed is 1b0.
      if (op.getOperand().getType().cast<IntType>().isUnsigned())
        return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 0));
      // Signed->Signed is a zero bit value.
      return setLowering(op, Value());
    });
  }

  // Signed to signed is a noop.
  if (op.getOperand().getType().cast<IntType>().isSigned())
    return setLowering(op, operand);

  // Otherwise prepend a zero bit.
  auto zero = builder->create<rtl::ConstantOp>(APInt(1, 0));
  return setLoweringTo<rtl::ConcatOp>(op, zero, operand);
}

LogicalResult FIRRTLLowering::visitExpr(NotPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();
  // ~x  ---> x ^ 0xFF
  auto allOnes = builder->create<rtl::ConstantOp>(operand.getType(), -1);
  return setLoweringTo<rtl::XorOp>(op, operand, allOnes);
}

LogicalResult FIRRTLLowering::visitExpr(NegPrimOp op) {
  // FIRRTL negate always adds a bit, and always does a sign extension even if
  // the input is unsigned.
  // -x  ---> 0-sext(x)
  auto operand = getLoweredValue(op.input());
  if (!operand) {
    return handleZeroBit(op.getOperand(), [&]() {
      // Negate of a zero bit value is 1b0.
      return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 0));
    });
  }
  auto resultType = lowerType(op.getType());
  operand = builder->createOrFold<rtl::SExtOp>(resultType, operand);

  auto zero = builder->create<rtl::ConstantOp>(resultType, 0);
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
  if (!operand) {
    return handleZeroBit(op.input(), [&]() {
      return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 0));
    });
    return failure();
  }

  return setLoweringTo<rtl::XorROp>(op, builder->getIntegerType(1), operand);
}

LogicalResult FIRRTLLowering::visitExpr(AndRPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand) {
    return handleZeroBit(op.input(), [&]() {
      return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 1));
    });
  }

  return setLoweringTo<rtl::AndROp>(op, builder->getIntegerType(1), operand);
}

LogicalResult FIRRTLLowering::visitExpr(OrRPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand) {
    return handleZeroBit(op.input(), [&]() {
      return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 0));
    });
    return failure();
  }

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

  return setLoweringTo<ResultOpType>(op, lhs, rhs);
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

  auto cmpType = getWidestIntType(lhsIntType, rhsIntType);
  if (cmpType.getWidth() == 0) // Handle 0-width inputs by promoting to 1 bit.
    cmpType = UIntType::get(&getContext(), 1);
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
  // rtl has equal types for these, firrtl doesn't.  The type of the firrtl
  // RHS may be wider than the LHS, and we cannot truncate off the high bits
  // (because an overlarge amount is supposed to shift in sign or zero bits).
  auto opType = op->getResult(0).getType().cast<IntType>();
  if (opType.getWidth() == 0)
    return setLowering(op->getResult(0), Value());

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
  if (!lhs) {
    return handleZeroBit(op.lhs(), [&]() {
      if (rhs) // cat(0bit, x) --> x
        return setLowering(op, rhs);
      // cat(0bit, 0bit) --> 0bit
      return handleZeroBit(op.rhs(),
                           [&]() { return setLowering(op, Value()); });
    });
  }

  if (!rhs) // cat(x, 0bit) --> x
    return handleZeroBit(op.rhs(), [&]() { return setLowering(op, lhs); });

  return setLoweringTo<rtl::ConcatOp>(op, lhs, rhs);
}

LogicalResult FIRRTLLowering::visitExpr(RemPrimOp op) {
  // FIRRTL has the width of (a % b) = Min(W(a), W(b)), but the operation is
  // done at max(W(a), W(b))) so we need to extend one operand, then truncate
  // the result.
  auto lhsFirTy = op.lhs().getType().dyn_cast<IntType>();
  auto rhsFirTy = op.rhs().getType().dyn_cast<IntType>();
  if (!lhsFirTy || !rhsFirTy || !lhsFirTy.hasWidth() || !rhsFirTy.hasWidth())
    return failure();
  auto opType = getWidestIntType(lhsFirTy, rhsFirTy);
  auto lhs = getLoweredAndExtendedValue(op.lhs(), opType);
  auto rhs = getLoweredAndExtendedValue(op.rhs(), opType);
  if (!lhs || !rhs)
    return failure();

  auto resultFirType = op.getType().cast<IntType>();
  if (!resultFirType.hasWidth())
    return failure();
  auto destWidth = unsigned(resultFirType.getWidthOrSentinel());
  if (destWidth == 0)
    return setLowering(op, Value());

  Value modInst;
  if (resultFirType.isUnsigned()) {
    modInst = builder->createOrFold<rtl::ModUOp>(lhs, rhs);
  } else {
    modInst = builder->createOrFold<rtl::ModSOp>(lhs, rhs);
  }

  auto resultType = builder->getIntegerType(destWidth);
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

LogicalResult FIRRTLLowering::visitExpr(InvalidValuePrimOp op) {
  auto resultTy = lowerType(op.getType());
  if (!resultTy)
    return failure();

  // We lower invalid to 0.  TODO: the FIRRTL spec mentions something about
  // lowering it to a random value, we should see if this is what we need to
  // do.
  auto value = builder->create<rtl::ConstantOp>(resultTy, 0);

  if (!op.getType().isa<AnalogType>())
    return setLowering(op, value);

  // Values of analog type always need to be lowered to something with inout
  // type.  We do that by lowering to a wire and return that.
  auto wire = builder->create<rtl::WireOp>(resultTy, ".invalid_analog");
  builder->create<rtl::ConnectOp>(wire, value);
  return setLowering(op, wire);
}

LogicalResult FIRRTLLowering::visitExpr(HeadPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();
  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  if (op.amount() == 0)
    return setLowering(op, Value());
  Type resultType = builder->getIntegerType(op.amount());
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input,
                                       inWidth - op.amount());
}

LogicalResult FIRRTLLowering::visitExpr(ShlPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input) {
    return handleZeroBit(op.input(), [&]() {
      if (op.amount() == 0)
        return failure();
      return setLoweringTo<rtl::ConstantOp>(op, APInt(op.amount(), 0));
    });
  }

  // Handle the degenerate case.
  if (op.amount() == 0)
    return setLowering(op, input);

  // TODO: We could keep track of zeros and implicitly CSE them.
  auto zero = builder->create<rtl::ConstantOp>(APInt(op.amount(), 0));
  return setLoweringTo<rtl::ConcatOp>(op, input, zero);
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
  if (inWidth == op.amount())
    return setLowering(op, Value());
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

LogicalResult FIRRTLLowering::visitStmt(SkipOp op) {
  // Nothing!  We could emit an comment as a verbatim op if there were a reason
  // to.
  return success();
}

LogicalResult FIRRTLLowering::visitStmt(ConnectOp op) {
  auto dest = op.dest();
  // The source can be a smaller integer, extend it as appropriate if so.
  auto destType = dest.getType().cast<FIRRTLType>().getPassiveType();
  auto srcVal = getLoweredAndExtendedValue(op.src(), destType);
  if (!srcVal)
    return handleZeroBit(op.src(), []() { return success(); });

  auto destVal = getPossiblyInoutLoweredValue(dest);
  if (!destVal)
    return failure();

  if (!destVal.getType().isa<rtl::InOutType>())
    return op.emitError("destination isn't an inout type");

  // If this is an assignment to a register, then the connect implicitly
  // happens under the clock that gates the register.
  if (auto regOp = dyn_cast_or_null<RegOp>(dest.getDefiningOp())) {
    Value clockVal = getLoweredValue(regOp.clockVal());
    if (!clockVal)
      return failure();

    builder->create<sv::AlwaysFFOp>(EventControl::AtPosEdge, clockVal, [&]() {
      builder->create<sv::PAssignOp>(destVal, srcVal);
    });
    return success();
  }

  // If this is an assignment to a RegReset, then the connect implicitly
  // happens under the clock and reset that gate the register.
  if (auto regResetOp = dyn_cast_or_null<RegResetOp>(dest.getDefiningOp())) {
    Value clockVal = getLoweredValue(regResetOp.clockVal());
    Value resetSignal = getLoweredValue(regResetOp.resetSignal());
    if (!clockVal || !resetSignal)
      return failure();

    //    auto one = builder->create<rtl::ConstantOp>(APInt(1, 1));
    //    auto invResetSignal = builder->create<rtl::XorOp>(resetSignal, one);

    builder->create<sv::AlwaysFFOp>(
        EventControl::AtPosEdge, clockVal,
        regResetOp.resetSignal().getType().isa<AsyncResetType>()
            ? ::ResetType::AsyncReset
            : ::ResetType::SyncReset,
        EventControl::AtPosEdge, resetSignal, std::function<void()>(),
        [&]() { builder->create<sv::PAssignOp>(destVal, srcVal); });
    return success();
  }

  builder->create<rtl::ConnectOp>(destVal, srcVal);
  return success();
}

// This will have to handle struct connects at some point.
LogicalResult FIRRTLLowering::visitStmt(PartialConnectOp op) {
  auto dest = op.dest();
  // The source can be a different size integer, adjust it as appropriate if so.
  auto destType = dest.getType().cast<FIRRTLType>().getPassiveType();
  auto srcVal = getLoweredAndExtOrTruncValue(op.src(), destType);
  if (!srcVal)
    return handleZeroBit(op.src(), []() { return success(); });

  auto destVal = getPossiblyInoutLoweredValue(dest);
  if (!destVal)
    return failure();

  if (!destVal.getType().isa<rtl::InOutType>())
    return op.emitError("destination isn't an inout type");

  // If this is an assignment to a register, then the connect implicitly
  // happens under the clock that gates the register.
  if (auto regOp = dyn_cast_or_null<RegOp>(dest.getDefiningOp())) {
    Value clockVal = getLoweredValue(regOp.clockVal());
    if (!clockVal)
      return failure();

    builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clockVal, [&]() {
      builder->create<sv::PAssignOp>(destVal, srcVal);
    });
    return success();
  }

  // If this is an assignment to a RegInit, then the connect implicitly
  // happens under the clock and reset that gate the register.
  if (auto regResetOp = dyn_cast_or_null<RegResetOp>(dest.getDefiningOp())) {
    Value clockVal = getLoweredValue(regResetOp.clockVal());
    Value resetSignal = getLoweredValue(regResetOp.resetSignal());
    if (!clockVal || !resetSignal)
      return failure();

    auto one = builder->create<rtl::ConstantOp>(APInt(1, 1));
    auto invResetSignal = builder->create<rtl::XorOp>(resetSignal, one);

    auto resetFn = [&]() {
      builder->create<sv::IfOp>(invResetSignal, [&]() {
        builder->create<sv::PAssignOp>(destVal, srcVal);
      });
    };

    if (regResetOp.resetSignal().getType().isa<AsyncResetType>()) {
      builder->create<sv::AlwaysOp>(
          ArrayRef<EventControl>{EventControl::AtPosEdge,
                                 EventControl::AtPosEdge},
          ArrayRef<Value>{clockVal, resetSignal}, resetFn);
      return success();
    } else { // sync reset
      builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clockVal, resetFn);
    }
    return success();
  }

  builder->create<rtl::ConnectOp>(destVal, srcVal);
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
    if (!operands.back()) {
      // If this is a zero bit operand, just pass a one bit zero.
      if (!isZeroBitFIRRTLType(operand.getType()))
        return failure();
      operands.back() = builder->create<rtl::ConstantOp>(APInt(1, 0));
    }
  }

  // Emit this into an "sv.always posedge" body.
  builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clock, [&]() {
    // Emit an "#ifndef SYNTHESIS" guard into the always block.
    builder->create<sv::IfDefOp>("!SYNTHESIS", [&]() {
      // Emit an "sv.if '`PRINTF_COND_ & cond' into the #ifndef.
      Value ifCond =
          builder->create<sv::TextualValueOp>(cond.getType(), "`PRINTF_COND_");
      ifCond = builder->createOrFold<rtl::AndOp>(ifCond, cond);

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
  auto clock = getLoweredValue(op.clock());
  auto cond = getLoweredValue(op.cond());
  if (!clock || !cond)
    return failure();

  // Emit this into an "sv.always posedge" body.
  builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clock, [&]() {
    // Emit an "#ifndef SYNTHESIS" guard into the always block.
    builder->create<sv::IfDefOp>("!SYNTHESIS", [&]() {
      // Emit an "sv.if '`STOP_COND_ & cond' into the #ifndef.
      Value ifCond =
          builder->create<sv::TextualValueOp>(cond.getType(), "`STOP_COND_");
      ifCond = builder->createOrFold<rtl::AndOp>(ifCond, cond);
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

  builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clock, [&]() {
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
    if (!inoutValues.back()) {
      // Ignore zero bit values.
      if (!isZeroBitFIRRTLType(v.getType()))
        return failure();
      inoutValues.pop_back();
      continue;
    }

    if (!inoutValues.back().getType().isa<rtl::InOutType>())
      return op.emitError("operand isn't an inout type");
  }

  if (inoutValues.size() < 2)
    return success();

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
