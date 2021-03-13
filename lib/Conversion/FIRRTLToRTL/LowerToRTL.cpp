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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Parallel.h"

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

  if (BundleType bundle = firType.dyn_cast<BundleType>()) {
    mlir::SmallVector<rtl::StructType::FieldInfo, 8> rtlfields;
    for (auto element : bundle.getElements()) {
      Type etype = lowerType(element.type);
      if (!etype)
        return {};
      // TODO: make rtl::StructType contain StringAttrs.
      auto name = Identifier::get(element.name.getValue(), type.getContext());
      rtlfields.push_back(rtl::StructType::FieldInfo{name, etype});
    }
    return rtl::StructType::get(type.getContext(), rtlfields);
  }

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

  if (BundleType bundle = type.dyn_cast<BundleType>()) {
    val = builder.createOrFold<RTLStructCastOp>(firType.getPassiveType(), val);
  } else {
    val = builder.createOrFold<StdIntCastOp>(firType.getPassiveType(), val);
  }

  // Handle the flip type if needed.
  if (type != val.getType())
    val = builder.createOrFold<AsNonPassivePrimOp>(firType, val);
  return val;
}

/// Cast from a FIRRTL type (potentially with a flip) to a standard type.
static Value castFromFIRRTLType(Value val, Type type,
                                ImplicitLocOpBuilder &builder) {
  if (type.isa<rtl::InOutType>() && val.getType().isa<AnalogType>())
    return builder.createOrFold<AnalogInOutCastOp>(type, val);

  // Strip off Flip type if needed.
  val = builder.createOrFold<AsPassivePrimOp>(val);
  if (rtl::StructType structTy = type.dyn_cast<rtl::StructType>())
    return builder.createOrFold<RTLStructCastOp>(type, val);
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
  rtl::RTLModuleExternOp lowerExtModule(FExtModuleOp oldModule,
                                        Block *topLevelModule);

  void lowerModuleBody(FModuleOp oldModule,
                       DenseMap<Operation *, Operation *> &oldToNewModuleMap);

  void lowerInstance(InstanceOp instance, CircuitOp circuitOp,
                     DenseMap<Operation *, Operation *> &oldToNewModuleMap);
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::createLowerFIRRTLToRTLModulePass() {
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

  SmallVector<FModuleOp, 32> modulesToProcess;

  // Iterate through each operation in the circuit body, transforming any
  // FModule's we come across.
  for (auto &op : circuitBody->getOperations()) {
    if (auto module = dyn_cast<FModuleOp>(op)) {
      oldToNewModuleMap[&op] = lowerModule(module, moduleBody);
      modulesToProcess.push_back(module);
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
  // any instances that refer to the old modules.
  if (getContext().isMultithreadingEnabled()) {
    mlir::ParallelDiagnosticHandler diagHandler(&getContext());
    llvm::parallelForEachN(0, modulesToProcess.size(), [&](auto index) {
      lowerModuleBody(modulesToProcess[index], oldToNewModuleMap);
    });
  } else {
    for (auto module : modulesToProcess)
      lowerModuleBody(module, oldToNewModuleMap);
  }

  // Finally delete all the old modules.
  for (auto oldNew : oldToNewModuleMap)
    oldNew.first->erase();

  // Now that the modules are moved over, remove the Circuit.  We pop the 'main
  // module' specified in the Circuit into an attribute on the top level module.
  getOperation()->setAttr(
      "firrtl.mainModule",
      StringAttr::get(circuit.getContext(), circuit.name()));
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
      assert(defineTrue && "didn't define anything");
      b.create<sv::IfDefProceduralOp>(
          guard, [&]() { emitString(define + defineTrue); });
    } else {
      b.create<sv::IfDefProceduralOp>(
          guard,
          [&]() {
            if (defineTrue)
              emitString(define + defineTrue);
          },
          [&]() { emitString(define + defineFalse); });
    }
  };

  emitString("// Standard header to adapt well known macros to our needs.");
  emitGuardedDefine("RANDOMIZE_GARBAGE_ASSIGN", "RANDOMIZE");
  emitGuardedDefine("RANDOMIZE_INVALID_ASSIGN", "RANDOMIZE");
  emitGuardedDefine("RANDOMIZE_REG_INIT", "RANDOMIZE");
  emitGuardedDefine("RANDOMIZE_MEM_INIT", "RANDOMIZE");
  emitGuardedDefine("RANDOM", nullptr, "RANDOM $random");

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
  emitGuardedDefine("INIT_RANDOM", nullptr, "INIT_RANDOM");

  emitString("\n// If using random initialization, you can also define "
             "RANDOMIZE_DELAY to\n// customize the delay used, otherwise 0.002 "
             "is used.");
  emitGuardedDefine("RANDOMIZE_DELAY", nullptr, "RANDOMIZE_DELAY 0.002");

  emitString("\n// Define INIT_RANDOM_PROLOG_ for use in our modules below.");

  b.create<sv::IfDefProceduralOp>(
      "RANDOMIZE",
      [&]() {
        emitGuardedDefine(
            "VERILATOR", "INIT_RANDOM_PROLOG_ `INIT_RANDOM",
            "INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end");
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

rtl::RTLModuleExternOp
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
  return builder.create<rtl::RTLModuleExternOp>(oldModule.getLoc(), nameAttr,
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
/// This can happen when there are no connects to the value.  The 'mergePoint'
/// location is where a 'rtl.merge' operation should be inserted if needed.
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
  if (!insertPoint->hasTrait<OpTrait::IsTerminator>()) {
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
  return builder.createOrFold<comb::MergeOp>(results);
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
    if (!portType.isInteger(0)) {
      if (port.isInOut())
        portType = rtl::InOutType::get(portType);
      operands.push_back(castFromFIRRTLType(wire, portType, builder));
    }

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

  void optimizeTemporaryWire(sv::WireOp wire);

  // Helpers.
  Value getPossiblyInoutLoweredValue(Value value);
  Value getLoweredValue(Value value);
  Value getLoweredAndExtendedValue(Value value, Type destType);
  Value getLoweredAndExtOrTruncValue(Value value, Type destType);
  LogicalResult setLowering(Value orig, Value result);
  template <typename ResultOpType, typename... CtorArgTypes>
  LogicalResult setLoweringTo(Operation *orig, CtorArgTypes... args);
  void emitRandomizePrologIfNeeded();
  void initializeRegister(Value reg, Value resetSignal);

  void runWithInsertionPointAtEndOfBlock(std::function<void(void)> fn,
                                         Region &region);
  void addToAlwaysBlock(Value clock, std::function<void(void)> fn);
  void addToAlwaysFFBlock(EventControl clockEdge, Value clock,
                          ::ResetType resetStyle, EventControl resetEdge,
                          Value reset, std::function<void(void)> body = {},
                          std::function<void(void)> resetBody = {});
  void addToAlwaysFFBlock(EventControl clockEdge, Value clock,
                          std::function<void(void)> body = {}) {
    addToAlwaysFFBlock(clockEdge, clock, ::ResetType(), EventControl(), Value(),
                       body, std::function<void(void)>());
  }
  void addToIfDefBlock(StringRef cond, std::function<void(void)> thenCtor,
                       std::function<void(void)> elseCtor = {});
  void addToInitialBlock(std::function<void(void)> body);
  void addToIfDefProceduralBlock(StringRef cond,
                                 std::function<void(void)> thenCtor,
                                 std::function<void(void)> elseCtor = {});
  void addIfProceduralBlock(Value cond, std::function<void(void)> thenCtor,
                            std::function<void(void)> elseCtor = {});

  // Create a temporary wire at the current insertion point, and try to
  // eliminate it later as part of lowering post processing.
  sv::WireOp createTmpWireOp(Type type, StringRef name) {
    auto result = builder->create<sv::WireOp>(type, name);
    tmpWiresToOptimize.push_back(result);
    return result;
  }

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
  LogicalResult visitExpr(RTLStructCastOp op);
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
    return lowerBinOpToVariadic<comb::AndOp>(op);
  }
  LogicalResult visitExpr(OrPrimOp op) {
    return lowerBinOpToVariadic<comb::OrOp>(op);
  }
  LogicalResult visitExpr(XorPrimOp op) {
    return lowerBinOpToVariadic<comb::XorOp>(op);
  }
  LogicalResult visitExpr(AddPrimOp op) {
    return lowerBinOpToVariadic<comb::AddOp>(op);
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

  LogicalResult visitExpr(SubPrimOp op) { return lowerBinOp<comb::SubOp>(op); }
  LogicalResult visitExpr(MulPrimOp op) {
    return lowerBinOpToVariadic<comb::MulOp>(op);
  }
  LogicalResult visitExpr(DivPrimOp op) {
    return lowerDivLikeOp<comb::DivSOp, comb::DivUOp>(op);
  }
  LogicalResult visitExpr(RemPrimOp op);

  // Other Operations
  LogicalResult visitExpr(BitsPrimOp op);
  LogicalResult visitExpr(InvalidValuePrimOp op);
  LogicalResult visitExpr(HeadPrimOp op);
  LogicalResult visitExpr(ShlPrimOp op);
  LogicalResult visitExpr(ShrPrimOp op);
  LogicalResult visitExpr(DShlPrimOp op) {
    return lowerDivLikeOp<comb::ShlOp, comb::ShlOp>(op);
  }
  LogicalResult visitExpr(DShrPrimOp op) {
    return lowerDivLikeOp<comb::ShrSOp, comb::ShrUOp>(op);
  }
  LogicalResult visitExpr(DShlwPrimOp op) {
    return lowerDivLikeOp<comb::ShrSOp, comb::ShrUOp>(op);
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

  // We auto-unique graph-level blocks to reduce the amount of generated code
  // and ensure that side effects are properly ordered in FIRRTL.
  llvm::SmallDenseMap<Value, sv::AlwaysOp> alwaysBlocks;

  using AlwaysFFKeyType = std::tuple<Block *, EventControl, Value, ::ResetType,
                                     EventControl, Value>;
  llvm::SmallDenseMap<AlwaysFFKeyType, sv::AlwaysFFOp> alwaysFFBlocks;
  llvm::SmallDenseMap<std::pair<Block *, Attribute>, sv::IfDefOp> ifdefBlocks;
  llvm::SmallDenseMap<Block *, sv::InitialOp> initialBlocks;

  /// This is a set of wires that get inserted as an artifact of the lowering
  /// process.  LowerToRTL should attempt to clean these up after lowering.
  SmallVector<sv::WireOp> tmpWiresToOptimize;

  /// This is true if we've emitted `INIT_RANDOM_PROLOG_ into an initial block
  /// in this module already.
  bool randomizePrologEmitted;
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::createLowerFIRRTLToRTLPass() {
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

  // Now that the IR is in a stable form, try to eliminate temporary wires
  // inserted by MemOp insertions.
  for (auto wire : tmpWiresToOptimize)
    optimizeTemporaryWire(wire);

  // Clear out the value mapping for next time, so we don't have dangling keys.
  valueMapping.clear();
  alwaysBlocks.clear();
  alwaysFFBlocks.clear();
  ifdefBlocks.clear();
  initialBlocks.clear();
  tmpWiresToOptimize.clear();
}

// Try to optimize out temporary wires introduced during lowering.
void FIRRTLLowering::optimizeTemporaryWire(sv::WireOp wire) {
  // Wires have inout type, so they'll have connects and read_inout operations
  // that work on them.  If anything unexpected is found then leave it alone.
  SmallVector<sv::ReadInOutOp> reads;
  sv::ConnectOp write;

  for (auto *user : wire->getUsers()) {
    if (auto read = dyn_cast<sv::ReadInOutOp>(user)) {
      reads.push_back(read);
      continue;
    }

    // Otherwise must be a connect, and we must not have seen a write yet.
    auto connect = dyn_cast<sv::ConnectOp>(user);
    if (!connect || write)
      return;
    write = connect;
  }

  // Must have found the write!
  if (!write)
    return;

  // If the write is happening at the model level then we don't have any
  // use-before-def checking to do, so we only handle that for now.
  if (!isa<rtl::RTLModuleOp>(write->getParentOp()))
    return;

  auto connected = write.src();

  // Ok, we can do this.  Replace all the reads with the connected value.
  for (auto read : reads) {
    read.replaceAllUsesWith(connected);
    read.erase();
  }
  // And remove the write and wire itself.
  write.erase();
  wire.erase();
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
    return builder->createOrFold<sv::ReadInOutOp>(result);

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
    return builder->createOrFold<comb::SExtOp>(resultType, result);

  auto zero = builder->create<rtl::ConstantOp>(APInt(destWidth - srcWidth, 0));
  return builder->createOrFold<comb::ConcatOp>(zero, result);
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
    return builder->createOrFold<comb::ExtractOp>(resultType, result, 0);
  } else {
    auto resultType = builder->getIntegerType(destWidth);

    // Extension follows the sign of the source value, not the destination.
    auto valueFIRType = value.getType().cast<FIRRTLType>().getPassiveType();
    if (valueFIRType.cast<IntType>().isSigned())
      return builder->createOrFold<comb::SExtOp>(resultType, result);

    auto zero =
        builder->create<rtl::ConstantOp>(APInt(destWidth - srcWidth, 0));
    return builder->createOrFold<comb::ConcatOp>(zero, result);
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

/// Switch the insertion point of the current builder to the end of the
/// specified block and run the closure.  This correctly handles the case where
/// the closure is null, but the caller needs to make sure the block exists.
void FIRRTLLowering::runWithInsertionPointAtEndOfBlock(
    std::function<void(void)> fn, Region &region) {
  if (!fn)
    return;

  auto oldIP = builder->saveInsertionPoint();

  // If this is the first logic injected into the specified block (e.g. an else
  // region), create the block and put an sv.yield.
  if (region.empty()) {
    // All SV dialect control flow operations use sv.yield.
    sv::IfOp::ensureTerminator(region, *builder,
                               region.getParentOp()->getLoc());
  }

  builder->setInsertionPoint(region.front().getTerminator());
  fn();
  builder->restoreInsertionPoint(oldIP);
}

void FIRRTLLowering::addToAlwaysBlock(Value clock,
                                      std::function<void(void)> fn) {
  // This isn't uniquing the parent region in.  This can be added as a key to
  // the alwaysBlocks set if needed.
  assert(isa<rtl::RTLModuleOp>(builder->getBlock()->getParentOp()) &&
         "only support inserting into the top level of a module so far");

  auto &op = alwaysBlocks[clock];
  if (op) {
    runWithInsertionPointAtEndOfBlock(fn, op.body());
  } else {
    op = builder->create<sv::AlwaysOp>(EventControl::AtPosEdge, clock, fn);
  }
}

void FIRRTLLowering::addToAlwaysFFBlock(EventControl clockEdge, Value clock,
                                        ::ResetType resetStyle,
                                        EventControl resetEdge, Value reset,
                                        std::function<void(void)> body,
                                        std::function<void(void)> resetBody) {
  auto &op = alwaysFFBlocks[std::make_tuple(
      builder->getBlock(), clockEdge, clock, resetStyle, resetEdge, reset)];
  if (op) {
    runWithInsertionPointAtEndOfBlock(body, op.bodyBlk());
    runWithInsertionPointAtEndOfBlock(resetBody, op.resetBlk());
  } else {
    if (reset) {
      op = builder->create<sv::AlwaysFFOp>(clockEdge, clock, resetStyle,
                                           resetEdge, reset, body, resetBody);
    } else {
      assert(!resetBody);
      op = builder->create<sv::AlwaysFFOp>(clockEdge, clock, body);
    }
  }
}

void FIRRTLLowering::addToIfDefBlock(StringRef cond,
                                     std::function<void(void)> thenCtor,
                                     std::function<void(void)> elseCtor) {
  auto condAttr = builder->getStringAttr(cond);
  auto &op = ifdefBlocks[{builder->getBlock(), condAttr}];
  if (op) {
    runWithInsertionPointAtEndOfBlock(thenCtor, op.thenRegion());
    runWithInsertionPointAtEndOfBlock(elseCtor, op.elseRegion());
  } else {
    op = builder->create<sv::IfDefOp>(condAttr, thenCtor, elseCtor);
  }
}

void FIRRTLLowering::addToInitialBlock(std::function<void(void)> body) {
  auto &op = initialBlocks[builder->getBlock()];
  if (op) {
    runWithInsertionPointAtEndOfBlock(body, op.body());
  } else {
    op = builder->create<sv::InitialOp>(body);
  }
}

void FIRRTLLowering::addToIfDefProceduralBlock(
    StringRef cond, std::function<void(void)> thenCtor,
    std::function<void(void)> elseCtor) {

  // Check to see if we already have an ifdef on this condition immediately
  // before the insertion point.  If so, extend it.
  auto insertIt = builder->getInsertionPoint();
  if (insertIt != builder->getBlock()->begin())
    if (auto ifdef = dyn_cast<sv::IfDefProceduralOp>(*--insertIt)) {
      if (ifdef.cond() == cond) {
        runWithInsertionPointAtEndOfBlock(thenCtor, ifdef.thenRegion());
        runWithInsertionPointAtEndOfBlock(elseCtor, ifdef.elseRegion());
        return;
      }
    }

  builder->create<sv::IfDefProceduralOp>(cond, thenCtor, elseCtor);
}

void FIRRTLLowering::addIfProceduralBlock(Value cond,
                                          std::function<void(void)> thenCtor,
                                          std::function<void(void)> elseCtor) {
  // Check to see if we already have an if on this condition immediately
  // before the insertion point.  If so, extend it.
  auto insertIt = builder->getInsertionPoint();
  if (insertIt != builder->getBlock()->begin())
    if (auto ifOp = dyn_cast<sv::IfOp>(*--insertIt)) {
      if (ifOp.cond() == cond) {
        runWithInsertionPointAtEndOfBlock(thenCtor, ifOp.thenRegion());
        runWithInsertionPointAtEndOfBlock(elseCtor, ifOp.elseRegion());
        return;
      }
    }

  builder->create<sv::IfOp>(cond, thenCtor, elseCtor);
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
      (void)setLowering(op->getResult(0), Value());
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
  // firrtl.mem lowering leaves invalid SubfieldOps.  Ignore these invalid ops.
  if (!op.input())
    return success();

  // Extracting a zero bit value from a struct is defined but doesn't do
  // anything.
  if (isZeroBitFIRRTLType(op->getResult(0).getType()))
    return setLowering(op, Value());

  auto resultType = lowerType(op->getResult(0).getType());
  Value value = getLoweredValue(op.input());
  assert(resultType && value && "subfield type lowering failed");

  return setLoweringTo<rtl::StructExtractOp>(op, resultType, value,
                                             op.fieldname());
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
  return setLoweringTo<sv::WireOp>(op, resultType, op.nameAttr());
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
      auto wire = builder->create<sv::WireOp>(operand.getType(), name);
      builder->create<sv::ConnectOp>(wire, operand);
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

void FIRRTLLowering::initializeRegister(Value reg, Value resetSignal) {
  // Construct and return a new reference to `RANDOM.
  auto randomVal = [&](Type type) {
    return builder->create<sv::TextualValueOp>(type, "`RANDOM");
  };

  // Randomly initialize everything in the register. If the register
  // is an aggregate type, then assign random values to all its
  // constituent ground types.
  // TODO: Extend this so it recursively initializes everything.
  auto randomInit = [&]() {
    auto type = reg.getType().dyn_cast<rtl::InOutType>().getElementType();
    TypeSwitch<Type>(type)
        .Case<rtl::UnpackedArrayType>([&](auto a) {
          for (size_t i = 0, e = a.getSize(); i != e; ++i) {
            auto iIdx = builder->create<rtl::ConstantOp>(APInt(log2(e + 1), i));
            auto arrayIndex = builder->create<sv::ArrayIndexInOutOp>(reg, iIdx);
            builder->create<sv::BPAssignOp>(arrayIndex,
                                            randomVal(a.getElementType()));
          }
        })
        .Default([&](auto a) {
          builder->create<sv::BPAssignOp>(reg, randomVal(a));
        });
  };

  // Emit the initializer expression for simulation that fills it with random
  // value.
  addToIfDefBlock("SYNTHESIS", std::function<void()>(), [&]() {
    addToInitialBlock([&]() {
      emitRandomizePrologIfNeeded();
      addToIfDefProceduralBlock("RANDOMIZE_REG_INIT", [&]() {
        if (resetSignal) {
          addIfProceduralBlock(resetSignal, {}, [&]() { randomInit(); });
        } else {
          randomInit();
        }
      });
    });
  });
}

LogicalResult FIRRTLLowering::visitDecl(RegOp op) {
  auto resultType = lowerType(op.result().getType());
  if (!resultType)
    return failure();
  if (resultType.isInteger(0))
    return setLowering(op, Value());

  auto regResult = builder->create<sv::RegOp>(resultType, op.nameAttr());
  (void)setLowering(op, regResult);

  initializeRegister(regResult, Value());

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
  // Reset values may be narrower than the register.  Extend appropriately.
  Value resetValue = getLoweredAndExtOrTruncValue(
      op.resetValue(), op.getType().cast<FIRRTLType>());

  if (!clockVal || !resetSignal || !resetValue)
    return failure();

  auto regResult = builder->create<sv::RegOp>(resultType, op.nameAttr());
  (void)setLowering(op, regResult);

  auto resetFn = [&]() {
    builder->create<sv::PAssignOp>(regResult, resetValue);
  };

  if (op.resetSignal().getType().isa<AsyncResetType>()) {
    addToAlwaysFFBlock(EventControl::AtPosEdge, clockVal,
                       ::ResetType::AsyncReset, EventControl::AtPosEdge,
                       resetSignal, std::function<void()>(), resetFn);
  } else { // sync reset
    addToAlwaysFFBlock(EventControl::AtPosEdge, clockVal,
                       ::ResetType::SyncReset, EventControl::AtPosEdge,
                       resetSignal, std::function<void()>(), resetFn);
  }

  initializeRegister(regResult, resetSignal);
  return success();
}

LogicalResult FIRRTLLowering::visitDecl(MemOp op) {
  StringRef memName = "mem";
  if (op.name().hasValue())
    memName = op.name().getValue();

  // TODO: Remove this restriction and preserve aggregates in
  // memories.
  if (op.getDataType().cast<FIRRTLType>().getPassiveType().isa<BundleType>())
    return op.emitOpError(
        "should have already been lowered from a ground type to an aggregate "
        "type using the LowerTypes pass. Use "
        "'firtool --enable-lower-types' or 'circt-opt "
        "--pass-pipeline='firrtl.circuit(firrtl-lower-types)' "
        "to run this.");

  uint64_t depth = op.depth();
  uint64_t readLatency = op.readLatency();
  uint64_t writeLatency = op.writeLatency();
  auto addrType = lowerType(op.getResult(0)
                                .getType()
                                .cast<FIRRTLType>()
                                .getPassiveType()
                                .cast<BundleType>()
                                .getElement("addr")
                                ->type);
  auto dataType = lowerType(op.getDataType());

  // A store of values associated with a delayed read.
  struct ReadPipeElement {
    Value en;
    Value addr;
    Value rd_en;
    Value rd_addr;
  };

  // A store of values associated with a delayed write.
  struct WritePipeElement {
    Value en;
    Value addr;
    Value mask;
    Value data;
    Value rd_en;
    Value rd_addr;
    Value rd_mask;
    Value rd_data;
  };

  // Create a register for the memory.
  Value reg =
      builder->create<sv::RegOp>(rtl::UnpackedArrayType::get(dataType, depth),
                                 builder->getStringAttr(memName.str()));

  // Some helpers.
  auto buildPAssign = [&](Value dest, Value value) {
    builder->create<sv::PAssignOp>(dest, value);
  };

  // Track pipeline registers that were added. These need to be
  // randomly initialized later.
  SmallVector<Value> pipeRegs;

  // Process each port in turn.
  SmallVector<std::pair<Identifier, MemOp::PortKind>> ports;
  op.getPorts(ports);
  assert(op.getNumResults() == ports.size());

  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    auto portName = ports[i].first.str();
    auto port = op.getResult(i);

    // Do not lower ports if they aren't used.
    if (port.use_empty())
      continue;

    auto portBundleType =
        port.getType().cast<FIRRTLType>().getPassiveType().cast<BundleType>();

    // Create wires for all the memory ports
    llvm::StringMap<Value> portWires;
    for (BundleType::BundleElement elt : portBundleType.getElements()) {
      auto fieldType = lowerType(elt.type);
      if (fieldType.isInteger(0)) {
        portWires.insert({elt.name.getValue(), Value()});
        continue;
      }
      auto name =
          (Twine(memName) + "_" + portName + "_" + elt.name.getValue()).str();
      auto fieldWire = createTmpWireOp(fieldType, name);
      portWires.insert({elt.name.getValue(), fieldWire});
    }

    // Now that we have the wires for each element, rewrite any subfields to
    // use them instead of the subfields.
    while (!port.use_empty()) {
      auto portField = cast<SubfieldOp>(*port.user_begin());
      portField->dropAllReferences();
      (void)setLowering(portField, portWires[portField.fieldname()]);
    }

    // Return the value corresponding to a port field.
    auto getPortFieldValue = [&](StringRef name) -> Value {
      return builder->create<sv::ReadInOutOp>(portWires[name]);
    };

    // Create an array register and keep track of it in pipeRegs.
    auto createArrayReg = [&](StringRef elementName, Type type,
                              uint64_t depth) -> Value {
      auto a = builder->create<sv::RegOp>(
          rtl::UnpackedArrayType::get(type, depth),
          builder->getStringAttr(memName.str() + "_" + portName +
                                 elementName.str()));
      pipeRegs.push_back(a);
      return a;
    };

    auto i1Type = IntegerType::get(builder->getContext(), 1);
    auto clk = getPortFieldValue("clk");

    // Loop over each element of the port
    switch (ports[i].second) {
    case MemOp::PortKind::ReadWrite:
      op.emitOpError("readwrite ports should be lowered into separate read and "
                     "write ports by previous passes");
      continue;
    case MemOp::PortKind::Read: {
      // Add delays for non-zero read latency
      SmallVector<ReadPipeElement> readPipe;
      auto rden = getPortFieldValue("en");
      auto rdaddr = getPortFieldValue("addr");
      readPipe.push_back({portWires["en"], portWires["addr"], rden, rdaddr});
      Value enReg, addrReg, enRegRd, addRegRd;
      for (size_t j = 0; j < readLatency; ++j) {
        if (j == 0) {
          enReg = createArrayReg("_en_pipe", i1Type, readLatency);
          addrReg = createArrayReg("_addr_pipe", addrType, readLatency);
        }
        auto jIdx =
            builder->create<rtl::ConstantOp>(APInt(log2(readLatency + 1), j));
        auto enJ = builder->create<sv::ArrayIndexInOutOp>(enReg, jIdx);
        auto addrJ = builder->create<sv::ArrayIndexInOutOp>(addrReg, jIdx);
        auto rdEnJ = builder->create<sv::ReadInOutOp>(enJ);
        auto rdAddrJ = builder->create<sv::ReadInOutOp>(addrJ);
        readPipe.push_back({enJ, addrJ, rdEnJ, rdAddrJ});
      }
      if (readLatency != 0) {
        addToAlwaysFFBlock(EventControl::AtPosEdge, clk, [&]() {
          for (size_t j = 1; j < readLatency + 1; ++j) {
            buildPAssign(readPipe[j].en, readPipe[j - 1].rd_en);
            addIfProceduralBlock(readPipe[j - 1].rd_en, [&]() {
              buildPAssign(readPipe[j].addr, readPipe[j - 1].rd_addr);
            });
          }
        });
      }
      Value addr = readPipe.back().rd_addr;
      Value value = builder->create<sv::ArrayIndexInOutOp>(reg, addr);
      value = builder->create<sv::ReadInOutOp>(value);

      // If we're masking, emit "addr < Depth ? mem[addr] : `RANDOM".
      if (llvm::isPowerOf2_64(depth)) {
        builder->create<sv::ConnectOp>(portWires["data"], value);
        continue;
      }

      addToIfDefBlock(
          "RANDOMIZE_GARBAGE_ASSIGN",
          [&]() {
            auto addrWidth = addr.getType().getIntOrFloatBitWidth();
            auto depthCst =
                builder->create<rtl::ConstantOp>(APInt(addrWidth, depth));
            auto cmp = builder->create<comb::ICmpOp>(ICmpPredicate::ult, addr,
                                                     depthCst);
            auto randomVal =
                builder->create<sv::TextualValueOp>(value.getType(), "`RANDOM");
            auto randomOrVal =
                builder->create<comb::MuxOp>(cmp, value, randomVal);
            builder->create<sv::ConnectOp>(portWires["data"], randomOrVal);
          },
          [&]() { builder->create<sv::ConnectOp>(portWires["data"], value); });
      continue;
    }
    case MemOp::PortKind::Write: {
      SmallVector<WritePipeElement> writePipe;
      auto rdEn = getPortFieldValue("en");
      auto rdAddr = getPortFieldValue("addr");
      auto rdMask = getPortFieldValue("mask");
      auto rdData = getPortFieldValue("data");
      writePipe.push_back({portWires["en"], portWires["addr"],
                           portWires["mask"], portWires["data"], rdEn, rdAddr,
                           rdMask, rdData});

      // Construct wripe pipe registers for non-unary write latency
      Value wRegEn, wRegAddr, wRegMask, wRegData;
      for (size_t j = 0; j < writeLatency - 1; ++j) {
        if (j == 0) {
          wRegEn = createArrayReg("_en_pipe", i1Type, writeLatency - 1);
          wRegAddr = createArrayReg("_addr_pipe", addrType, writeLatency - 1);
          wRegMask = createArrayReg("_mask_pipe", i1Type, writeLatency - 1);
          wRegData = createArrayReg("_data_pipe", dataType, writeLatency - 1);
        }
        auto jIdx =
            builder->create<rtl::ConstantOp>(APInt(log2(writeLatency), j));
        auto arEn = builder->create<sv::ArrayIndexInOutOp>(wRegEn, jIdx);
        auto arAddr = builder->create<sv::ArrayIndexInOutOp>(wRegAddr, jIdx);
        auto arMask = builder->create<sv::ArrayIndexInOutOp>(wRegMask, jIdx);
        auto arData = builder->create<sv::ArrayIndexInOutOp>(wRegData, jIdx);
        writePipe.push_back({arEn, arAddr, arMask, arData,
                             builder->create<sv::ReadInOutOp>(arEn),
                             builder->create<sv::ReadInOutOp>(arAddr),
                             builder->create<sv::ReadInOutOp>(arMask),
                             builder->create<sv::ReadInOutOp>(arData)});
      }

      // Build the write pipe for non-unary write latency
      if (writeLatency != 1) {
        addToAlwaysFFBlock(EventControl::AtPosEdge, clk, [&]() {
          for (size_t j = 1; j < writeLatency; ++j) {
            buildPAssign(writePipe[j].en, writePipe[j - 1].rd_en);
            addIfProceduralBlock(writePipe[j - 1].rd_en, [&]() {
              buildPAssign(writePipe[j].addr, writePipe[j - 1].rd_addr);
              buildPAssign(writePipe[j].mask, writePipe[j - 1].rd_mask);
              buildPAssign(writePipe[j].data, writePipe[j - 1].rd_data);
            });
          }
        });
      }

      // Attach the write port
      auto last = writePipe.back();
      auto enable = last.rd_en;
      auto cond = builder->create<comb::AndOp>(enable, last.rd_mask);
      auto slot = builder->create<sv::ArrayIndexInOutOp>(reg, last.rd_addr);
      auto rd_lastdata = last.rd_data;
      addToAlwaysFFBlock(EventControl::AtPosEdge, clk, [&]() {
        addIfProceduralBlock(cond, [&]() { buildPAssign(slot, rd_lastdata); });
      });
      continue;
    }
    }
  }

  // Emit the initializer expression for simulation that fills it with random
  // value.
  addToIfDefBlock("SYNTHESIS", {}, [&]() {
    addToInitialBlock([&]() {
      emitRandomizePrologIfNeeded();

      addToIfDefProceduralBlock("RANDOMIZE_MEM_INIT", [&]() {
        if (depth == 1) { // Don't emit a for loop for one element.
          auto type = sv::getInOutElementType(reg.getType());
          type = sv::getAnyRTLArrayElementType(type);
          auto randomVal = builder->create<sv::TextualValueOp>(type, "`RANDOM");
          auto zero = builder->create<rtl::ConstantOp>(APInt(1, 0));
          auto subscript = builder->create<sv::ArrayIndexInOutOp>(reg, zero);
          builder->create<sv::BPAssignOp>(subscript, randomVal);
        } else {
          assert(depth < (1ULL << 31) && "FIXME: Our initialization logic uses "
                                         "'integer' which doesn't support "
                                         "mems greater than 2^32");

          std::string action = "integer {{0}}_initvar;\n";
          action += "for ({{0}}_initvar = 0; {{0}}_initvar < " +
                    llvm::utostr(depth) + "; {{0}}_initvar = {{0}}_initvar+1)";
          action += "\n  {{0}}[{{0}}_initvar] = `RANDOM;";

          builder->create<sv::VerbatimOp>(action, reg);
        }
      });
    });
  });

  // Randomly initialize any pipeline registers that were created.
  for (auto pipeReg : pipeRegs)
    initializeRegister(pipeReg, Value());

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

LogicalResult FIRRTLLowering::visitExpr(RTLStructCastOp op) {
  // Conversions from rtl struct types to FIRRTL types are lowered as the input
  // operand.
  if (auto opStructType = op.getOperand().getType().dyn_cast<rtl::StructType>())
    return setLowering(op, op.getOperand());

  // Otherwise must be a conversion from FIRRTL bundle type to rtl struct type.
  auto result = getLoweredValue(op.getOperand());
  if (!result)
    return failure();

  // We lower firrtl.stdStructCast converting from a firrtl bundle to an rtl
  // struct type into the lowered operand.
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

  if (!result.getType().isa<rtl::InOutType>())
    return op.emitOpError("operand didn't lower to inout type correctly");

  op.replaceAllUsesWith(result);
  return success();
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
  return setLoweringTo<comb::ConcatOp>(op, zero, operand);
}

LogicalResult FIRRTLLowering::visitExpr(NotPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();
  // ~x  ---> x ^ 0xFF
  auto allOnes = builder->create<rtl::ConstantOp>(operand.getType(), -1);
  return setLoweringTo<comb::XorOp>(op, operand, allOnes);
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
  operand = builder->createOrFold<comb::SExtOp>(resultType, operand);

  auto zero = builder->create<rtl::ConstantOp>(resultType, 0);
  return setLoweringTo<comb::SubOp>(op, zero, operand);
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

  return setLoweringTo<comb::ParityOp>(op, builder->getIntegerType(1), operand);
}

LogicalResult FIRRTLLowering::visitExpr(AndRPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand) {
    return handleZeroBit(op.input(), [&]() {
      return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 1));
    });
  }

  // Lower AndR to == -1
  return setLoweringTo<comb::ICmpOp>(
      op, ICmpPredicate::eq, operand,
      builder->create<rtl::ConstantOp>(
          APInt(operand.getType().getIntOrFloatBitWidth(), -1)));
}

LogicalResult FIRRTLLowering::visitExpr(OrRPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand) {
    return handleZeroBit(op.input(), [&]() {
      return setLoweringTo<rtl::ConstantOp>(op, APInt(1, 0));
    });
    return failure();
  }

  // Lower OrR to != 0
  return setLoweringTo<comb::ICmpOp>(
      op, ICmpPredicate::ne, operand,
      builder->create<rtl::ConstantOp>(
          APInt(operand.getType().getIntOrFloatBitWidth(), 0)));
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
  return setLoweringTo<comb::ICmpOp>(
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
  return setLoweringTo<comb::ExtractOp>(op, lowerType(opType), result, 0);
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

  return setLoweringTo<comb::ConcatOp>(op, lhs, rhs);
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
    modInst = builder->createOrFold<comb::ModUOp>(lhs, rhs);
  } else {
    modInst = builder->createOrFold<comb::ModSOp>(lhs, rhs);
  }

  auto resultType = builder->getIntegerType(destWidth);
  return setLoweringTo<comb::ExtractOp>(op, resultType, modInst, 0);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitExpr(BitsPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  Type resultType = builder->getIntegerType(op.hi() - op.lo() + 1);
  return setLoweringTo<comb::ExtractOp>(op, resultType, input, op.lo());
}

LogicalResult FIRRTLLowering::visitExpr(InvalidValuePrimOp op) {
  auto resultTy = lowerType(op.getType());
  if (!resultTy)
    return failure();

  // Values of analog type always need to be lowered to something with inout
  // type.  We do that by lowering to a wire and return that.  As with the SFC,
  // we do not connect anything to this, because it is bidirectional.
  if (op.getType().isa<AnalogType>())
    return setLoweringTo<sv::WireOp>(op, resultTy, ".invalid_analog");

  // We lower invalid to 0.  TODO: the FIRRTL spec mentions something about
  // lowering it to a random value, we should see if this is what we need to
  // do.
  if (auto intType = resultTy.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 0) // Let the caller handle zero width values.
      return failure();
    return setLoweringTo<rtl::ConstantOp>(op, resultTy, 0);
  }

  // Invalid for bundles isn't supported.
  op.emitOpError("unsupported type");
  return failure();
}

LogicalResult FIRRTLLowering::visitExpr(HeadPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();
  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  if (op.amount() == 0)
    return setLowering(op, Value());
  Type resultType = builder->getIntegerType(op.amount());
  return setLoweringTo<comb::ExtractOp>(op, resultType, input,
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
  return setLoweringTo<comb::ConcatOp>(op, input, zero);
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
  return setLoweringTo<comb::ExtractOp>(op, resultType, input, shiftAmount);
}

LogicalResult FIRRTLLowering::visitExpr(TailPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  if (inWidth == op.amount())
    return setLowering(op, Value());
  Type resultType = builder->getIntegerType(inWidth - op.amount());
  return setLoweringTo<comb::ExtractOp>(op, resultType, input, 0);
}

LogicalResult FIRRTLLowering::visitExpr(MuxPrimOp op) {
  auto cond = getLoweredValue(op.sel());
  auto ifTrue = getLoweredAndExtendedValue(op.high(), op.getType());
  auto ifFalse = getLoweredAndExtendedValue(op.low(), op.getType());
  if (!cond || !ifTrue || !ifFalse)
    return failure();

  return setLoweringTo<comb::MuxOp>(op, ifTrue.getType(), cond, ifTrue,
                                    ifFalse);
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

    addToAlwaysFFBlock(EventControl::AtPosEdge, clockVal, [&]() {
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

    addToAlwaysFFBlock(EventControl::AtPosEdge, clockVal,
                       regResetOp.resetSignal().getType().isa<AsyncResetType>()
                           ? ::ResetType::AsyncReset
                           : ::ResetType::SyncReset,
                       EventControl::AtPosEdge, resetSignal, [&]() {
                         builder->create<sv::PAssignOp>(destVal, srcVal);
                       });
    return success();
  }

  builder->create<sv::ConnectOp>(destVal, srcVal);
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

    addToAlwaysFFBlock(EventControl::AtPosEdge, clockVal, [&]() {
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

    auto resetStyle = regResetOp.resetSignal().getType().isa<AsyncResetType>()
                          ? ::ResetType::AsyncReset
                          : ::ResetType::SyncReset;
    addToAlwaysFFBlock(EventControl::AtPosEdge, clockVal, resetStyle,
                       EventControl::AtPosEdge, resetSignal, [&]() {
                         builder->create<sv::PAssignOp>(destVal, srcVal);
                       });
    return success();
  }

  builder->create<sv::ConnectOp>(destVal, srcVal);
  return success();
}

// Printf is a macro op that lowers to an sv.ifdef.procedural, an sv.if,
// and an sv.fwrite all nested together.
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

  addToAlwaysBlock(clock, [&]() {
    // Emit an "#ifndef SYNTHESIS" guard into the always block.
    addToIfDefProceduralBlock("SYNTHESIS", std::function<void()>(), [&]() {
      // Emit an "sv.if '`PRINTF_COND_ & cond' into the #ifndef.
      Value ifCond =
          builder->create<sv::TextualValueOp>(cond.getType(), "`PRINTF_COND_");
      ifCond = builder->createOrFold<comb::AndOp>(ifCond, cond);

      addIfProceduralBlock(ifCond, [&]() {
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
  addToAlwaysBlock(clock, [&]() {
    // Emit an "#ifndef SYNTHESIS" guard into the always block.
    addToIfDefProceduralBlock("SYNTHESIS", std::function<void()>(), [&]() {
      // Emit an "sv.if '`STOP_COND_ & cond' into the #ifndef.
      Value ifCond =
          builder->create<sv::TextualValueOp>(cond.getType(), "`STOP_COND_");
      ifCond = builder->createOrFold<comb::AndOp>(ifCond, cond);
      addIfProceduralBlock(ifCond, [&]() {
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

  addToAlwaysBlock(clock, [&]() {
    addIfProceduralBlock(enable, [&]() {
      // Create BOpTy inside the always/if.
      builder->create<BOpTy>(predicate);
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

  addToIfDefBlock(
      "SYNTHESIS",
      // If we're doing synthesis, we emit an all-pairs assign complex.
      [&]() {
        SmallVector<Value, 4> values;
        for (size_t i = 0, e = inoutValues.size(); i != e; ++i)
          values.push_back(
              builder->createOrFold<sv::ReadInOutOp>(inoutValues[i]));

        for (size_t i1 = 0, e = inoutValues.size(); i1 != e; ++i1) {
          for (size_t i2 = 0; i2 != e; ++i2)
            if (i1 != i2)
              builder->create<sv::ConnectOp>(inoutValues[i1], values[i2]);
        }
      },
      // In the non-synthesis case, we emit a SystemVerilog alias statement.
      [&]() {
        builder->create<sv::IfDefOp>(
            "verilator",
            [&]() {
              builder->create<sv::VerbatimOp>(
                  "`error \"Verilator does not support alias and thus cannot "
                  "arbitrarily connect bidirectional wires and ports\"");
            },
            [&]() { builder->create<sv::AliasOp>(inoutValues); });
      });

  return success();
}
