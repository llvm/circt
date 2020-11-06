//===- LowerToRTL.cpp - Lower FIRRTL -> RTL dialect -----------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/Visitors.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/SV/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
using namespace circt;
using namespace firrtl;

/// Return the type of the specified value, casted to the template type.
template <typename T = FIRRTLType>
static T getTypeOf(Value v) {
  return v.getType().cast<T>();
}

#define GEN_PASS_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLPasses.h.inc"

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
    return IntegerType::get(width, type.getContext());

  return {};
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
    if (firrtlPort.isOutput()) {
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

  // Build the new rtl.module op.
  OpBuilder builder(topLevelModule->getTerminator());
  auto nameAttr = builder.getStringAttr(oldModule.getName());
  return builder.create<rtl::RTLExternModuleOp>(oldModule.getLoc(), nameAttr,
                                                ports);
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

/// Cast from a standard type to a FIRRTL type, potentially with a flip.
static Value castToFIRRTLType(Location loc, Value val, Type type,
                              OpBuilder &builder) {
  val = builder.create<StdIntCast>(
      loc, type.cast<FIRRTLType>().getPassiveType(), val);

  // Handle the flip type if needed.
  if (type != val.getType())
    val = builder.create<AsNonPassivePrimOp>(loc, type, val);
  return val;
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

  OpBuilder bodyBuilder(newModule.body());

  // Insert argument casts, and re-vector users in the old body to use them.
  SmallVector<rtl::ModulePortInfo, 8> ports;
  newModule.getPortInfo(ports);

  size_t nextNewArg = 0;
  size_t firrtlArg = 0;
  SmallVector<Value, 4> outputs;
  for (auto &port : ports) {
    // Inputs and outputs are both modeled as arguments in the FIRRTL level.
    auto oldArg = oldModule.body().getArgument(firrtlArg++);

    Value newArg;
    if (!port.isOutput()) {
      // Inputs and InOuts are modeled as arguments in the result, so we can
      // just map them over.
      newArg = newModule.body().getArgument(nextNewArg++);
    } else {
      // Outputs need a temporary wire so they can be assign'd to, which we then
      // return.
      newArg = bodyBuilder.create<rtl::WireOp>(oldModule.getLoc(), port.type,
                                               /*name=*/StringAttr());
      outputs.push_back(newArg);
    }

    // Cast the argument to the old type, reintroducing sign information in
    // the rtl.module body.
    newArg = castToFIRRTLType(oldModule.getLoc(), newArg, oldArg.getType(),
                              bodyBuilder);

    // Switch all uses of the old operands to the new ones.
    oldArg.replaceAllUsesWith(newArg);
  }

  // Update the rtl.output terminator with the list of outputs we have.
  newModule.getBodyBlock()->getTerminator()->setOperands(outputs);

  // Finally splice the body over, don't move the old terminator over though.
  auto &oldBlockInstList = oldModule.getBodyBlock()->getOperations();
  auto &newBlockInstList = newModule.getBodyBlock()->getOperations();
  newBlockInstList.splice(std::prev(newBlockInstList.end()), oldBlockInstList,
                          oldBlockInstList.begin(),
                          std::prev(oldBlockInstList.end()));
}

/// lowerInstance - Lower a firrtl.instance operation to an rtl.instance
/// operation.  This is a bit more involved than it sounds because we have to
/// clean up the subfield operations that are hanging off of it, handle the
/// differences between FIRRTL and RTL approaches to module parameterization and
/// output ports.
///
/// On success, this returns with the firrtl.instance op having no users,
/// letting the caller erase it.
void FIRRTLModuleLowering::lowerInstance(
    InstanceOp oldInstance,
    DenseMap<Operation *, Operation *> &oldToNewModuleMap) {

  auto *oldModule = oldInstance.getReferencedModule();
  auto newModule =
      dyn_cast_or_null<rtl::RTLModuleOp>(oldToNewModuleMap[oldModule]);
  if (!newModule)
    return;

  // Decode information about the input and output ports on the referenced
  // module.
  SmallVector<ModulePortInfo, 8> portInfo;
  getModulePortInfo(oldModule, portInfo);

  OpBuilder builder(oldInstance);
  SmallVector<Type, 8> resultTypes;
  SmallVector<Value, 8> operands;
  for (auto &port : portInfo) {
    auto portType = lowerType(port.type);
    if (!portType)
      return;

    if (port.isOutput())
      // outputs become results.
      resultTypes.push_back(portType);
    else {
      assert(port.isInput() &&
             "TODO: Handle inout ports when we can lower mid FIRRTL bundles");
      // Create a wire for each input/inout operand, so there is something to
      // connect to.
      auto name = builder.getStringAttr(port.getName().str() + ".wire");
      operands.push_back(
          builder.create<rtl::WireOp>(oldInstance.getLoc(), portType, name));
    }
  }

  // Create the new rtl.instance operation.
  StringRef instanceName;
  if (oldInstance.name().hasValue())
    instanceName = oldInstance.name().getValue();

  auto newInst = builder.create<rtl::InstanceOp>(oldInstance.getLoc(),
                                                 resultTypes, instanceName,
                                                 newModule.getName(), operands);

  // Now that we have the new rtl.instance, we need to remap all of the users
  // of the firrtl.instance.  Burn through them connecting them up the right
  // way to the new world.
  while (!oldInstance.use_empty()) {
    // The only operation that can use the instance is a subfield operation.
    auto *user = *Value(oldInstance).user_begin();
    auto subfield = dyn_cast<SubfieldOp>(user);
    if (!subfield) {
      user->emitOpError("unexpected user of firrtl.instance operation");
      return;
    }

    // Figure out which inputNo or resultNo this is.
    size_t inputNo = 0, resultNo = 0;
    for (auto &port : portInfo) {
      if (port.name == subfield.fieldnameAttr())
        break;

      if (port.isOutput())
        ++resultNo;
      else
        ++inputNo;
    }

    // If this subfield has flip type, then it is an input.  Otherwise it is a
    // result.
    Value resultVal;
    if (subfield.getType().isa<FlipType>()) {
      // Use the wire we created as the input.
      resultVal = operands[inputNo];
    } else {
      resultVal = newInst.getResult(resultNo);
    }

    // Cast the value to the right signedness and flippedness.
    auto val = castToFIRRTLType(oldInstance.getLoc(), resultVal,
                                subfield.getType(), builder);
    subfield.replaceAllUsesWith(val);
    subfield.erase();
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
  Value getLoweredValue(Value operand);
  Value getLoweredAndExtendedValue(Value operand, Type destType);
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
  LogicalResult visitExpr(CvtPrimOp op);
  LogicalResult visitExpr(NotPrimOp op);
  LogicalResult visitExpr(NegPrimOp op);
  LogicalResult visitExpr(PadPrimOp op);
  LogicalResult visitExpr(XorRPrimOp op);
  LogicalResult visitExpr(AndRPrimOp op);
  LogicalResult visitExpr(OrRPrimOp op);

  // Binary Ops.

  template <typename ResultOpType>
  LogicalResult lowerBinOp(Operation *op);
  template <typename ResultOpType>
  LogicalResult lowerBinOpToVariadic(Operation *op);
  LogicalResult lowerCmpOp(Operation *op, ICmpPredicate signedOp,
                           ICmpPredicate unsignedOp);

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
  LogicalResult visitExpr(DivPrimOp op) { return lowerBinOp<rtl::DivOp>(op); }
  LogicalResult visitExpr(RemPrimOp op);

  // Other Operations
  LogicalResult visitExpr(BitsPrimOp op);
  LogicalResult visitExpr(HeadPrimOp op);
  LogicalResult visitExpr(ShlPrimOp op);
  LogicalResult visitExpr(ShrPrimOp op);
  LogicalResult visitExpr(TailPrimOp op);
  LogicalResult visitExpr(MuxPrimOp op);

  // Statements
  LogicalResult visitStmt(ConnectOp op);
  LogicalResult visitStmt(PrintFOp op);
  LogicalResult visitStmt(StopOp op);
  LogicalResult visitStmt(AssertOp op);
  LogicalResult visitStmt(AssumeOp op);
  LogicalResult visitStmt(CoverOp op);

private:
  /// This builder is set to the right location for each visit call.
  OpBuilder *builder = nullptr;

  /// Each value lowered (e.g. operation result) is kept track in this map.  The
  /// key should have a FIRRTL type, the result will have an RTL dialect type.
  DenseMap<Value, Value> valueMapping;

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
  template <typename A, typename B>
  LogicalResult lowerVerificationStatement(A op) {
    auto clock = getLoweredValue(op.clock());
    auto enable = getLoweredValue(op.enable());
    auto predicate = getLoweredValue(op.predicate());
    if (!clock || !enable || !predicate)
      return failure();

    auto always = builder->create<sv::AlwaysAtPosEdgeOp>(op.getLoc(), clock);
    auto oldIP = &*builder->getInsertionPoint();
    builder->setInsertionPointToStart(always.getBody());

    auto svIf = builder->create<sv::IfOp>(op.getLoc(), enable);
    builder->setInsertionPointToStart(svIf.getBodyBlock());
    builder->create<B>(op.getLoc(), predicate);
    builder->setInsertionPoint(oldIP);

    return success();
  }
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
  OpBuilder theBuilder(&getContext());
  builder = &theBuilder;
  for (auto &op : body->getOperations()) {
    builder->setInsertionPoint(&op);
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
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return the lowered value corresponding to the specified original value.
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredValue(Value value) {
  // All FIRRTL dialect values have FIRRTL types, so if we see something else
  // mixed in, it must be something we can't lower.  Just return it directly.
  auto firType = value.getType().dyn_cast<FIRRTLType>();
  if (!firType)
    return value;

  // If we lowered this value, then return the lowered value.
  auto it = valueMapping.find(value);
  if (it != valueMapping.end())
    return it->second;

  // Otherwise, we need to introduce a cast to the right FIRRTL type.
  auto resultType = lowerType(firType);
  if (!resultType)
    return value;

  if (!resultType.isa<IntegerType>())
    return {};

  // Cast FIRRTL -> standard type.
  auto loc = builder->getInsertionPoint()->getLoc();
  if (!firType.isPassive()) {
    value =
        builder->create<AsPassivePrimOp>(loc, firType.getPassiveType(), value);
  }

  return builder->create<StdIntCast>(loc, resultType, value);
}

/// Return the lowered value corresponding to the specified original value and
/// then extend it to match the width of destType if needed.
///
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredAndExtendedValue(Value value, Type destType) {
  assert(value.getType().isa<FIRRTLType>() && destType.isa<FIRRTLType>() &&
         "input/output value should be FIRRTL");

  auto result = getLoweredValue(value);
  if (!result)
    return {};

  auto destFIRType = destType.cast<FIRRTLType>();
  if (value.getType().cast<FIRRTLType>() == destFIRType)
    return result;

  // We only know how to extend integer types with known width.
  auto destIntType = destFIRType.dyn_cast<IntType>();
  if (!destIntType || !destIntType.hasWidth())
    return {};

  auto destWidth = unsigned(destIntType.getWidthOrSentinel());
  auto srcWidth = result.getType().cast<IntegerType>().getWidth();
  if (srcWidth == destWidth)
    return result;

  auto loc = builder->getInsertionPoint()->getLoc();
  if (srcWidth > destWidth) {
    emitError(loc) << "operand should not be a truncation";
    return {};
  }

  auto resultType = builder->getIntegerType(destWidth);

  if (destIntType.isSigned())
    return builder->create<rtl::SExtOp>(loc, resultType, result);

  return builder->create<rtl::ZExtOp>(loc, resultType, result);
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
  auto result = builder->create<ResultOpType>(orig->getLoc(), args...);
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
    auto wire =
        builder->create<rtl::WireOp>(op.getLoc(), operand.getType(), name);
    builder->create<rtl::ConnectOp>(op.getLoc(), wire, operand);
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
    Value mapped = castToFIRRTLType(op->getLoc(), it->second,
                                    origValue.getType(), *builder);
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

LogicalResult FIRRTLLowering::visitExpr(CvtPrimOp op) {
  auto operand = getLoweredValue(op.getOperand());
  if (!operand)
    return failure();

  // Signed to signed is a noop.
  if (getTypeOf<IntType>(op.getOperand()).isSigned())
    setLowering(op, operand);

  // Otherwise prepend a zero bit.
  auto zero = builder->create<rtl::ConstantOp>(op.getLoc(), APInt(1, 0));
  return setLoweringTo<rtl::ConcatOp>(op, ValueRange({zero, operand}));
}

LogicalResult FIRRTLLowering::visitExpr(NotPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();
  // ~x  ---> x ^ 0xFF
  auto type = operand.getType().cast<IntegerType>();
  auto allOnes = builder->create<rtl::ConstantOp>(op.getLoc(), -1, type);
  return setLoweringTo<rtl::XorOp>(op, ValueRange({operand, allOnes}),
                                   ArrayRef<NamedAttribute>{});
}

LogicalResult FIRRTLLowering::visitExpr(NegPrimOp op) {
  auto operand = getLoweredValue(op.input());
  if (!operand)
    return failure();

  // FIRRTL negate always adds a bit.
  // -x  ---> 0-sext(x) or 0-zext(x)
  auto resultType = lowerType(op.getType());
  if (getTypeOf<IntType>(op.input()).isSigned())
    operand = builder->create<rtl::SExtOp>(op.getLoc(), resultType, operand);
  else
    operand = builder->create<rtl::ZExtOp>(op.getLoc(), resultType, operand);

  auto zero = builder->create<rtl::ConstantOp>(op.getLoc(), 0,
                                               resultType.cast<IntegerType>());
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
template <typename ResultOpType>
LogicalResult FIRRTLLowering::lowerBinOp(Operation *op) {
  // Extend the two operands to match the destination type.
  auto resultType = op->getResult(0).getType();
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);
  if (!lhs || !rhs)
    return failure();

  // Emit the result operation.
  return setLoweringTo<ResultOpType>(op, lhs, rhs);
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

LogicalResult FIRRTLLowering::visitExpr(CatPrimOp op) {
  auto lhs = getLoweredValue(op.lhs());
  auto rhs = getLoweredValue(op.rhs());
  if (!lhs || !rhs)
    return failure();

  return setLoweringTo<rtl::ConcatOp>(op, ValueRange({lhs, rhs}));
}

LogicalResult FIRRTLLowering::visitExpr(RemPrimOp op) {
  // FIRRTL has the width of (a % b) = Min(W(a), W(b)) so we need to truncate
  // operands to the minimum width before doing the mod, not extend them.
  auto lhs = getLoweredValue(op.lhs());
  auto rhs = getLoweredValue(op.rhs());
  if (!lhs || !rhs)
    return failure();

  auto resultFirType = op.getType().cast<IntType>();
  if (!resultFirType.hasWidth())
    return failure();
  auto destWidth = unsigned(resultFirType.getWidthOrSentinel());
  auto resultType = builder->getIntegerType(destWidth);

  // Truncate either operand if required.
  if (lhs.getType().cast<IntegerType>().getWidth() != destWidth)
    lhs = builder->create<rtl::ExtractOp>(op.getLoc(), resultType, lhs, 0);
  if (rhs.getType().cast<IntegerType>().getWidth() != destWidth)
    rhs = builder->create<rtl::ExtractOp>(op.getLoc(), resultType, rhs, 0);

  return setLoweringTo<rtl::ModOp>(op, ValueRange({lhs, rhs}));
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
  auto zero =
      builder->create<rtl::ConstantOp>(op.getLoc(), APInt(op.amount(), 0));
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

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitStmt(ConnectOp op) {
  auto dest = getLoweredValue(op.lhs());

  // The source can be a smaller integer, extend it as appropriate if so.
  auto lhsType = op.lhs().getType().cast<FIRRTLType>().getPassiveType();
  Value src = getLoweredAndExtendedValue(op.rhs(), lhsType);

  if (!dest || !src)
    return failure();

  builder->create<rtl::ConnectOp>(op.getLoc(), dest, src);
  return success();
}

// Printf is a macro op that lowers to an sv.ifdef, an sv.if, and an sv.fwrite
// all nested together.
LogicalResult FIRRTLLowering::visitStmt(PrintFOp op) {
  // Emit this into an "sv.alwaysat_posedge" body.
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

  auto always = builder->create<sv::AlwaysAtPosEdgeOp>(op.getLoc(), clock);

  // We're going to move insertion points.
  auto oldIP = &*builder->getInsertionPoint();

  // Emit an "#ifndef SYNTHESIS" guard into the always block.
  builder->setInsertionPointToStart(always.getBody());
  auto ifndef = builder->create<sv::IfDefOp>(op.getLoc(), "!SYNTHESIS");

  // Emit an "sv.if '`PRINTF_COND_ & cond' into the #ifndef.
  builder->setInsertionPointToStart(ifndef.getBodyBlock());
  Value ifCond = builder->create<sv::TextualValueOp>(
      op.getLoc(), cond.getType(), "`PRINTF_COND_");
  ifCond = builder->create<rtl::AndOp>(op.getLoc(), ValueRange{ifCond, cond},
                                       ArrayRef<NamedAttribute>{});
  auto svIf = builder->create<sv::IfOp>(op.getLoc(), ifCond);

  // Emit the sv.fwrite.
  builder->setInsertionPointToStart(svIf.getBodyBlock());
  builder->create<sv::FWriteOp>(op.getLoc(), op.formatString(), operands);

  builder->setInsertionPoint(oldIP);
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

  auto always = builder->create<sv::AlwaysAtPosEdgeOp>(op.getLoc(), clock);

  // We're going to move insertion points.
  auto oldIP = &*builder->getInsertionPoint();

  // Emit an "#ifndef SYNTHESIS" guard into the always block.
  builder->setInsertionPointToStart(always.getBody());
  auto ifndef = builder->create<sv::IfDefOp>(op.getLoc(), "!SYNTHESIS");

  // Emit an "sv.if '`STOP_COND_ & cond' into the #ifndef.
  builder->setInsertionPointToStart(ifndef.getBodyBlock());
  Value ifCond = builder->create<sv::TextualValueOp>(
      op.getLoc(), cond.getType(), "`STOP_COND_");
  ifCond = builder->create<rtl::AndOp>(op.getLoc(), ValueRange{ifCond, cond},
                                       ArrayRef<NamedAttribute>{});
  auto svIf = builder->create<sv::IfOp>(op.getLoc(), ifCond);

  // Emit the sv.fatal or sv.finish.
  builder->setInsertionPointToStart(svIf.getBodyBlock());

  if (op.exitCode())
    builder->create<sv::FatalOp>(op.getLoc());
  else
    builder->create<sv::FinishOp>(op.getLoc());

  builder->setInsertionPoint(oldIP);
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
