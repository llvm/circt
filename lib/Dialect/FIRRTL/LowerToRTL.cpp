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
  void lowerModule(FModuleOp oldModule, Block *topLevelModule);
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

  CircuitOp circuit;
  for (auto &op : *moduleBody) {
    if ((circuit = dyn_cast<CircuitOp>(&op)))
      break;
  }

  if (!circuit)
    return;

  // TODO: We should drop the CircuitOp as well.
  auto *circuitBody = circuit.getBody();

  // Iterate through each operation in the circuit body, transforming any
  // FModule's we come across.
  for (auto opIt = circuitBody->getOperations().begin(),
            opEnd = circuitBody->getOperations().end();
       opIt != opEnd;) {
    // Step through the operations carefully to avoid invalidating the iterator.
    if (auto module = dyn_cast<FModuleOp>(*opIt++))
      lowerModule(module, moduleBody);

    // TODO: Lower extmodule.
  }

  // Now that the modules are moved over, remove the Circuit.  We pop the 'main
  // module' specified in the Circuit into an attribute on the top level module.
  getOperation().setAttr("firrtl.mainModule",
                         StringAttr::get(circuit.name(), circuit.getContext()));
  circuit.erase();
}

/// Run on each module, transforming it from an firrtl.module into an
/// rtl.module, then deleting the old one.
void FIRRTLModuleLowering::lowerModule(FModuleOp oldModule,
                                       Block *topLevelModule) {
  OpBuilder builder(topLevelModule->getTerminator());

  // Map the ports over, lowering their types as we go.
  SmallVector<ModulePortInfo, 8> firrtlPorts;
  oldModule.getPortInfo(firrtlPorts);

  SmallVector<rtl::RTLModulePortInfo, 8> ports;
  ports.reserve(firrtlPorts.size());
  for (auto firrtlPort : firrtlPorts) {
    rtl::RTLModulePortInfo rtlPort;

    rtlPort.name = firrtlPort.first;
    auto firrtlType = firrtlPort.second;
    rtlPort.type = lowerType(firrtlType);

    // We can't lower all types, so make sure to cleanly reject them.
    if (!rtlPort.type) {
      oldModule.emitError("cannot lower this port type to RTL");
      return;
    }

    // Figure out the direction of the port.
    const char *direction;
    if (firrtlType.isa<FlipType>()) {
      // If the top-level type in FIRRTL is a flip, then this is an output.
      assert(firrtlType.cast<FlipType>().getElementType().isPassiveType() &&
             "Flip types should be completely passive internally");
      direction = "out";
    } else if (firrtlType.cast<FIRRTLType>().isPassiveType()) {
      direction = "input";
    } else {
      // This isn't currently expressible in low-firrtl, due to bundle types
      // being lowered.
      direction = "inout";
    }
    rtlPort.direction = builder.getStringAttr(direction);
    ports.push_back(rtlPort);
  }

  // Build the new rtl.module op.
  auto nameAttr = builder.getStringAttr(oldModule.getName());
  auto newModule =
      builder.create<rtl::RTLModuleOp>(oldModule.getLoc(), nameAttr, ports);

  OpBuilder bodyBuilder(newModule.body());

  // Insert argument casts, and revector users in the old body to use them.
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    auto oldArg = oldModule.body().getArgument(i);
    auto newArg = newModule.body().getArgument(i);

    // Cast the argument to the old type, reintroducing sign information in the
    // rtl.module body.
    Value castedNewArg = bodyBuilder.create<StdIntCast>(
        oldModule.getLoc(),
        oldArg.getType().cast<FIRRTLType>().getPassiveType(), newArg);

    if (oldArg.getType() != castedNewArg.getType())
      castedNewArg = bodyBuilder.create<AsNonPassivePrimOp>(
          oldModule.getLoc(), oldArg.getType(), castedNewArg);

    // Switch all uses of the old operands to the new ones.
    oldArg.replaceAllUsesWith(castedNewArg);
  }

  // Splice the body over, don't move the old terminator over though.
  auto &oldBlockInstList = oldModule.getBodyBlock()->getOperations();
  auto &newBlockInstList = newModule.getBodyBlock()->getOperations();
  newBlockInstList.splice(std::prev(newBlockInstList.end()), oldBlockInstList,
                          oldBlockInstList.begin(),
                          std::prev(oldBlockInstList.end()));

  // TODO: Update instances.

  // Finally, remove the firrtl.module operation.
  oldModule.getOperation()->erase();
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

private:
  /// This builder is set to the right location for each visit call.
  OpBuilder *builder = nullptr;

  /// Each value lowered (e.g. operation result) is kept track in this map.  The
  /// key should have a FIRRTL type, the result will have an RTL dialect type.
  DenseMap<Value, Value> valueMapping;
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
  if (!firType.isPassiveType()) {
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
    Value mapped = builder->create<StdIntCast>(op->getLoc(),
                                               origValue.getType(), it->second);
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
