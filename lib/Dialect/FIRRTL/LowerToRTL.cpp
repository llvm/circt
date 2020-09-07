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

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLPasses.h.inc"

namespace {
struct FIRRTLLowering : public LowerFIRRTLToRTLBase<FIRRTLLowering>,
                        public FIRRTLVisitor<FIRRTLLowering, LogicalResult> {

  void runOnOperation() override;

  // Helpers.
  Type lowerType(Type firrtlType);
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
  LogicalResult visitUnhandledOp(Operation *op) { return failure(); }
  LogicalResult visitInvalidOp(Operation *op) { return failure(); }

  // Unary Ops.
  LogicalResult lowerNoopCast(Operation *op);
  LogicalResult visitExpr(AsSIntPrimOp op) { return lowerNoopCast(op); }
  LogicalResult visitExpr(AsUIntPrimOp op) { return lowerNoopCast(op); }
  LogicalResult visitExpr(CatPrimOp op);
  LogicalResult visitExpr(PadPrimOp op);

  // Binary Ops.
  template <typename ResultOpType>
  LogicalResult lowerBinOp(Operation *op);
  template <typename ResultOpType>
  LogicalResult lowerBinOpToVariadic(Operation *op);

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
  LogicalResult visitExpr(SubPrimOp op) { return lowerBinOp<rtl::SubOp>(op); }

  // Other Operations
  LogicalResult visitExpr(BitsPrimOp op);
  LogicalResult visitExpr(HeadPrimOp op);
  LogicalResult visitExpr(ShlPrimOp op);
  LogicalResult visitExpr(ShrPrimOp op);
  LogicalResult visitExpr(TailPrimOp op);

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

/// Given a FIRRTL type, return the corresponding type for the RTL dialect.
/// This returns a null type if it cannot be lowered.
Type FIRRTLLowering::lowerType(Type type) {
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

  auto destFIRType = destType.cast<FIRRTLType>().getPassiveType();
  if (value.getType().cast<FIRRTLType>().getPassiveType() == destFIRType)
    return result;

  // We only know how to extend integer types with known width.
  auto destIntType = destFIRType.dyn_cast<IntType>();
  if (!destIntType || !destIntType.hasWidth())
    return {};

  auto destWidth = unsigned(destIntType.getWidthOrSentinel());
  auto srcWidth = result.getType().cast<IntegerType>().getWidth();
  if (srcWidth == destWidth)
    return result;

  assert(srcWidth < destWidth && "Only extensions");
  auto resultType = builder->getIntegerType(destWidth);

  auto loc = builder->getInsertionPoint()->getLoc();
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

// Pad is a noop or extension operation.
LogicalResult FIRRTLLowering::visitExpr(PadPrimOp op) {
  auto operand = getLoweredAndExtendedValue(op.input(), op.getType());
  if (!operand)
    return failure();
  return setLowering(op, operand);
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

LogicalResult FIRRTLLowering::visitExpr(CatPrimOp op) {
  auto lhs = getLoweredValue(op.lhs());
  auto rhs = getLoweredValue(op.rhs());
  if (!lhs || !rhs)
    return failure();

  return setLoweringTo<rtl::ConcatOp>(op, ValueRange({lhs, rhs}));
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitExpr(BitsPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  Type resultType = builder->getIntegerType(op.getHi() - op.getLo() + 1);
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input, op.getLo());
}

LogicalResult FIRRTLLowering::visitExpr(HeadPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  Type resultType = builder->getIntegerType(op.getAmount());
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input,
                                       inWidth - op.getAmount());
}

LogicalResult FIRRTLLowering::visitExpr(ShlPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  // Handle the degenerate case.
  if (op.getAmount() == 0)
    return setLowering(op, input);

  // TODO: We could keep track of zeros and implicitly CSE them.
  auto zero =
      builder->create<rtl::ConstantOp>(op.getLoc(), APInt(op.getAmount(), 0));
  return setLoweringTo<rtl::ConcatOp>(op, ValueRange({input, zero}));
}

LogicalResult FIRRTLLowering::visitExpr(ShrPrimOp op) {
  auto input = getLoweredValue(op.input());
  if (!input)
    return failure();

  // Handle the special degenerate cases.
  auto inWidth = input.getType().cast<IntegerType>().getWidth();
  auto shiftAmount = op.getAmount();
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
  Type resultType = builder->getIntegerType(inWidth - op.getAmount());
  return setLoweringTo<rtl::ExtractOp>(op, resultType, input, 0);
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitStmt(ConnectOp op) {
  auto dest = getLoweredValue(op.lhs());

  // The source can be a smaller integer, extend it as appropriate if so.
  Value src = getLoweredAndExtendedValue(op.rhs(), op.lhs().getType());

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
  auto always = builder->create<sv::AlwaysAtPosEdgeOp>(op.getLoc(), clock);

  // We're going to move insertion points.
  auto oldIP = &*builder->getInsertionPoint();

  // Emit an "#ifndef SYNTHESIS" guard into the always block.
  builder->setInsertionPointToStart(always.getBody());
  auto ifndef = builder->create<sv::IfDefOp>(op.getLoc(), "!SYNTHESIS");

  // Emit an "sv.if '`PRINTF_COND_ & cond' into the #ifndef.
  builder->setInsertionPointToStart(ifndef.getBodyBlock());
  auto cond = getLoweredValue(op.cond());
  Value ifCond = builder->create<sv::TextualValueOp>(
      op.getLoc(), cond.getType(), "`PRINTF_COND_");
  ifCond = builder->create<rtl::AndOp>(op.getLoc(), ValueRange{ifCond, cond},
                                       ArrayRef<NamedAttribute>{});
  auto svIf = builder->create<sv::IfOp>(op.getLoc(), ifCond);

  // Emit the sv.fwrite.
  builder->setInsertionPointToStart(svIf.getBodyBlock());
  SmallVector<Value, 4> operands;
  operands.reserve(op.operands().size());
  for (auto operand : op.operands())
    operands.push_back(getLoweredValue(operand));

  builder->create<sv::FWriteOp>(op.getLoc(), op.formatString(), operands);

  builder->setInsertionPoint(oldIP);
  return success();
}

// Stop lowers into a nested series of behavioral statements plus $fatal or
// $finish.
LogicalResult FIRRTLLowering::visitStmt(StopOp op) {
  // Emit this into an "sv.alwaysat_posedge" body.
  auto clock = getLoweredValue(op.clock());
  auto always = builder->create<sv::AlwaysAtPosEdgeOp>(op.getLoc(), clock);

  // We're going to move insertion points.
  auto oldIP = &*builder->getInsertionPoint();

  // Emit an "#ifndef SYNTHESIS" guard into the always block.
  builder->setInsertionPointToStart(always.getBody());
  auto ifndef = builder->create<sv::IfDefOp>(op.getLoc(), "!SYNTHESIS");

  // Emit an "sv.if '`STOP_COND_ & cond' into the #ifndef.
  builder->setInsertionPointToStart(ifndef.getBodyBlock());
  auto cond = getLoweredValue(op.cond());
  Value ifCond = builder->create<sv::TextualValueOp>(
      op.getLoc(), cond.getType(), "`STOP_COND_");
  ifCond = builder->create<rtl::AndOp>(op.getLoc(), ValueRange{ifCond, cond},
                                       ArrayRef<NamedAttribute>{});
  auto svIf = builder->create<sv::IfOp>(op.getLoc(), ifCond);

  // Emit the sv.fatal or sv.finish.
  builder->setInsertionPointToStart(svIf.getBodyBlock());

  if (op.exitCode().getLimitedValue())
    builder->create<sv::FatalOp>(op.getLoc());
  else
    builder->create<sv::FinishOp>(op.getLoc());

  builder->setInsertionPoint(oldIP);
  return success();
}