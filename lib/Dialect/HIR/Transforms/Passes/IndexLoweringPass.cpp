//=======-- IndexLoweringPass.cpp - Lower IndexType to IntegerType.--------===//
//
// This file implements a pass to convert IndexType to IntegerType.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#define MACHINE_WORD_SIZE 64

using namespace circt;
using namespace hir;

namespace {
// This pass replaces all variables of IndexType that are created from
// IntegerTypes into IntegerType of MACHINE_WORD_SIZE recursively.
class IndexLoweringPass : public hir::IndexLoweringBase<IndexLoweringPass> {
public:
  void runOnOperation() override {
    hir::FuncOp funcOp = getOperation();
    if (failed(updateRegion(funcOp.getFuncBody())))
      signalPassFailure();
    for (auto *op : opsToErase)
      op->erase();
    return;
  }
  // Track all IndexType vars that are created using index_cast from an integer.
  // For every use of the variable that can be converted to IntegerType, replace
  // the index-var with the integer-var.
  // These uses include:
  // - DelayOp $input.
  // - BinOp $op1, $op2
  // - ForOp(no unroll) $lb, $ub, %step.
  // - IndexCastOp $in
  // SignExtendIOp(s32->_)/ZeroExtendIOp(u32/i32->_)/TruncateIOp
private:
  LogicalResult updateRegion(Region &region) {
    for (auto &operation : region.front()) {
      if (auto op = dyn_cast<mlir::IndexCastOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::ForOp>(operation)) {
        if (failed(updateOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::AddIOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::SubIOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::MulIOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::AddFOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::SubFOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<hir::MulFOp>(operation)) {
        if (failed(updateBinOp(op)))
          return failure();
      } else if (auto op = dyn_cast<mlir::ConstantOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<hir::LoadOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<hir::StoreOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<hir::YieldOp>(operation)) {
        continue;
      } else if (auto op = dyn_cast<hir::ReturnOp>(operation)) {
        continue;
      } else {
        return operation.emitError("Unsupported op in IndexLoweringPass.");
      }
    }
    return success();
  }

private:
  LogicalResult updateOp(mlir::IndexCastOp);
  LogicalResult updateOp(hir::ForOp);
  template <typename T>
  LogicalResult updateBinOp(T);

private:
  llvm::DenseMap<Value, Value> mapIndexToIntegerVar;
  SmallVector<Operation *> opsToErase;
};
} // end anonymous namespace

//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------

/// IndexType is signed. So we use sext for increasing the bitwidth.
Value castToMachineWordSizedInteger(mlir::ImplicitLocOpBuilder &builder,
                                    Value in) {
  Type inTy = in.getType().dyn_cast<IntegerType>();
  assert(inTy);
  assert(inTy.isSignlessInteger());
  if (inTy.getIntOrFloatBitWidth() == MACHINE_WORD_SIZE)
    return in;
  if (inTy.getIntOrFloatBitWidth() > MACHINE_WORD_SIZE)
    return builder.create<mlir::TruncateIOp>(
        builder.getIntegerType(MACHINE_WORD_SIZE), in);

  return builder.create<mlir::SignExtendIOp>(
      builder.getIntegerType(MACHINE_WORD_SIZE), in);
}

Value replaceIndexCastWithIntegerCast(mlir::ImplicitLocOpBuilder &builder,
                                      mlir::IndexCastOp op) {
  Value out;
  Type inTy = op.in().getType().dyn_cast<IntegerType>();
  assert(inTy);
  assert(inTy.isSignlessInteger());
  assert(op.getResult().getType().isSignlessInteger());
  if (inTy.getIntOrFloatBitWidth() ==
      op.getResult().getType().getIntOrFloatBitWidth())
    out = op.in();
  else if (inTy.getIntOrFloatBitWidth() >
           op.getResult().getType().getIntOrFloatBitWidth())
    out = builder.create<mlir::TruncateIOp>(op.getResult().getType(), op.in());
  else
    out =
        builder.create<mlir::SignExtendIOp>(op.getResult().getType(), op.in());
  return out;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// IndexLoweringPass method definitions.
//-----------------------------------------------------------------------------

/// If index_cast from an IntegerType to IndexType then note the original
/// integer-var in a map.

LogicalResult IndexLoweringPass::updateOp(mlir::IndexCastOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  Value res = op.getResult();
  Value in = op.in();
  // IndexType to IntegerType casts are allowed.
  if (in.getType().isa<IndexType>() && res.getType().isa<IntegerType>())
    return success();

  Value resI;
  // Both may become IntegerType because 'in' may have been converted from
  // IndexType to IntegerType. In that case replace the IndexCastOp.
  if (in.getType().isa<IntegerType>() && res.getType().isa<IntegerType>())
    resI = replaceIndexCastWithIntegerCast(builder, op);
  else {
    resI = castToMachineWordSizedInteger(builder, in);
    mapIndexToIntegerVar[res] = resI;
  }
  res.replaceAllUsesWith(resI);
  opsToErase.push_back(op);
  return success();
}

LogicalResult IndexLoweringPass::updateOp(hir::ForOp op) {
  // There is nothing to do for unroll loops.
  if (op->getAttr("unroll")) {
    if (failed(updateRegion(op.getLoopBody())))
      return failure();
    return success();
  }

  // There is nothing to do if the loop vars are already integer.
  if (op.getInductionVar().getType().isa<IntegerType>()) {
    if (failed(updateRegion(op.getLoopBody())))
      return failure();
    return success();
  }

  ImplicitLocOpBuilder builder(op.getLoc(), op);

  // Cast the bounds to IntegerType if they still IndexType (which means they
  // are constants).
  if (op.lb().getType().isIndex()) {
    Value newLb = builder
                      .create<mlir::IndexCastOp>(
                          builder.getIntegerType(MACHINE_WORD_SIZE), op.lb())
                      .getResult();
    op.lbMutable().assign(newLb);
  }
  if (op.ub().getType().isIndex()) {
    Value newUb = builder
                      .create<mlir::IndexCastOp>(
                          builder.getIntegerType(MACHINE_WORD_SIZE), op.ub())
                      .getResult();
    op.ubMutable().assign(newUb);
  }
  if (op.step().getType().isIndex()) {
    Value newStep =
        builder
            .create<mlir::IndexCastOp>(
                builder.getIntegerType(MACHINE_WORD_SIZE), op.step())
            .getResult();
    op.stepMutable().assign(newStep);
  }

  auto &body = op.getLoopBody().front();
  Value newInductionVar = body.insertArgument(
      (unsigned)0, builder.getIntegerType(MACHINE_WORD_SIZE));
  body.getArgument(1).replaceAllUsesWith(newInductionVar);
  body.eraseArgument(1);
  if (failed(updateRegion(op.getLoopBody())))
    return failure();
  return success();
}

template <typename T>
LogicalResult IndexLoweringPass::updateBinOp(T op) {
  // LogicalResult IndexLoweringPass::updateBinOp(hir::AddIOp op) {
  if (op.op1().getType().template isa<IntegerType>() &&
      op.op2().getType().template isa<IndexType>()) {
    ImplicitLocOpBuilder builder(op.getLoc(), op);
    Value rhs = builder
                    .create<mlir::IndexCastOp>(
                        builder.getIntegerType(MACHINE_WORD_SIZE), op.op2())
                    .getResult();
    Value newRes = builder.create<hir::AddIOp>(
        builder.getIntegerType(MACHINE_WORD_SIZE), op.op1(), rhs,
        op.delayAttr(), op.tstart(), op.offsetAttr());
    op.res().replaceAllUsesWith(newRes);
    opsToErase.push_back(op);
  } else if (op.op1().getType().template isa<IndexType>() &&
             op.op2().getType().template isa<IntegerType>()) {
    ImplicitLocOpBuilder builder(op.getLoc(), op);
    Value lhs = builder
                    .create<mlir::IndexCastOp>(
                        builder.getIntegerType(MACHINE_WORD_SIZE), op.op1())
                    .getResult();
    Value newRes = builder.create<hir::AddIOp>(
        builder.getIntegerType(MACHINE_WORD_SIZE), lhs, op.op2(),
        op.delayAttr(), op.tstart(), op.offsetAttr());
    op.res().replaceAllUsesWith(newRes);
    opsToErase.push_back(op);
  }
  return success();
}

//-----------------------------------------------------------------------------
namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createIndexLoweringPass() {
  return std::make_unique<IndexLoweringPass>();
}
} // namespace hir
} // namespace circt
