#include "../PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

using namespace circt;
using namespace hir;

Value emitReg(OpBuilder &builder, Type elementTy, Value input, Value tstart) {
  auto uLoc = builder.getUnknownLoc();
  auto c0 =
      builder.create<mlir::arith::ConstantOp>(uLoc, builder.getIndexAttr(0));
  auto ivReg = helper::emitRegisterAlloca(builder, elementTy);
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);

  builder.create<hir::StoreOp>(uLoc, input, ivReg, ArrayRef<Value>({c0}),
                               oneAttr, oneAttr, tstart, zeroAttr);
  return builder.create<hir::LoadOp>(uLoc, elementTy, ivReg,
                                     ArrayRef<Value>({c0}), zeroAttr, zeroAttr,
                                     tstart, zeroAttr);
}
std::pair<Value, Value> insertForOpEntryLogic(OpBuilder &builder,
                                              Value isFirstIter, Value lb,
                                              Value ub, Value step,
                                              Value tstartLoopBody) {
  auto uLoc = builder.getUnknownLoc();
  auto c0 =
      builder.create<mlir::arith::ConstantOp>(uLoc, builder.getIndexAttr(0));
  auto zeroBit = helper::materializeIntegerConstant(builder, 0, 1);
  auto lbWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, lb}));
  auto ubWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, ub}));
  auto stepWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, step}));

  auto ivReg = helper::emitRegisterAlloca(builder, lbWide.getType());
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);
  auto ivRegOut = builder.create<hir::LoadOp>(
      uLoc, lbWide.getType(), ivReg, ArrayRef<Value>({c0}), zeroAttr, zeroAttr,
      tstartLoopBody, zeroAttr);

  auto ivWide =
      builder.create<comb::MuxOp>(uLoc, isFirstIter, lbWide, ivRegOut);
  auto ivNext = builder.create<comb::AddOp>(uLoc, ivWide, stepWide);

  builder.create<hir::StoreOp>(uLoc, ivNext, ivReg, ArrayRef<Value>({c0}),
                               oneAttr, oneAttr, tstartLoopBody, zeroAttr);

  auto nextIterCondition = builder.create<comb::ICmpOp>(
      uLoc,
      ivWide.getType().isSignedInteger() ? comb::ICmpPredicate::slt
                                         : comb::ICmpPredicate::ult,
      ivNext, ubWide);
  // The ivNext of THIS iteration decides the condition for next iteration.
  // The condition var should be available at the next tstartLoopBody.
  // So we store it in a reg to be read in the next iteration.
  auto nextIterConditionSafe =
      emitReg(builder, builder.getI1Type(), nextIterCondition, tstartLoopBody);
  auto iv = builder.create<comb::ExtractOp>(uLoc, lb.getType(), ivWide,
                                            builder.getI32IntegerAttr(0));
  auto ivLatched = builder.create<hir::LatchOp>(uLoc, iv.getType(), iv,
                                                tstartLoopBody, zeroAttr);
  return std::make_pair(nextIterConditionSafe, ivLatched);
}
