#include "../PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"

using namespace circt;
using namespace hir;

Value hwSelect(OpBuilder &builder, Value sel, Value trueVal, Value falseVal) {
  auto muxInputs = builder.create<hw::ArrayCreateOp>(
      builder.getUnknownLoc(), ArrayRef<Value>({trueVal, falseVal}));
  return builder.create<hw::ArrayGetOp>(builder.getUnknownLoc(), muxInputs,
                                        sel);
}

std::pair<Value, Value> insertForOpEntryLogic(OpBuilder &builder,
                                              Value isFirstIter, Value lb,
                                              Value ub, Value step,
                                              Value tstart) {
  auto uLoc = builder.getUnknownLoc();
  auto zeroBit = helper::materializeIntegerConstant(builder, 0, 1);
  auto lbWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, lb}));
  auto ubWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, ub}));
  auto stepWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, step}));

  auto ivReg = builder.create<sv::RegOp>(uLoc, lbWide.getType());
  auto ivRegOut = builder.create<sv::ReadInOutOp>(uLoc, ivReg);
  auto ivWide = hwSelect(builder, isFirstIter, lbWide, ivRegOut);
  auto ivNext = builder.create<comb::AddOp>(uLoc, ivWide, stepWide);
  auto ivRegIn = hwSelect(builder, isFirstIter, lbWide, ivNext);
  Value clk =
      builder.create<hir::TimeGetClkOp>(uLoc, builder.getI1Type(), tstart);
  builder.create<sv::AlwaysOp>(uLoc, sv::EventControl::AtPosEdge, clk,
                               [&builder, &uLoc, &ivReg, &ivRegIn] {
                                 builder.create<sv::PAssignOp>(
                                     uLoc, ivReg.result(), ivRegIn);
                               });
  auto nextIterCondition = builder.create<comb::ICmpOp>(
      uLoc,
      ivWide.getType().isSignedInteger() ? comb::ICmpPredicate::slt
                                         : comb::ICmpPredicate::ult,
      ivNext, ubWide);
  return std::make_pair(nextIterCondition, builder.create<comb::ExtractOp>(
                                               uLoc, lb.getType(), ivWide,
                                               builder.getI32IntegerAttr(0)));
}
