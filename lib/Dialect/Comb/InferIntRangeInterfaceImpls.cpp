//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for comb -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::intrange;
using namespace circt;
using namespace circt::comb;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void comb::AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferAdd(argRanges, intrange::OverflowFlags::None));
};

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void comb::SubOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferSub(argRanges, intrange::OverflowFlags::None));
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void comb::MulOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferMul(argRanges, intrange::OverflowFlags::None));
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

void comb::DivUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivU(argRanges));
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

void comb::DivSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivS(argRanges));
}

//===----------------------------------------------------------------------===//
// ModUOp
//===----------------------------------------------------------------------===//

void comb::ModUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemU(argRanges));
}

//===----------------------------------------------------------------------===//
// ModSOp
//===----------------------------------------------------------------------===//

void comb::ModSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemS(argRanges));
}
//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

void comb::AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAnd(argRanges));
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

void comb::OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferOr(argRanges));
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

void comb::XorOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferXor(argRanges));
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

void comb::ShlOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferShl(argRanges, intrange::OverflowFlags::None));
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

void comb::ShrUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrU(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

void comb::ShrSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrS(argRanges));
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

void comb::ConcatOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  // Compute concat as an unsigned integer of bits
  auto total_width = 0;
  auto res_width = getResult().getType().getIntOrFloatBitWidth();
  APInt umin = APInt::getZero(res_width);
  APInt umax = APInt::getZero(res_width);
  for (int i = getNumOperands() - 1; i >= 0; --i) {
    auto umin_upd = argRanges[i].umin().zext(res_width).ushl_sat(total_width);
    auto umax_upd = argRanges[i].umax().zext(res_width).ushl_sat(total_width);
    umin = umin.uadd_sat(umin_upd);
    umax = umax.uadd_sat(umax_upd);
    total_width += getOperand(i).getType().getIntOrFloatBitWidth();
  }
  auto urange = ConstantIntRanges::fromUnsigned(umin, umax);
  setResultRange(getResult(), urange);
};

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

void comb::ExtractOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRange) {
  // Right-shift and truncate (trunaction implicitly handled)
  // auto res_width = getResult().getType().getIntOrFloatBitWidth();
  auto low_bit = getLowBit();
  auto umin = argRanges[0].umin().ushl_sat(low_bit);
  auto umax = argRanges[0].umax().ushl_sat(low_bit);
  auto urange = ConstantIntRanges::fromUnsigned(umin, umax);
  setResultRange(getResult(), urange);
};

//===----------------------------------------------------------------------===//
// ReplicateOp
//===----------------------------------------------------------------------===//
void comb::ReplicateOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRange) {
  // Compute concat as an unsigned integer of bits
  const auto operand_width = getOperand().getType().getIntOrFloatBitWidth();
  const auto res_width = getResult().getType().getIntOrFloatBitWidth();
  APInt umin = APInt::getZero(res_width);
  APInt umax = APInt::getZero(res_width);
  auto umin_in = argRanges[0].umin().zext(res_width);
  auto umax_in = argRanges[0].umax().zext(res_width);
  for (auto total_width = 0; total_width < res_width;
       total_width += operand_width) {
    auto umin_upd = umin_in.ushl_sat(total_width);
    auto umax_upd = umax_in.ushl_sat(total_width);
    umin = umin.uadd_sat(umin_upd);
    umax = umax.uadd_sat(umax_upd);
  }
  auto urange = ConstantIntRanges::fromUnsigned(umin, umax);
  setResultRange(getResult(), urange);
};

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

void comb::MuxOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> argRanges, SetIntLatticeFn setResultRange) {
  std::optional<APInt> mbCondVal =
      argRanges[0].isUninitialized()
          ? std::nullopt
          : argRanges[0].getValue().getConstantValue();

  const IntegerValueRange &trueCase = argRanges[1];
  const IntegerValueRange &falseCase = argRanges[2];

  if (mbCondVal) {
    if (mbCondVal->isZero())
      setResultRange(getResult(), falseCase);
    else
      setResultRange(getResult(), trueCase);
    return;
  }
  setResultRange(getResult(), IntegerValueRange::join(trueCase, falseCase));
}

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

void comb::ICmpOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  comb::ICmpPredicate combPred = getPredicate();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(combPred);
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  std::optional<bool> truthValue = intrange::evaluatePred(pred, lhs, rhs);
  if (truthValue.has_value() && *truthValue)
    min = max;
  else if (truthValue.has_value() && !(*truthValue))
    max = min;

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}