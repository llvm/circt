//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for comb -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interval range analysis interface.
// The overflow flags are not set for the comb operations since they is
// no meaningful concept of overflow detection in comb.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

using namespace mlir;
using namespace mlir::intrange;
using namespace circt;
using namespace circt::comb;
//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void comb::AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  auto resultRange = argRanges[0];
  for (size_t i = 1; i < argRanges.size(); ++i)
    resultRange =
        inferAdd({resultRange, argRanges[i]}, intrange::OverflowFlags::None);

  setResultRange(getResult(), resultRange);
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
  auto resultRange = argRanges[0];
  for (size_t i = 1; i < argRanges.size(); ++i)
    resultRange =
        inferMul({resultRange, argRanges[i]}, intrange::OverflowFlags::None);

  setResultRange(getResult(), resultRange);
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
  auto resultRange = argRanges[0];
  for (size_t i = 1; i < argRanges.size(); ++i)
    resultRange = inferAnd({resultRange, argRanges[i]});

  setResultRange(getResult(), resultRange);
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

void comb::OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
  auto resultRange = argRanges[0];
  for (size_t i = 1; i < argRanges.size(); ++i)
    resultRange = inferOr({resultRange, argRanges[i]});

  setResultRange(getResult(), resultRange);
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

void comb::XorOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  auto resultRange = argRanges[0];
  for (size_t i = 1; i < argRanges.size(); ++i)
    resultRange = inferXor({resultRange, argRanges[i]});

  setResultRange(getResult(), resultRange);
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
  const auto resWidth = getResult().getType().getIntOrFloatBitWidth();
  auto totalWidth = resWidth;
  APInt umin = APInt::getZero(resWidth);
  APInt umax = APInt::getZero(resWidth);
  for (auto [operand, arg] : llvm::zip(getOperands(), argRanges)) {
    assert(totalWidth >= operand.getType().getIntOrFloatBitWidth() &&
           "ConcatOp: total width in interval range calculation is negative");
    totalWidth -= operand.getType().getIntOrFloatBitWidth();
    auto uminUpd = arg.umin().zext(resWidth).ushl_sat(totalWidth);
    auto umaxUpd = arg.umax().zext(resWidth).ushl_sat(totalWidth);
    umin = umin.uadd_sat(uminUpd);
    umax = umax.uadd_sat(umaxUpd);
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
  const auto resWidth = getResult().getType().getIntOrFloatBitWidth();
  const auto lowBit = getLowBit();
  auto umin = argRanges[0].umin().lshr(lowBit).trunc(resWidth);
  auto umax = argRanges[0].umax().lshr(lowBit).trunc(resWidth);
  auto urange = ConstantIntRanges::fromUnsigned(umin, umax);
  setResultRange(getResult(), urange);
};

//===----------------------------------------------------------------------===//
// ReplicateOp
//===----------------------------------------------------------------------===//

void comb::ReplicateOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRange) {
  // Compute replicate as an unsigned integer of bits
  const auto operandWidth = getOperand().getType().getIntOrFloatBitWidth();
  const auto resWidth = getResult().getType().getIntOrFloatBitWidth();
  APInt umin = APInt::getZero(resWidth);
  APInt umax = APInt::getZero(resWidth);
  auto uminIn = argRanges[0].umin().zext(resWidth);
  auto umaxIn = argRanges[0].umax().zext(resWidth);
  for (unsigned int totalWidth = 0; totalWidth < resWidth;
       totalWidth += operandWidth) {
    auto uminUpd = uminIn.ushl_sat(totalWidth);
    auto umaxUpd = umaxIn.ushl_sat(totalWidth);
    umin = umin.uadd_sat(uminUpd);
    umax = umax.uadd_sat(umaxUpd);
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

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  if (combPred == comb::ICmpPredicate::ceq ||
      combPred == comb::ICmpPredicate::cne ||
      combPred == comb::ICmpPredicate::weq ||
      combPred == comb::ICmpPredicate::wne) {
    // These predicates are not supported for integer range analysis
    setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
    return;
  }

  intrange::CmpPredicate pred;
  switch (combPred) {
  case comb::ICmpPredicate::eq:
    pred = intrange::CmpPredicate::eq;
    break;
  case comb::ICmpPredicate::ne:
    pred = intrange::CmpPredicate::ne;
    break;
  case comb::ICmpPredicate::slt:
    pred = intrange::CmpPredicate::slt;
    break;
  case comb::ICmpPredicate::sle:
    pred = intrange::CmpPredicate::sle;
    break;
  case comb::ICmpPredicate::sgt:
    pred = intrange::CmpPredicate::sgt;
    break;
  case comb::ICmpPredicate::sge:
    pred = intrange::CmpPredicate::sge;
    break;
  case comb::ICmpPredicate::ult:
    pred = intrange::CmpPredicate::ult;
    break;
  case comb::ICmpPredicate::ule:
    pred = intrange::CmpPredicate::ule;
    break;
  case comb::ICmpPredicate::ugt:
    pred = intrange::CmpPredicate::ugt;
    break;
  case comb::ICmpPredicate::uge:
    pred = intrange::CmpPredicate::uge;
    break;
  default:
    llvm_unreachable("Unknown comparison predicate");
  }

  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  std::optional<bool> truthValue = intrange::evaluatePred(pred, lhs, rhs);
  if (truthValue.has_value() && *truthValue)
    min = max;
  else if (truthValue.has_value() && !(*truthValue))
    max = min;

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}
