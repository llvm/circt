//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for arith -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  setResultRange(getResult(), inferAdd(argRanges, intrange::OverflowFlags::None));
};

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

// void arith::SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
//                                       SetIntRangeFn setResultRange) {
//   setResultRange(getResult(), inferSub(argRanges, intrange::OverflowFlags::None));
// }

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

// void arith::MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
//                                       SetIntRangeFn setResultRange) {
//   setResultRange(getResult(), inferMul(argRanges, intrange::OverflowFlags::None));
// }

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void comb::ConcatOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRange) {

    auto total_width = 0;
    auto res_width = getResult().getType().getIntOrFloatBitWidth();
    APInt umin = APInt::getZero(res_width);
    APInt umax = APInt::getZero(res_width);
    for (int i = getNumOperands()-1; i >= 0; --i) {
      auto umin_upd = argRanges[i].umin().zext(res_width).ushl_sat(total_width);
      auto umax_upd = argRanges[i].umax().zext(res_width).ushl_sat(total_width);
      umin = umin.uadd_sat(umin_upd);
      umax = umax.uadd_sat(umax_upd);
      total_width += getOperand(i).getType().getIntOrFloatBitWidth();

    }
    auto urange = ConstantIntRanges::fromUnsigned(umin, umax);
    // auto srange = ConstantIntRanges::fromUnsigned(umin, umax);

  setResultRange(getResult(), urange);
};
