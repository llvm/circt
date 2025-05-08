//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for HW -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

using namespace mlir;
using namespace mlir::intrange;
using namespace circt;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void hw::ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), ConstantIntRanges::constant(getValue()));
}
