//===- HWArithToHW.cpp - HWArith to HW Lowering pass ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HWArith to HW Lowering Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWArithToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HWArith/HWArithOps.h"

using namespace llvm;
using namespace mlir;

namespace circt {

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class HWArithToHWPass : public HWArithToHWBase<HWArithToHWPass> {
public:
  void runOnOperation() override{};
};

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createHWArithToHWPass() {
  return std::make_unique<HWArithToHWPass>();
}

} // namespace circt
