//===- CalyxToHW.cpp - Translate Calyx into HW ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx to SV Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CalyxToHW/CalyxToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::hw;
using namespace circt::sv;

namespace {
class CalyxToHWPass : public CalyxToHWBase<CalyxToHWPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void CalyxToHWPass::runOnOperation() {
  // auto op = getOperation();
}

std::unique_ptr<mlir::Pass> circt::createCalyxToHWPass() {
  return std::make_unique<CalyxToHWPass>();
}
