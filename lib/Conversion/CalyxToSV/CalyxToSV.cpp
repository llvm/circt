//===- CalyxToSV.cpp - Translate Calyx into SV ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx to SV Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CalyxToSV/CalyxToSV.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::sv;

namespace {
class CalyxToSVPass : public CalyxToSVBase<CalyxToSVPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void CalyxToSVPass::runOnOperation() {
  // auto op = getOperation();
  ConversionTarget target(getContext());
  target.addIllegalDialect<CalyxDialect>();
  target.addLegalDialect<SVDialect>();
}

std::unique_ptr<mlir::Pass> circt::createCalyxToSVPass() {
  return std::make_unique<CalyxToSVPass>();
}
