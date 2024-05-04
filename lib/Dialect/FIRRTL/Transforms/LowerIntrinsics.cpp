//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// generic intrinsic operations and rewrites to their implementation.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass : public LowerIntrinsicsBase<LowerIntrinsicsPass> {
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() override;

  std::shared_ptr<IntrinsicLowerings> lowering;
};
} // namespace

/// Initialize the conversions for use during execution.
LogicalResult LowerIntrinsicsPass::initialize(MLIRContext *context) {
  IntrinsicLowerings lowering(context);

  IntrinsicLoweringInterfaceCollection loweringCollection(context);
  loweringCollection.populateIntrinsicLowerings(lowering);

  this->lowering = std::make_shared<IntrinsicLowerings>(std::move(lowering));
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  if (failed(lowering->lower(getOperation())))
    return signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerIntrinsicsPass() {
  return std::make_unique<LowerIntrinsicsPass>();
}
