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

#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERINTRINSICS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass
    : public circt::firrtl::impl::LowerIntrinsicsBase<LowerIntrinsicsPass> {

  template <typename Op>
  void runOnOp(Op op) {
    auto result = lowering->lower(op);
    if (failed(result))
      return signalPassFailure();

    numConverted += *result;

    if (*result == 0)
      markAllAnalysesPreserved();
  }

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

  TypeSwitch<Operation *>(&(*getOperation()))
      .Case<FModuleOp, FExtModuleOp, FIntModuleOp, FMemModuleOp, ClassOp,
            ExtClassOp, FormalOp>([&](auto op) { runOnOp(op); })
      // All other ops are ignored -- particularly ops that don't implement
      // the `getBodyBlock()` method. We don't want an error here because the
      // pass wasn't designed to run on those ops.
      .Default([&](auto) {});
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerIntrinsicsPass() {
  return std::make_unique<LowerIntrinsicsPass>();
}
