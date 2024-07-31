//===- Lint.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APSInt.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LINT
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
struct LintPass : public circt::firrtl::impl::LintBase<LintPass> {
  void runOnOperation() override {
    auto fModule = getOperation();
    auto walkResult = fModule.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<WhenOp>(op))
        return WalkResult::skip();
      if (isa<AssertOp, VerifAssertIntrinsicOp>(op))
        if (checkAssert(op).failed())
          return WalkResult::interrupt();

      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    markAllAnalysesPreserved();
  };

  LogicalResult checkAssert(Operation *op) {
    Value predicate;
    if (auto a = dyn_cast<AssertOp>(op)) {
      if (auto constant = a.getEnable().getDefiningOp<firrtl::ConstantOp>())
        if (constant.getValue().isOne()) {
          predicate = a.getPredicate();
        }
    } else if (auto a = dyn_cast<VerifAssertIntrinsicOp>(op))
      predicate = a.getProperty();

    if (!predicate)
      return success();
    if (auto constant = predicate.getDefiningOp<firrtl::ConstantOp>())
      if (constant.getValue().isZero())
        return op->emitOpError(
                     "is guaranteed to fail simulation, as the predicate is "
                     "constant false")
                   .attachNote(constant.getLoc())
               << "constant defined here";

    if (auto reset = predicate.getDefiningOp<firrtl::AsUIntPrimOp>())
      if (firrtl::type_isa<ResetType, AsyncResetType>(
              reset.getInput().getType()))
        return op->emitOpError("is guaranteed to fail simulation, as the "
                               "predicate is a reset signal")
                   .attachNote(reset.getInput().getLoc())
               << "reset signal defined here";

    return success();
  }
};
} // namespace

std::unique_ptr<Pass> firrtl::createLintingPass() {
  return std::make_unique<LintPass>();
}
