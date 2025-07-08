//===- Lint.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
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
/// Linter configuration.
struct Config {
  /// If true, then assertions that are statically false (and will trivially
  /// fail simulation) will result in an error.
  bool lintStaticAsserts;
};

/// Class that stores state related to linting.  This exists to avoid needing to
/// clear members of `LintPass` and instead just rely on `Linter` objects being
/// deleted.
class Linter {

public:
  Linter(FModuleOp fModule, const Config &config)
      : fModule(fModule), config(config){};

  /// Lint the specified module.
  LogicalResult lint() {
    bool failed = false;
    fModule.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<WhenOp>(op))
        return WalkResult::skip();
      if (isa<AssertOp, VerifAssertIntrinsicOp>(op))
        if (config.lintStaticAsserts && checkAssert(op).failed())
          failed = true;

      return WalkResult::advance();
    });

    if (failed)
      return failure();

    return success();
  }

private:
  FModuleOp fModule;
  const Config &config;

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

struct LintPass : public circt::firrtl::impl::LintBase<LintPass> {
  using LintBase::lintStaticAsserts;

  void runOnOperation() override {

    CircuitOp circuitOp = getOperation();

    auto reduce = [](LogicalResult a, LogicalResult b) -> LogicalResult {
      if (succeeded(a) && succeeded(b))
        return success();
      return failure();
    };
    auto transform = [&](FModuleOp moduleOp) -> LogicalResult {
      return Linter(moduleOp, {lintStaticAsserts}).lint();
    };

    SmallVector<FModuleOp> modules(circuitOp.getOps<FModuleOp>());
    if (failed(transformReduce(circuitOp.getContext(), modules, success(),
                               reduce, transform)))
      return signalPassFailure();

    markAllAnalysesPreserved();
  };
};
} // namespace

std::unique_ptr<Pass> firrtl::createLintingPass(bool lintStaticAsserts) {
  auto pass = std::make_unique<LintPass>();
  pass->lintStaticAsserts = lintStaticAsserts;
  return pass;
}
