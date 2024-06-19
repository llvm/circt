//===- CreateCompanionAssume.cpp - Create companion assume ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateCompanionAssume pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CREATECOMPANIONASSUME
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct CreateCompanionAssumePass
    : public circt::firrtl::impl::CreateCompanionAssumeBase<
          CreateCompanionAssumePass> {
  void runOnOperation() override {
    StringAttr emptyMessage = StringAttr::get(&getContext(), "");
    getOperation().walk([&](firrtl::AssertOp assertOp) {
      OpBuilder builder(assertOp);
      builder.setInsertionPointAfter(assertOp);
      auto guards = assertOp->getAttrOfType<ArrayAttr>("guards");
      Operation *assume;
      bool isUnrOnlyAssert = false;
      // Regard the assertion as UNR only if "USE_UNR_ONLY_CONSTRAINTS" is
      // included in the guards.
      if (guards) {
        isUnrOnlyAssert = llvm::any_of(guards, [](Attribute attr) {
          StringAttr strAttr = dyn_cast<StringAttr>(attr);
          return strAttr && strAttr.getValue() == "USE_UNR_ONLY_CONSTRAINTS";
        });
      }

      // TODO: Currently messages are dropped to preserve the old behaviour.
      // Copy messages once we confirmed that it works well with UNR tools.
      if (isUnrOnlyAssert)
        // If UNROnly, use UnclockedAssumeIntrinsicOp.
        assume = builder.create<firrtl::UnclockedAssumeIntrinsicOp>(
            assertOp.getLoc(), assertOp.getPredicate(), assertOp.getEnable(),
            emptyMessage, ValueRange{}, assertOp.getName());
      else
        // Otherwise use concurrent assume.
        assume = builder.create<firrtl::AssumeOp>(
            assertOp.getLoc(), assertOp.getClock(), assertOp.getPredicate(),
            assertOp.getEnable(), emptyMessage, ValueRange{},
            assertOp.getName(),
            /*isConcurrent=*/true);

      // Add a guard "USE_PROPERTY_AS_CONSTRAINT" to companion assumes.
      SmallVector<Attribute> newGuards{
          builder.getStringAttr("USE_PROPERTY_AS_CONSTRAINT")};
      if (guards)
        newGuards.append(guards.begin(), guards.end());
      assume->setAttr("guards", builder.getArrayAttr(newGuards));
    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createCreateCompanionAssume() {
  return std::make_unique<CreateCompanionAssumePass>();
}
