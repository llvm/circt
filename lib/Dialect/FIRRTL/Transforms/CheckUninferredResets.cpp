//===- CheckUninferredResets.cpp - Verify no uninferred resets ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass checks to see if there are abstract type resets which have not
// been inferred correctly.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKUNINFERREDRESETS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
struct CheckUninferredResetsPass
    : public circt::firrtl::impl::CheckUninferredResetsBase<
          CheckUninferredResetsPass> {
  void runOnOperation() override;
};
} // namespace

void CheckUninferredResetsPass::runOnOperation() {
  auto module = getOperation();
  for (auto port : module.getPorts()) {
    if (getBaseOfType<ResetType>(port.type)) {
      auto diag = emitError(port.loc)
                  << "a port \"" << port.getName()
                  << "\" with abstract reset type was unable to be "
                     "inferred by InferResets (is this a top-level port?)";
      diag.attachNote(module->getLoc())
          << "the module with this uninferred reset port was defined here";
      return signalPassFailure();
    }
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckUninferredResetsPass() {
  return std::make_unique<CheckUninferredResetsPass>();
}
