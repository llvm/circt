//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/UnusedOpPruner.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_LOWERVALIDATETOLABELSPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Lower Validate To Labels Pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerValidateToLabelsPass
    : public rtg::impl::LowerValidateToLabelsPassBase<
          LowerValidateToLabelsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerValidateToLabelsPass::runOnOperation() {
  auto *rootOp = getOperation();
  UnusedOpPruner pruner;

  auto result = rootOp->walk([&](rtg::ValidateOp validateOp) -> WalkResult {
    Location loc = validateOp.getLoc();
    auto regOp = validateOp.getRef().getDefiningOp<rtg::ConstantOp>();
    if (!regOp)
      return validateOp->emitError(
          "could not determine register defining operation");

    auto reg = dyn_cast<rtg::RegisterAttrInterface>(regOp.getValue());
    if (!reg)
      return validateOp->emitError("could not determine register");

    if (!validateOp.getId().has_value())
      return validateOp.emitError("expected ID to be set");

    OpBuilder builder(validateOp);
    auto intrinsicLabel = validateOp.getRef().getType().getIntrinsicLabel(
        reg, validateOp.getId().value());
    Value lbl = rtg::ConstantOp::create(
        builder, loc, rtg::LabelAttr::get(&getContext(), intrinsicLabel));
    rtg::LabelOp::create(builder, loc, rtg::LabelVisibility::global, lbl);
    validateOp.getValue().replaceAllUsesWith(validateOp.getDefaultValue());
    validateOp.getValues().replaceAllUsesWith(
        validateOp.getDefaultUsedValues());

    pruner.eraseNow(validateOp);
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();

  pruner.eraseNow();
}
