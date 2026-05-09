//===- FoldValueFormatters.cpp - Fold constant value formatters -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts `sim.fmt.* %cst` operations with constant inputs to
// `sim.fmt.literal` operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_FOLDVALUEFORMATTERS
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace llvm;
using namespace circt;
using namespace sim;

namespace {
struct FoldValueFormattersPass
    : impl::FoldValueFormattersBase<FoldValueFormattersPass> {
public:
  void runOnOperation() override;
};
} // namespace

void FoldValueFormattersPass::runOnOperation() {

  bool anyChanged = false;
  OpBuilder builder(getOperation());

  getOperation().getBody()->walk([&](Operation *op) {
    // Check if this is a value formatter with a constant input.
    if (!isa_and_nonnull<SimDialect>(op->getDialect()))
      return;
    auto valFmtOp = dyn_cast<ValueFormatter>(op);
    if (!valFmtOp)
      return;
    auto fmtValue = getFormattedValue(op);
    auto *valDefOp = fmtValue.getDefiningOp();
    if (!valDefOp || !valDefOp->hasTrait<OpTrait::ConstantLike>())
      return;
    // Call the defining op's fold method to get the constant attribute.
    SmallVector<OpFoldResult, 1> foldResult;
    if (failed(valDefOp->fold((foldResult))))
      return;
    auto opResultIdx = cast<OpResult>(fmtValue).getResultNumber();
    assert((foldResult.size() > opResultIdx &&
            isa<Attribute>(foldResult[opResultIdx])) &&
           "ConstantLike operation should fold to constant attributes");
    // Perform the formatting
    auto stringConst =
        valFmtOp.formatConstant(cast<Attribute>(foldResult[opResultIdx]));
    if (!stringConst)
      return;
    // Replace the operation with a literal.
    builder.setInsertionPoint(valFmtOp);
    auto literalOp =
        FormatLiteralOp::create(builder, valFmtOp.getLoc(), stringConst);
    assert(valFmtOp.getOperation()->getNumResults() == 1);
    valFmtOp.getOperation()->getResult(0).replaceAllUsesWith(
        literalOp.getResult());
    valFmtOp.getOperation()->erase();
    anyChanged = true;
  });

  if (!anyChanged)
    markAllAnalysesPreserved();
}
