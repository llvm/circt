//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOpInterfaces.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_MATERIALIZECONSTRAINTSPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Materialize Constraints Pass
//===----------------------------------------------------------------------===//

namespace {
struct MaterializeConstraintsPass
    : public rtg::impl::MaterializeConstraintsPassBase<
          MaterializeConstraintsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void MaterializeConstraintsPass::runOnOperation() {
  getOperation()->walk([&](ImplicitConstraintOpInterface op) {
    if (op.isConstraintMaterialized())
      return;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto *newOp = op.materializeConstraint(builder);
    if (newOp == op)
      return;
    if (newOp && op->getNumResults() > 0)
      op->replaceAllUsesWith(newOp);
    assert(newOp ||
           op->getNumResults() == 0 &&
               "cannot erase operation without result value replacements");
    op->erase();
  });
}
