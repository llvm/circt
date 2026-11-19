//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Support/ProceduralRegionTrait.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-hoist-projections"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_HOISTPROJECTIONSPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

/// Hoist a signal projection one level out of the procedural region it is
/// nested in. Returns the op that now represents the projection outside the
/// region, or null if the projection could not be hoisted.
static Operation *hoistProjection(Operation *op) {
  auto *parentOp = op->getParentOp();
  if (!parentOp || !parentOp->hasTrait<ProceduralRegion>())
    return nullptr;

  // Check if we can hoist.
  auto *parentRegion = op->getParentRegion();
  if (llvm::any_of(op->getOperands(), [&](Value operand) {
        return parentRegion->isAncestor(operand.getParentRegion()) &&
               !matchPattern(operand.getDefiningOp(), m_Constant());
      }))
    return nullptr;

  // Move the projection in front of the region, cloning any constant
  // operands defined inside the region along with it.
  LLVM_DEBUG(llvm::dbgs() << "Hoisting " << *op << "\n");
  op->moveBefore(parentOp);
  OpBuilder builder(op);
  for (auto &operand : op->getOpOperands()) {
    if (!parentRegion->isAncestor(operand.get().getParentRegion()))
      continue;
    auto *constOp = builder.clone(*operand.get().getDefiningOp());
    operand.set(constOp->getResult(0));
  }
  return op;
}

namespace {
struct HoistProjectionsPass
    : public llhd::impl::HoistProjectionsPassBase<HoistProjectionsPass> {
protected:
  void runOnOperation() override;
};
} // namespace

void HoistProjectionsPass::runOnOperation() {
  // Collect the projections in pre-order, such that projections are visited
  // before projections derived from them. This allows chains of projections to
  // be hoisted in a single sweep.
  SmallVector<Operation *> projections;
  getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<SigExtractOp, SigArraySliceOp, SigArrayGetOp, SigStructExtractOp>(
            op) &&
        isInProceduralRegion(op))
      projections.push_back(op);
  });

  // Hoist each projection out of as many procedural regions as possible.
  for (auto *op : projections) {
    bool hoisted = false;
    while (auto *newOp = hoistProjection(op)) {
      if (newOp != op)
        break;
      hoisted = true;
    }
    if (hoisted)
      ++numProjectionsHoisted;
  }
}
