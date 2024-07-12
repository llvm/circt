//===- PullNonBlockingUp.cpp - Pull non-blocking assignments upwards ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is aimed at pulling non-blokcing assignments upwards.
// For example:

// From:
// moore.procedure {
//   scf.if not(%rst) {
//     ...
//   } else {
//     scf.if eq(cond1, a1) {
//       %0 = moore.constant 1
//       %1 = moore.extract_ref %q, %0
//       moore.nonblokcing %1, %0
//     }

//     scf.if eq(cond2, a2) {
//        %0 = moore.constant 1
//        %1 = moore.extract_ref %q, %0
//        moore.nonblokcing %1, %0
//     }
//    }
// }

// To:
// moore.procedure {
//   scf.if not(%rst) {
//     ...
//   } else {
//     %0 = moore.constant 1
//     %1 = moore.extract_ref %q, %0
//     moore.nonblokcing %1, %0 if eq(cond1, a1)

//     %2 = moore.constant 1
//     %3 = moore.extract_ref %q, %0
//     moore.nonblokcing %3, %2 if eq(cond2, a2)
//    }
// }

// Round and round until 'scf.if' doesn't exist.

//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_PULLNONBLOCKINGUP
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace mlir;
using namespace moore;

namespace {
static LogicalResult lowerSCFIfOp(scf::IfOp ifOp, PatternRewriter &rewriter) {
  SmallVector<Operation *> assignOps;
  auto cond = ifOp.getCondition().getDefiningOp()->getOperand(0);

  // First to traverse else-region.
  for (auto &elseOp : ifOp.getElseRegion().getOps()) {
    // Handle the nested if-else statements.
    if (isa<scf::IfOp>(elseOp))
      if (failed(lowerSCFIfOp(cast<scf::IfOp>(elseOp), rewriter)))
        return failure();

    // Handle non-blocking assignments.
    if (auto assignOp = llvm::dyn_cast_or_null<NonBlockingAssignOp>(elseOp)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(assignOp);

      // The condition of the else-region is contrary to the one of the
      // then-region.
      Value elseCond = rewriter.create<NotOp>(assignOp.getLoc(), cond);

      // When there are nested if-else statements, we need to guarantee the
      // nested non-blocking assignments have the correct conditions.
      if (auto en = assignOp.getEnable())
        elseCond = rewriter.create<AndOp>(assignOp->getLoc(), en, elseCond);
      rewriter.create<NonBlockingAssignOp>(
          assignOp->getLoc(), assignOp.getDst(), assignOp.getSrc(), elseCond);
      assignOps.push_back(assignOp);
    }
  }

  // Seconde to traverse then region.
  for (auto &thenOp : ifOp.getThenRegion().getOps()) {
    // Handle the nested if-else statements.
    if (isa<scf::IfOp>(thenOp)) {
      if (failed(lowerSCFIfOp(cast<scf::IfOp>(thenOp), rewriter)))
        return failure();
    }

    // Handle non-blocking assignments.
    if (auto assignOp = llvm::dyn_cast_or_null<NonBlockingAssignOp>(thenOp)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(assignOp);
      Value thenCond = cond;

      // As above.
      if (auto en = assignOp.getEnable())
        thenCond = rewriter.create<AndOp>(assignOp->getLoc(), en, cond);
      rewriter.create<NonBlockingAssignOp>(
          assignOp->getLoc(), assignOp.getDst(), assignOp.getSrc(), thenCond);
      assignOps.push_back(assignOp);
    }
  }

  for (auto *assignOp : assignOps)
    assignOp->erase();

  // After merging conditions on the non-blocking assignment, we need to move
  // these nested operations in the if-else body outside to this body.
  // Meanwhile, erase the scf.yield. Then erase scf.if.
  rewriter.eraseOp(ifOp.thenYield());
  rewriter.inlineBlockBefore(&ifOp.getThenRegion().front(), ifOp);
  if (!ifOp.getElseRegion().empty()) {
    rewriter.eraseOp(ifOp.elseYield());
    rewriter.inlineBlockBefore(&ifOp.getElseRegion().front(), ifOp);
  }
  rewriter.eraseOp(ifOp);
  return success();
}

struct PullNonBlockingUpPass
    : public circt::moore::impl::PullNonBlockingUpBase<PullNonBlockingUpPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createPullNonBlockingUpPass() {
  return std::make_unique<PullNonBlockingUpPass>();
}

void PullNonBlockingUpPass::runOnOperation() {
  getOperation()->walk([](scf::IfOp op) {
    PatternRewriter rewriter(op.getContext());
    if (failed(lowerSCFIfOp(op, rewriter)))
      return;
  });
}
