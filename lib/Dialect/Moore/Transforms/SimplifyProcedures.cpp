//===- DeleteLocalVar.cpp - Delete local temporary variables --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SimplifyProcedures pass.
// Use to insert a local "shadow" variable in always blocks for every
// module-level variable that the process modifies.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_SIMPLIFYPROCEDURES
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

namespace {
struct SimplifyProceduresPass
    : public circt::moore::impl::SimplifyProceduresBase<
          SimplifyProceduresPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createSimplifyProceduresPass() {
  return std::make_unique<SimplifyProceduresPass>();
}

void SimplifyProceduresPass::runOnOperation() {
  getOperation()->walk([&](ProcedureOp procedureOp) {
    mlir::OpBuilder builder(&getContext());

    // Use to collect blocking assignments that have been replaced by a "shadow"
    // variable.
    DenseSet<Operation *> assignOps;
    procedureOp.walk([&](Operation *op) {
      auto &nestedOp = *op;
      // Only create a "shadow" varaible for the global variable used by other
      // operations in the procedure body.
      if (isa<ReadOp>(nestedOp) &&
          isa<SVModuleOp>(
              nestedOp.getOperand(0).getDefiningOp()->getParentOp())) {
        // Collect the users of the global variable that is mentioned above.
        DenseSet<Operation *> users;
        for (auto *user : nestedOp.getOperand(0).getUsers())
          // Ensuring don't handle the users existing in another procedure body.
          if (procedureOp->isAncestor(user))
            users.insert(user);

        // Because the operand of moore.event_wait is net.
        if (auto varOp = llvm::dyn_cast_or_null<VariableOp>(
                nestedOp.getOperand(0).getDefiningOp())) {
          auto resultType = varOp.getResult().getType();
          builder.setInsertionPointToStart(&procedureOp.getBody().front());
          auto readOp = builder.create<ReadOp>(
              nestedOp.getLoc(), cast<RefType>(resultType).getNestedType(),
              varOp.getResult());
          auto newVarOp = builder.create<VariableOp>(
              nestedOp.getLoc(), resultType, StringAttr{}, readOp);
          builder.clearInsertionPoint();

          // Replace the users of the global variable with a corresponding
          // "shadow" variable.
          for (auto *user : users) {
            user->replaceUsesOfWith(user->getOperand(0), newVarOp);
            if (isa<BlockingAssignOp>(user))
              assignOps.insert(user);
          }
        }
      }

      // Ensure the global variable has the correct value. So needing to create
      // a blocking assign for the global variable when the "shadow" variable
      // has a new value.
      for (auto *assignOp : assignOps)
        if (auto localVarOp = llvm::dyn_cast_or_null<VariableOp>(
                assignOp->getOperand(0).getDefiningOp())) {
          auto resultType = localVarOp.getResult().getType();
          builder.setInsertionPointAfter(assignOp);
          auto readOp = builder.create<ReadOp>(
              localVarOp.getLoc(), cast<RefType>(resultType).getNestedType(),
              localVarOp.getResult());
          builder.create<BlockingAssignOp>(
              nestedOp.getLoc(),
              localVarOp.getInitial().getDefiningOp()->getOperand(0), readOp);
          builder.clearInsertionPoint();
          assignOps.erase(assignOp);
        }
    });
  });
}
