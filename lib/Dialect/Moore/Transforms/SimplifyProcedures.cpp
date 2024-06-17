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
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

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
    for (auto &nestedOp : procedureOp) {
      // Only create a "shadow" varaible for the global variable used by other
      // operations in the procedure body.
      if (isa<ReadOp>(nestedOp) &&
          isa<SVModuleOp>(
              nestedOp.getOperand(0).getDefiningOp()->getParentOp())) {
        // Collect the users of the global variable that is mentioned above.
        DenseSet<Operation *> users;
        for (auto *user : nestedOp.getOperand(0).getUsers())
          if (!users.contains(user))
            users.insert(user);

        auto varOp = cast<VariableOp>(nestedOp.getOperand(0).getDefiningOp());
        auto varName = builder.getStringAttr("local_" + varOp.getName());
        auto resultType = varOp.getResult().getType();
        builder.setInsertionPointToStart(procedureOp.getBody());
        auto readOp = builder.create<ReadOp>(
            nestedOp.getLoc(), cast<RefType>(resultType).getNestedType(),
            varOp.getResult());
        auto newVarOp = builder.create<VariableOp>(nestedOp.getLoc(),
                                                   resultType, varName, readOp);
        builder.clearInsertionPoint();

        // Replace the users of the global variable with a corresponding
        // "shadow" variable.
        for (auto *user : users) {
          user->replaceUsesOfWith(user->getOperand(0), newVarOp);
          if (isa<BlockingAssignOp>(user))
            assignOps.insert(user);
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
    }
    return WalkResult::advance();
  });
}
