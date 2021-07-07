//===- GoInsertion.cpp - Go Insertion Pass ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Go Insertion pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

using namespace circt;
using namespace calyx;

namespace {

/// Adds the group's "go" signal to the guards of assignments within `group`,
/// with the exception of the "done" terminator. If the assignment already
/// has a guard, then the bitwise 'and' is taken of the current guard and the
/// "go" signal. For example:
///    ```mlir
///      %go = calyx.group_go %c1_0 : i1
///
///      // Case 1: No Guard
///      %in = %out : i8
///      =>
///      %in = %out, %go ? : i8
///
///      // Case 2: Guard
///      %in = %out, %guard ? : i8
///      =>
///      %and = comb.and %guard, %go : i1
///      %in = %out, %and ? : i8
///    ```
static void updateGroupAssignmentGuards(OpBuilder &builder, GroupOp &group,
                                        GroupGoOp &goOp) {
  group.walk([&](Operation *op) {
    if (!isa<AssignOp>(op))
      return;
    auto assign = cast<AssignOp>(op);
    if (assign.guard())
      // Take the bitwise & of the current guard and the group's go signal.
      assign->setOperand(
          2, builder.create<comb::AndOp>(group.getLoc(), assign.guard(), goOp));
    else
      assign->insertOperands(2, {goOp});
  });
}

struct GoInsertionPass : public GoInsertionBase<GoInsertionPass> {
  void runOnOperation() override {
    ComponentOp component = getOperation();
    OpBuilder builder(component->getRegion(0));

    auto zeroConstant = builder.create<hw::ConstantOp>(
        component->getLoc(), APInt(/*numBits=*/1, /*val=*/0));

    auto wiresOp = component.getWiresOp();
    wiresOp.walk([&](Operation *op) {
      if (!isa<GroupOp>(op))
        return;
      auto group = cast<GroupOp>(op);
      OpBuilder builder(group->getRegion(0));
      // Since the source of a GroupOp's go signal isn't set until the
      // a following pass, a dummy constant is used as a placeholder.
      auto goOp = builder.create<GroupGoOp>(group->getLoc(), zeroConstant);

      updateGroupAssignmentGuards(builder, group, goOp);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::calyx::createGoInsertionPass() {
  return std::make_unique<GoInsertionPass>();
}
