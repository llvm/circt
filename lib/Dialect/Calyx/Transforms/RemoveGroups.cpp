//===- RemoveGroups.cpp - Remove Groups Pass --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Remove Groups pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

/// ???
static void inlineGroupAssignments(ComponentOp &component) {
  // Get the only group in the control.
  auto control = component.getControlOp();
  EnableOp topLevel;
  // TODO: Fix
  control.walk([&](EnableOp e) { topLevel = e; });

  auto wires = component.getWiresOp();
  auto componentGoPort = component.getGoPort();
  auto componentDonePort = component.getDonePort();

  wires.walk([&](GroupOp group) {
    auto &groupRegion = group->getRegion(0);
    OpBuilder builder(groupRegion);
    // Walk the assignments. Add `& %go` to each guard (see: GoInsertion).
    updateGroupAssignmentGuards(builder, group, componentGoPort);

    auto groupDone = group.getDoneOp().clone();
    if (topLevel.groupName() == group.sym_name()) {
      // 1. Replace `calyx.group_done %0, %1 ? : i1`
      //    with
      //    `calyx.assign %done, %0, %1 ? : i1`
      builder.create<AssignOp>(group->getLoc(), componentDonePort,
                               groupDone.src(), groupDone.guard());
    } else {
      // 1. Remove calyx.group_done, these values were inserted in the last
      // pass.
      // 2. Replace calyx.group_go's uses with its guard:
      //    e.g.
      //    %A_go = calyx.group_go %true, %3 ? : i1
      //    ...
      //    %0 = comb.and %1, %A_go : i1
      //    with
      //    %0 = comb.and %1, %3 : i1
      // 3. Remove calyx.group_go operations.
      auto groupGo = group.getGoOp();
      auto guard = groupGo.guard();

      groupGo.replaceAllUsesWith(guard);
      groupGo->erase();
    }
    // In either case, remove the group's done value.
    groupDone->erase();

    // Inline the group.
    // TODO: Causing an error :-(
    // wires->getRegion(0).takeBody(groupRegion);
  });
}

namespace {

struct RemoveGroupsPass : public RemoveGroupsBase<RemoveGroupsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RemoveGroupsPass::runOnOperation() {
  ComponentOp component = getOperation();
  inlineGroupAssignments(component);

  // Remove the groups.
  component.getWiresOp().walk([](GroupOp group) { group->erase(); });

  // Remove the last EnableOp from the control.
  auto control = component.getControlOp();
  control.walk([&](EnableOp enable) { enable->erase(); });
  assert(control.getBody()->empty() &&
         "The calyx.control should be compiled after this pass.");
}

std::unique_ptr<mlir::Pass> circt::calyx::createRemoveGroupsPass() {
  return std::make_unique<RemoveGroupsPass>();
}
