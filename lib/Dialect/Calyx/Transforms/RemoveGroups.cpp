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
static void modifyGroups(ComponentOp &component) {
  // Get the only group in the control.
  auto control = component.getControlOp();
  EnableOp topLevel;
  // TODO: Fix. Emit op error if there is more than a single Enable.
  control.walk([&](EnableOp e) { topLevel = e; });

  auto wires = component.getWiresOp();
  Value componentGoPort = component.getGoPort();
  Value componentDonePort = component.getDonePort();

  wires.walk([&](GroupOp group) {
    auto &groupRegion = group->getRegion(0);
    OpBuilder builder(groupRegion);
    // Walk the assignments. Append component's `%go` signal to each guard.
    updateGroupAssignmentGuards(builder, group, componentGoPort);

    auto groupDone = group.getDoneOp();
    if (topLevel.groupName() == group.sym_name()) {
      // Replace `calyx.group_done %0, %1 ? : i1`
      //    with `calyx.assign %done, %0, %1 ? : i1`
      auto assignOp =
          builder.create<AssignOp>(group->getLoc(), componentDonePort,
                                   groupDone.src(), groupDone.guard());
      groupDone->replaceAllUsesWith(assignOp);
    } else {
      // Replace calyx.group_go's uses with its guard, e.g.
      //    %A.go = calyx.group_go %true, %3 ? : i1
      //    %x = comb.and %1, %A.go : i1
      //    ->
      //    %x = comb.and %1, %3 : i1
      auto groupGo = group.getGoOp();
      auto groupGoGuard = groupGo.guard();
      groupGo.replaceAllUsesWith(groupGoGuard);
      groupGo->erase();
    }
    // In either case, remove the group's done value.
    groupDone->erase();
  });
}

/// ???
// static void inlineGroups(ComponentOp &component) {
//
//  auto wires = component.getWiresOp();
//  wires.walk([&](GroupOp group) {
//    for (auto &&op : group) {
//    }
//    auto &groupRegion = group->getRegion(0);
//    OpBuilder builder(groupRegion);
//  });
//}

namespace {

struct RemoveGroupsPass : public RemoveGroupsBase<RemoveGroupsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RemoveGroupsPass::runOnOperation() {
  ComponentOp component = getOperation();
  modifyGroups(component);

  // Inline the body of each group.
  auto wiresBody = component.getWiresOp().getBody();
  wiresBody->walk([&](GroupOp group) {
    auto body = group.getBody();
    body->print(llvm::errs());
    for (auto &&op : *body) {
      wiresBody->push_back(op.clone());
    }
  });

  wiresBody->walk([&](GroupOp group) { group->dropAllDefinedValueUses(); group->erase(); });

  // Remove the last EnableOp from the control.
  auto control = component.getControlOp();
  control.walk([&](EnableOp enable) { enable->erase(); });
  assert(control.getBody()->empty() &&
         "The calyx.control should be compiled after this pass.");
}

std::unique_ptr<mlir::Pass> circt::calyx::createRemoveGroupsPass() {
  return std::make_unique<RemoveGroupsPass>();
}
