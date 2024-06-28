//===- MergeAssignments.cpp - Merge declaration and assignment ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MergeAssignments pass.
// Find easy uses of declaration and assignments and merge them into
// assigned_variable. Easy use represents the declaration doesn't be performed
// bit slice. Like bit [127:0] b; assign b = 128'b0;
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_MERGEASSIGNMENTS
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

namespace {
struct MergeAssignmentsPass
    : public circt::moore::impl::MergeAssignmentsBase<MergeAssignmentsPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createMergeAssignmentsPass() {
  return std::make_unique<MergeAssignmentsPass>();
}

// TODO: The net can be driven multiple times. However, the related rule is
// complicated. So we will implement this in the future.

// Only collect the easy declaration and its value at module level.
static void collectAssignmets(SVModuleOp moduleOp,
                              llvm::MapVector<Value, Value> &assignments) {
  moduleOp->walk([&](ContinuousAssignOp op) {
    if (isa<VariableOp, NetOp>(op.getDst().getDefiningOp())) {
      if (!assignments.lookup(op.getDst()))
        assignments[op.getDst()] = op.getSrc();
      else
        assignments.erase(op.getDst());
    }
  });
}

void MergeAssignmentsPass::runOnOperation() {
  OpBuilder builder(&getContext());

  // Use to collect the easy declaration and its value.
  llvm::MapVector<Value, Value> assignments;

  collectAssignmets(getOperation(), assignments);
  for (auto assignment : assignments) {
    auto varName =
        assignment.first.getDefiningOp()->getAttrOfType<StringAttr>("name");

    builder.setInsertionPointAfterValue(assignment.first);
    builder.create<AssignedVarOp>(assignment.first.getLoc(),
                                  assignment.first.getType(), varName,
                                  assignment.second);
  }
}
