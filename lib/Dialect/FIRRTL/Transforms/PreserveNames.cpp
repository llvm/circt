//===- PreserveNames.cpp - FIRRTL name preservation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Preserve names to improve output readability.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace firrtl;

/// Predicates to decide when to materialize namehints for readability.

// Check if an expression is used in another expression of the same op, which
// would be flattened by comb canonicalizers. Preventing flattening will improve
// readability.
static bool isCandidateForFlattening(Operation *op) {
  // The op defining the expression that could be flattened.
  Operation *definingOp = op;

  // If op is a BitsPrimOp, look through it.
  if (auto bits = dyn_cast<BitsPrimOp>(op))
    if (auto *bitsInput = bits.getInput().getDefiningOp())
      definingOp = bitsInput;

  // If any of the users are the same op as the defining op, block optimization.
  for (auto *user : op->getUsers())
    if (user->getName() == definingOp->getName())
      return true;

  return false;
}

// General checks, with dispatch to specific scenarios.
static bool shouldMaterializeNamehints(Operation *op) {
  // If the op is already a declaration, nothing to do.
  if (isa<FNamableOp>(op))
    return false;

  // If there is no name, nothing to do.
  auto name = op->getAttrOfType<StringAttr>("name");
  if (!name)
    return false;

  // If the namehint is useless, nothing to do.
  if (isUselessName(name.getValue()))
    return false;

  // If the op has more than one result, we can't name them all, nothing to do.
  if (op->getNumResults() != 1)
    return false;

  // If the op is dead, nothing to do.
  if (op->getResult(0).use_empty())
    return false;

  // After the generic checks, check some specific scenarios.
  return isCandidateForFlattening(op);
}

/// Materialize good names as NodeOps with symbols for readability.
static void materializeNamehints(Operation *op, OpBuilder &builder) {
  auto name = op->getAttrOfType<StringAttr>("name");
  builder.setInsertionPointAfter(op);
  auto nodeOp = builder.create<NodeOp>(
      op->getLoc(), op->getResult(0), name, NameKindEnum::InterestingName,
      /*annotations=*/ArrayAttr::get(op->getContext(), {}), name);
  op->getResult(0).replaceAllUsesExcept(nodeOp.getResult(), nodeOp);
}

/// The pass declaration.
namespace {
struct PreserveNamesPass : public PreserveNamesBase<PreserveNamesPass> {
  void runOnOperation() override;
};
} // namespace

/// The pass entrypoint.
void PreserveNamesPass::runOnOperation() {
  // Create a builder for IR modifications.
  OpBuilder builder(getOperation());

  // Walk the module body materializing any namehints.
  getOperation().walk([&](Operation *op) {
    if (shouldMaterializeNamehints(op))
      materializeNamehints(op, builder);
  });
}

/// The pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createPreserveNamesPass() {
  return std::make_unique<PreserveNamesPass>();
}
