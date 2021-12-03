//===- RemoveResets.cpp - Remove resets of invalid value --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass converts registers that are reset to an invalid value to resetless
// registers.  This is a reduced implementation of the Scala FIRRTL Compiler's
// RemoveResets pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-resets"

using namespace circt;
using namespace firrtl;

struct RemoveResetsPass : public RemoveResetsBase<RemoveResetsPass> {
  void runOnOperation() override;
};

// Returns true if this value is invalidated.  This requires that a value is
// only ever driven once.  This is guaranteed if this runs after the
// `ExpandWhens` pass .
static bool isInvalid(Value val) {

  // Update `val` to the source of the connection driving `thisVal`.  This walks
  // backwards across users to find the first connection and updates `val` to
  // the source.  This assumes that only one connect is driving `thisVal`, i.e.,
  // this pass runs after `ExpandWhens`.
  auto updateVal = [&](Value thisVal) {
    for (auto *user : thisVal.getUsers()) {
      if (auto connect = dyn_cast<ConnectOp>(user)) {
        if (connect.dest() != val)
          continue;
        val = connect.src();
        return;
      }
    }
    val = nullptr;
    return;
  };

  while (val) {
    // The value is a port.
    if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      FModuleOp op = cast<FModuleOp>(val.getParentBlock()->getParentOp());
      auto direction = op.getPortDirection(blockArg.getArgNumber());
      // Base case: this is an input port and cannot be invalidated in module
      // scope.
      if (direction == Direction::In)
        return false;
      updateVal(blockArg);
      continue;
    }

    auto *op = val.getDefiningOp();

    // The value is an instance port.
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto resultNo = val.cast<OpResult>().getResultNumber();
      // An output port of an instance crosses a module boundary.  This is not
      // invalid within module scope.
      if (inst.getPortDirection(resultNo) == Direction::Out)
        return false;
      updateVal(val);
      continue;
    }

    // Base case: we found an invalid value.  We're done, return true.
    if (isa<InvalidValueOp>(op))
      return true;

    // Base case: we hit something that is NOT a wire, e.g., a PrimOp.  We're
    // done, return false.
    if (!isa<WireOp>(op))
      return false;

    // Update `val` with the driver of the wire.  If no driver found, `val` will
    // be set to nullptr and we exit on the next while iteration.
    updateVal(op->getResult(0));
  };
  return false;
};

void RemoveResetsPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===----- Running RemoveResets "
                      "-----------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);

  bool madeModifications = false;
  for (auto reg : llvm::make_early_inc_range(
           getOperation().getBody()->getOps<RegResetOp>())) {

    // If the `RegResetOp` has an invalidated initialization, then replace it
    // with a `RegOp`.
    if (isInvalid(reg.resetValue())) {
      LLVM_DEBUG(llvm::dbgs() << "  - RegResetOp '" << reg.name()
                              << "' will be replaced with a RegOp\n");
      ImplicitLocOpBuilder builder(reg.getLoc(), reg);
      RegOp newReg =
          builder.create<RegOp>(reg.getType(), reg.clockVal(), reg.name(),
                                reg.annotations(), reg.inner_symAttr());
      reg.replaceAllUsesWith(newReg.getResult());
      reg.erase();
      madeModifications = true;
    }
  }

  if (!madeModifications)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createRemoveResetsPass() {
  return std::make_unique<RemoveResetsPass>();
}
