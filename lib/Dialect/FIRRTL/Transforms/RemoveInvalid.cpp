//===- RemoveInvalids.cpp - Remove invalid values ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass removes invalid values from the circuit.  This is a combination of
// the Scala FIRRTL Compiler's RemoveRests pass and RemoveValidIf.  This is done
// to remove two "interpretations" of invalid.  Namely: (1) registers that are
// initialized to an invalid value (module scoped and looking through wires and
// connects only) are converted to an unitialized register and (2) invalid
// values are converted to zero (after rule 1 is applied).
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-resets"

using namespace circt;
using namespace firrtl;

struct RemoveInvalidPass : public RemoveInvalidBase<RemoveInvalidPass> {
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
      if (auto connect = dyn_cast<FConnectLike>(user)) {
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
}

void RemoveInvalidPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===----- Running RemoveInvalid "
                      "----------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);

  bool madeModifications = false;
  SmallVector<InvalidValueOp> invalidOps;
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    // Populate invalidOps for later handling.
    if (auto inv = dyn_cast<InvalidValueOp>(op)) {
      invalidOps.push_back(inv);
      continue;
    }
    auto reg = dyn_cast<RegResetOp>(op);
    if (!reg)
      continue;

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

  // Convert all invalid values to zero.
  for (auto inv : invalidOps) {
    // Skip invalids which have no uses.
    if (inv->getUses().empty())
      continue;
    ImplicitLocOpBuilder builder(inv.getLoc(), inv);
    Value replacement =
        TypeSwitch<FIRRTLType, Value>(inv.getType())
            .Case<ClockType, AsyncResetType, ResetType>(
                [&](auto type) -> Value {
                  return builder.create<SpecialConstantOp>(
                      type, builder.getBoolAttr(false));
                })
            .Case<IntType>([&](IntType type) -> Value {
              return builder.create<ConstantOp>(type, getIntZerosAttr(type));
            })
            .Case<BundleType, FVectorType>([&](auto type) -> Value {
              auto width = circt::firrtl::getBitWidth(type);
              assert(width && "width must be inferred");
              auto zero = builder.create<ConstantOp>(APSInt(*width));
              return builder.create<BitCastOp>(type, zero);
            })
            .Default([&](auto) {
              llvm_unreachable("all types are supported");
              return Value();
            });
    inv.replaceAllUsesWith(replacement);
    inv.erase();
    madeModifications = true;
  }

  if (!madeModifications)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createRemoveInvalidPass() {
  return std::make_unique<RemoveInvalidPass>();
}
