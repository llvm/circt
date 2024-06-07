//===- StrictWires.cpp - Make Wires Strict ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the StrictWires pass.  This pass converts passive wires
// to strict wires.  Strict wires have read and write ports.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#include <deque>

#define DEBUG_TYPE "firrtl-strict-wires"

using namespace circt;
using namespace firrtl;

static StrictWireOp cloneWireTo(mlir::OpBuilder &builder, WireOp wire) {
  builder.setInsertionPoint(wire);
  return builder.create<StrictWireOp>(
      wire.getLoc(), cast<FIRRTLBaseType>(wire.getResult().getType()),
      wire.getNameAttr(), wire.getNameKindAttr(), wire.getAnnotations(),
      wire.getInnerSymAttr(), wire.getForceableAttr());
}

static StrictInstanceOp cloneInstTo(mlir::OpBuilder &builder, InstanceOp inst) {
  builder.setInsertionPoint(inst);

  SmallVector<Direction> fixedDirections;
  for (auto d : inst.getPortDirections())
    fixedDirections.push_back(direction::get(d));

  // This uses the builtin and needs types fixed
  return builder.create<StrictInstanceOp>(
      inst.getLoc(), inst.getResultTypes(), inst.getModuleName(),
      inst.getName(), inst.getNameKind(), fixedDirections,
      inst.getPortNames().getValue(), inst.getAnnotations().getValue(),
      inst.getPortAnnotations().getValue(), inst.getLayers(),
      inst.getLowerToBind(), inst.getInnerSymAttr());
}

// This is recursive, but effectively recursive on a type.
static void updateUses(mlir::OpBuilder &builder,
                       std::deque<Operation *> &toDelete, Value toReplace,
                       Value readSide, Value writeSide) {
  for (auto *user : toReplace.getUsers()) {
    TypeSwitch<Operation *>(user)
        .Case<SubfieldOp>([&](auto op) {
          builder.setInsertionPoint(op);
          auto newWrite = builder.create<LHSSubfieldOp>(op.getLoc(), writeSide,
                                                        op.getFieldIndex());
          return updateUses(builder, toDelete, op, op.getResult(), newWrite);
        })
        .Case<SubindexOp>([&](auto op) {
          builder.setInsertionPoint(op);
          auto newWrite = builder.create<LHSSubindexOp>(op.getLoc(), writeSide,
                                                        op.getIndex());
          return updateUses(builder, toDelete, op, op.getResult(), newWrite);
        })
        .Case<MatchingConnectOp>([&](auto op) {
          if (op.getDest() == toReplace) {
            builder.setInsertionPoint(op);
            builder.create<StrictConnectOp>(op.getLoc(), writeSide,
                                            op.getSrc());
            toDelete.push_back(op);
          }
          return;
        });
  }

  if (toReplace != readSide && readSide)
    toReplace.replaceAllUsesWith(readSide);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct StrictWiresPass : public StrictWiresBase<StrictWiresPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void StrictWiresPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n";);
  auto module = getOperation();
  mlir::OpBuilder builder(module);
  std::deque<Operation *> toDelete;

  module.walk([&](Operation *op) -> WalkResult {
    if (auto wire = dyn_cast<WireOp>(op)) {
      if (!cast<FIRRTLType>(wire.getResult().getType())
               .getRecursiveTypeProperties()
               .isPassive)
        return WalkResult::advance();
      auto newWire = cloneWireTo(builder, wire);
      updateUses(builder, toDelete, wire.getResult(), newWire.getRead(),
                 newWire.getWrite());
      toDelete.push_back(wire);
    } else if (auto inst = dyn_cast<InstanceOp>(op)) {
      for (auto type : inst.getResultTypes()) {
        auto rtype = type_dyn_cast<FIRRTLBaseType>(type);
        if (rtype && !rtype.isPassive())
          return WalkResult::advance();
      }
      auto newInst = cloneInstTo(builder, inst);
      // For now, assume outputs are not duplex.  If this isn't true, make a
      // bounce wire.
      for (auto [result, newResult] :
           llvm::zip(inst.getResults(), newInst.getResults()))
        if (isa<LHSType>(newResult.getType()))
          updateUses(builder, toDelete, result, {}, newResult);
        else
          updateUses(builder, toDelete, result, newResult, {});
      toDelete.push_back(inst);
    }
    return WalkResult::advance();
  });

  for (auto w : toDelete)
    w->erase();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createStrictWiresPass() {
  return std::make_unique<StrictWiresPass>();
}
