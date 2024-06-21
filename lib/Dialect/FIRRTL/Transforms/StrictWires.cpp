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

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
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

static FStrictModuleOp cloneModuleTo(mlir::OpBuilder &builder, FModuleOp mod) {
  builder.setInsertionPoint(mod);

  // This uses the builtin and needs types fixed
  auto newMod = builder.create<FStrictModuleOp>(
      mod.getLoc(), mod.getNameAttr(), mod.getConventionAttr(), mod.getPorts(),

      mod.getAnnotationsAttr(), mod.getLayersAttr());

  // Move the body of the module.
  newMod.getRegion().takeBody(mod.getRegion());
  // Update the types in the body.
  builder.setInsertionPointToStart(newMod.getBodyBlock());
  for (auto &arg : newMod.getArguments()) {
    if (mod.getPortDirection(arg.getArgNumber()) == Direction::Out &&
        type_isa<FIRRTLBaseType>(arg.getType())) {
      auto wire = builder.create<WireOp>(arg.getLoc(), arg.getType());
      arg.replaceAllUsesWith(wire.getResult());
      arg.setType(LHSType::get(type_cast<FIRRTLBaseType>(arg.getType())));
      builder.create<StrictConnectOp>(arg.getLoc(), arg, wire.getResult());
    }
  }

  return newMod;
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

  // Let RAUW fail if readers exist and we didn't make a read value.  If This
  // happens (for port or instance), then the solutions is to have made a wire
  // prior to calling this.
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
struct StrictModulesPass : public StrictModulesBase<StrictModulesPass> {
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
               .isPassive) {
                wire.emitWarning("Wire is not passive, skipping.  All wires should be passive by this point in the pipeline.");
        return WalkResult::advance();
               }
      auto newWire = cloneWireTo(builder, wire);
      updateUses(builder, toDelete, wire.getResult(), newWire.getRead(),
                 newWire.getWrite());
      toDelete.push_back(wire);
    } else if (auto inst = dyn_cast<InstanceOp>(op)) {
      for (auto type : inst.getResultTypes()) {
        auto rtype = type_dyn_cast<FIRRTLBaseType>(type);
        if (rtype && !rtype.isPassive()) {
          inst.emitWarning("Instance has non-passive type, skipping.  All instances should have passive types by this point in the pipeline.");
          return WalkResult::advance();
        }
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

// This is the main entrypoint for the lowering pass.
void StrictModulesPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n";);
  auto circuit = getOperation();
  mlir::OpBuilder builder(circuit);
  std::deque<Operation *> toDelete;

  circuit.walk([&](Operation *op) -> WalkResult {
    if (auto module = dyn_cast<FModuleOp>(op)) {
      for (auto port : module.getPorts())
        if (auto type = type_dyn_cast<FIRRTLBaseType>(port.type))
          if (!type.isPassive()) {
            module.emitWarning("Module has non-passive ports, skipping.  All modules should have passive ports by this point in the pipeline.");
            return WalkResult::advance();
          }

      auto newMod = cloneModuleTo(builder, module);
      toDelete.push_back(module);
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

std::unique_ptr<mlir::Pass> circt::firrtl::createStrictModulesPass() {
  return std::make_unique<StrictModulesPass>();
}
