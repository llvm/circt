//===- PassiveWires.cpp - Make Wires Passive --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassiveWires pass.  This pass eliminated flips from
// wires with aggregate types.  Since flips only determine connect direction,
// they are unnecessary on wires and just get in the way.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-passive-wires"

using namespace circt;
using namespace firrtl;

static bool hasFlip(Type t) {
  if (auto type = type_dyn_cast<FIRRTLBaseType>(t))
    return !type.isPassive();
  return false;
}

static void updateWireUses(mlir::OpBuilder& builder, Operation* op, Value readVal, Value writeVal) {
  llvm::errs() << "updateWireUses\n";
  op->dump();
  readVal.dump();
  writeVal.dump();
  for (auto* user : op->getUsers())
    user->dump();
  llvm::errs() << "\n";

  for (auto* user : llvm::make_early_inc_range(op->getUsers())) {
    llvm::TypeSwitch<Operation*, void>(user)
    .Case<StrictConnectOp, ConnectOp>([&](auto con){
      if (con.getDest() == op->getResult(0))
        con->setOperand(0, writeVal);
      if (con.getSrc() == op->getResult(0))
        con->setOperand(1, readVal);
    })
    .Case<SubindexOp>([&](auto index){
      builder.setInsertionPointAfter(index);
      auto newReadVal = builder.create<SubindexOp>(index.getLoc(), readVal, index.getIndex());
      auto newWriteVal = builder.create<SubindexOp>(index.getLoc(), writeVal, index.getIndex());
      updateWireUses(builder, index, newReadVal, newWriteVal);
      index.erase();
      if (newReadVal.getResult().use_empty())
        newReadVal.erase();
      if (newWriteVal.getResult().use_empty())
        newWriteVal.erase();      
    })
    .Case<SubfieldOp>([&](auto index){
      builder.setInsertionPointAfter(index);
      auto newReadVal = builder.create<SubfieldOp>(index.getLoc(), readVal, index.getFieldIndex());
      auto newWriteVal = builder.create<SubfieldOp>(index.getLoc(), writeVal, index.getFieldIndex());
      updateWireUses(builder, index, newReadVal, newWriteVal);
      index.erase();
      if (newReadVal.getResult().use_empty())
        newReadVal.erase();
      if (newWriteVal.getResult().use_empty())
        newWriteVal.erase();
    })
    .Case<SubaccessOp>([&](auto index){
      builder.setInsertionPointAfter(index);
      auto newReadVal = builder.create<SubaccessOp>(index.getLoc(), readVal, index.getIndex());
      auto newWriteVal = builder.create<SubaccessOp>(index.getLoc(), writeVal, index.getIndex());
      updateWireUses(builder, index, newReadVal, newWriteVal);
      index.erase();
      if (newReadVal.getResult().use_empty())
        newReadVal.erase();
      if (newWriteVal.getResult().use_empty())
        newWriteVal.erase();      
    })
    .Default([&](auto v) {
      llvm::errs() << "Default\n";
      v->dump();
      for (auto idx = 0U, e = v->getNumOperands(); idx < e; ++idx)
        if (v->getOperand(idx) == op->getResult(0))
          v->setOperand(idx, readVal);
      v->dump();
    });
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct PassiveWiresPass : public PassiveWiresBase<PassiveWiresPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void PassiveWiresPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n";);
  auto module = getOperation();

  // First, expand any connects to resolve flips.
  SmallVector<Operation *> worklist;
  SmallVector<WireOp> allWires;
  module.walk([&](Operation *op) -> WalkResult {
    if (auto wire = dyn_cast<WireOp>(op)) {
      allWires.push_back(wire);
      if (hasFlip(wire.getType(0)))
        worklist.push_back(wire);
      return WalkResult::advance();
    }
    if (!isa<ConnectOp, StrictConnectOp>(op))
      return WalkResult::advance();
    // connect/strictconnect
    if (!hasFlip(op->getOperand(0).getType()))
      return WalkResult::advance();

    mlir::ImplicitLocOpBuilder builder(op->getLoc(), op);
    // This will "blow out" a connect to passive pieces
    emitConnect(builder, op->getOperand(0), op->getOperand(1));
    op->erase();
    return WalkResult::advance();
  });

  // Second, remove flips from most things.
  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();
    auto r = op->getResult(0);
    if (!hasFlip(r.getType()))
      continue;
    for (auto users : r.getUsers())
      worklist.push_back(users);
    // In-place updates is safe as consumers don't care about flip.
    r.setType(type_cast<FIRRTLBaseType>(r.getType()).getPassiveType());
  }

  mlir::OpBuilder builder(module.getContext());
  // Finally, convert the wires to the strict form.
  for (auto wire : allWires) {
    builder.setInsertionPointAfter(wire);
    StrictWireOp newWire;
    if (wire.getRef()) {
      newWire = builder.create<StrictWireOp>(wire.getLoc(), wire.getDataType(), LHSType::get(module.getContext(), wire.getDataType()), wire.getRef().getType(),
    wire.getName(), wire.getNameKind(), wire.getAnnotationsAttr(), wire.getInnerSymAttr(), wire.getForceable());
    wire.getRef().replaceAllUsesWith(newWire.getRef());
    } else {
      newWire = builder.create<StrictWireOp>(wire.getLoc(), wire.getDataType(), LHSType::get(module.getContext(), wire.getDataType()), Type(), wire.getName(), wire.getNameKind(), wire.getAnnotationsAttr(), wire.getInnerSymAttr(), wire.getForceable());
    }
    updateWireUses(builder, wire, newWire.getResult(), newWire.getWriteport());
    wire.erase();
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createPassiveWiresPass() {
  return std::make_unique<PassiveWiresPass>();
}
