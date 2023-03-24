//===- AddTaps.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/SV/SVOps.h"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::Optional;

namespace {
struct AddTapsPass : public AddTapsBase<AddTapsPass> {
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      TypeSwitch<Operation *>(op).Case<HWModuleOp, sv::WireOp, hw::WireOp>(
          [&](auto op) { tap(op); });
    });
  }

  // Add taps for all module ports.
  void tap(HWModuleOp moduleOp) {
    if (!tapPorts)
      return;
    auto *outputOp = moduleOp.getBodyBlock()->getTerminator();
    ModulePortInfo ports = moduleOp.getPorts();

    // Add taps to inputs.
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBodyBlock());
    for (auto [port, arg] : llvm::zip(ports.inputs, moduleOp.getArguments()))
      builder.create<arc::TapOp>(arg.getLoc(), arg, port.getName());

    // Add taps to outputs.
    builder.setInsertionPoint(outputOp);
    for (auto [port, result] :
         llvm::zip(ports.outputs, outputOp->getOperands()))
      builder.create<arc::TapOp>(result.getLoc(), result, port.getName());
  }

  // Add taps for SV wires.
  void tap(sv::WireOp wireOp) {
    if (!tapWires)
      return;
    sv::ReadInOutOp readOp;
    for (auto *user : wireOp->getUsers())
      if (auto op = dyn_cast<sv::ReadInOutOp>(user))
        readOp = op;

    OpBuilder builder(wireOp);
    if (!readOp) {
      builder.setInsertionPointAfter(wireOp);
      readOp = builder.create<sv::ReadInOutOp>(wireOp.getLoc(), wireOp);
    }

    builder.setInsertionPointAfter(readOp);
    builder.create<arc::TapOp>(readOp.getLoc(), readOp, wireOp.getName());
  }

  // Add taps for HW wires.
  void tap(hw::WireOp wireOp) {
    if (!tapWires)
      return;
    if (auto name = wireOp.getName()) {
      OpBuilder builder(wireOp);
      builder.setInsertionPointAfter(wireOp);
      builder.create<arc::TapOp>(wireOp.getLoc(), wireOp, *name);
    }
  }

  using AddTapsBase::tapPorts;
  using AddTapsBase::tapWires;
};
} // namespace

std::unique_ptr<Pass> arc::createAddTapsPass(Optional<bool> tapPorts,
                                             Optional<bool> tapWires) {
  auto pass = std::make_unique<AddTapsPass>();
  if (tapPorts)
    pass->tapPorts = *tapPorts;
  if (tapWires)
    pass->tapWires = *tapWires;
  return pass;
}
