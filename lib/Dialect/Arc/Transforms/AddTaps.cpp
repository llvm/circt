//===- AddTaps.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_ADDTAPS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct AddTapsPass : public arc::impl::AddTapsBase<AddTapsPass> {
  using AddTapsBase::AddTapsBase;

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<HWModuleOp, sv::WireOp, hw::WireOp>([&](auto op) { tap(op); })
          .Default([&](auto) { tapIfNamed(op); });
    });
  }

  // Add taps for all module ports.
  void tap(HWModuleOp moduleOp) {
    if (!tapPorts)
      return;
    auto *outputOp = moduleOp.getBodyBlock()->getTerminator();
    ModulePortInfo ports(moduleOp.getPortList());

    // Add taps to inputs.
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBodyBlock());
    for (auto [port, arg] :
         llvm::zip(ports.getInputs(), moduleOp.getBodyBlock()->getArguments()))
      buildTap(builder, arg.getLoc(), arg, port.getName());

    // Add taps to outputs.
    builder.setInsertionPoint(outputOp);
    for (auto [port, result] :
         llvm::zip(ports.getOutputs(), outputOp->getOperands()))
      buildTap(builder, result.getLoc(), result, port.getName());
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
    if (!readOp)
      readOp = sv::ReadInOutOp::create(builder, wireOp.getLoc(), wireOp);
    buildTap(builder, readOp.getLoc(), readOp, wireOp.getName());
  }

  // Add taps for HW wires.
  void tap(hw::WireOp wireOp) {
    if (auto name = wireOp.getName(); name && tapWires) {
      OpBuilder builder(wireOp);
      buildTap(builder, wireOp.getLoc(), wireOp, *name);
    }
    wireOp.getResult().replaceAllUsesWith(wireOp.getInput());
    wireOp->erase();
  }

  // Add taps for named values.
  void tapIfNamed(Operation *op) {
    if (!tapNamedValues || op->getNumResults() != 1)
      return;
    if (auto name = op->getAttrOfType<StringAttr>("sv.namehint")) {
      OpBuilder builder(op);
      buildTap(builder, op->getLoc(), op->getResult(0), name);
    }
  }

  void buildTap(OpBuilder &builder, Location loc, Value value, StringRef name) {
    if (name.empty())
      return;
    if (isa<seq::ClockType>(value.getType()))
      value = builder.createOrFold<seq::FromClockOp>(loc, value);
    arc::TapOp::create(builder, loc, value, name);
  }
};
} // namespace

std::unique_ptr<Pass> arc::createAddTapsPass(const AddTapsOptions &options) {
  return std::make_unique<AddTapsPass>(options);
}
