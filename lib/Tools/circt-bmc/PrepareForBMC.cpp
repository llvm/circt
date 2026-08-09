//===- PrepareForBMC.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_PREPAREFORBMC
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

namespace {
struct PrepareForBMCPass
    : public circt::impl::PrepareForBMCBase<PrepareForBMCPass> {
  using PrepareForBMCBase::PrepareForBMCBase;

  void normalizeClockPorts(hw::HWModuleOp module) {
    // SystemVerilog import represents clocks as i1 ports converted by
    // seq.to_clock. BMC needs a native clock block argument so it can own the
    // clock waveform and update registers only on rising edges.
    auto *body = module.getBodyBlock();
    SmallVector<BlockArgument> clockArguments;
    for (auto argument : body->getArguments()) {
      if (!argument.getType().isInteger(1))
        continue;
      if (llvm::any_of(argument.getUsers(), [](Operation *user) {
            return isa<seq::ToClockOp>(user);
          }))
        clockArguments.push_back(argument);
    }
    if (clockArguments.empty())
      return;

    SmallVector<hw::ModulePort> ports(module.getHWModuleType().getPorts());
    auto clockType = seq::ClockType::get(&getContext());
    for (auto argument : clockArguments) {
      SmallVector<seq::ToClockOp> toClockOps;
      SmallVector<OpOperand *> rawClockUses;
      for (auto &use : argument.getUses()) {
        if (auto toClock = dyn_cast<seq::ToClockOp>(use.getOwner()))
          toClockOps.push_back(toClock);
        else
          rawClockUses.push_back(&use);
      }

      argument.setType(clockType);
      OpBuilder builder = OpBuilder::atBlockBegin(body);
      auto rawClock =
          seq::FromClockOp::create(builder, argument.getLoc(), argument);
      for (auto *use : rawClockUses)
        use->set(rawClock);
      for (auto toClock : toClockOps) {
        toClock.replaceAllUsesWith(argument);
        toClock.erase();
      }

      auto portID =
          module.getHWModuleType().getPortIdForInputId(argument.getArgNumber());
      ports[portID].type = clockType;
    }
    module.setHWModuleType(hw::ModuleType::get(&getContext(), ports));
  }

  void runOnOperation() override {
    auto module = getOperation().lookupSymbol<hw::HWModuleOp>(topModule);
    if (!module)
      return;

    normalizeClockPorts(module);
  }
};
} // namespace
