//===- PrepareForBMC.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
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

  FailureOr<Value> getPreviousClock(Value clock, OpBuilder &builder) {
    if (auto it = previousClockValues.find(clock);
        it != previousClockValues.end())
      return it->second;

    auto fromClock = clock.getDefiningOp<seq::FromClockOp>();
    if (!fromClock) {
      emitError(clock.getLoc(),
                "expected a clock normalized by seq.from_clock");
      return failure();
    }

    auto initialValue = seq::createConstantInitialValue(
        builder, clock.getLoc(),
        builder.getIntegerAttr(builder.getI1Type(), 0));
    // ExternalizeRegisters turns this into BMC state whose next value is the
    // current clock, giving every property the clock value from the preceding
    // BMC transition.
    auto previousClock = seq::CompRegOp::create(
        builder, clock.getLoc(), clock, fromClock.getInput(), /*reset=*/Value{},
        /*rstValue=*/Value{}, initialValue);
    previousClockValues.try_emplace(clock, previousClock);
    return previousClock.getData();
  }

  template <typename ClockedOp, typename UnclockedOp>
  LogicalResult lowerClockedProperty(ClockedOp op) {
    if (!op.getProperty().getType().isInteger(1)) {
      op.emitError("unsupported clocked property after LTL lowering");
      return failure();
    }

    OpBuilder builder(op);
    auto previousClock = getPreviousClock(op.getClock(), builder);
    if (failed(previousClock))
      return failure();
    auto trueValue =
        hw::ConstantOp::create(builder, op.getLoc(), builder.getI1Type(), 1);
    Value active;
    switch (op.getEdge()) {
    case verif::ClockEdge::Pos: {
      auto notPreviousClock =
          comb::XorOp::create(builder, op.getLoc(), *previousClock, trueValue);
      active = comb::AndOp::create(builder, op.getLoc(), op.getClock(),
                                   notPreviousClock);
      break;
    }
    case verif::ClockEdge::Neg: {
      auto notCurrentClock =
          comb::XorOp::create(builder, op.getLoc(), op.getClock(), trueValue);
      active = comb::AndOp::create(builder, op.getLoc(), notCurrentClock,
                                   *previousClock);
      break;
    }
    case verif::ClockEdge::Both:
      active = comb::XorOp::create(builder, op.getLoc(), op.getClock(),
                                   *previousClock);
      break;
    }
    if (op.getEnable())
      active =
          comb::AndOp::create(builder, op.getLoc(), active, op.getEnable());

    auto inactive =
        comb::XorOp::create(builder, op.getLoc(), active, trueValue);
    auto property =
        comb::OrOp::create(builder, op.getLoc(), inactive, op.getProperty());
    UnclockedOp::create(builder, op.getLoc(), property, /*enable=*/Value{},
                        op.getLabelAttr());
    op.erase();
    return success();
  }

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
    previousClockValues.clear();
    auto module = getOperation().lookupSymbol<hw::HWModuleOp>(topModule);
    if (!module)
      return;

    normalizeClockPorts(module);
    LogicalResult result = success();
    module->walk([&](Operation *operation) {
      if (failed(result))
        return;
      if (auto assertOp = dyn_cast<verif::ClockedAssertOp>(operation))
        result = lowerClockedProperty<verif::ClockedAssertOp, verif::AssertOp>(
            assertOp);
      else if (auto assumeOp = dyn_cast<verif::ClockedAssumeOp>(operation))
        result = lowerClockedProperty<verif::ClockedAssumeOp, verif::AssumeOp>(
            assumeOp);
    });
    if (failed(result))
      signalPassFailure();
  }

  DenseMap<Value, Value> previousClockValues;
};
} // namespace
