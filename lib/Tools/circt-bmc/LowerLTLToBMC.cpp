//===- LowerLTLToBMC.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers a subset of LTL (clocked delay + implication) into core comb/seq ops
// suitable for circt-bmc. This is intentionally narrow and only supports the
// patterns emitted by Micro-VHDL.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_LOWERLTLTOBMC
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

namespace {
struct LowerLTLToBMCPass
    : public circt::impl::LowerLTLToBMCBase<LowerLTLToBMCPass> {
  void runOnOperation() override;

private:
  int nameID = 0;

  StringAttr nextName(OpBuilder &builder, StringRef prefix) {
    return builder.getStringAttr((prefix + "_" + Twine(nameID++)).str());
  }

  Value createCompReg(OpBuilder &builder, Location loc, Value input, Value clk,
                      StringRef prefix) {
    auto nameAttr = nextName(builder, prefix);
    auto i1 = builder.getI1Type();
    Value initZero = seq::createConstantInitialValue(
        builder, loc, builder.getIntegerAttr(i1, 0));
    return seq::CompRegOp::create(builder, loc, input, clk,
                                  /*reset*/ Value(), /*rstValue*/ Value(),
                                  nameAttr, /*initialValue*/ initZero)
        .getResult();
  }

  Value shift(OpBuilder &builder, Location loc, Value input, Value clk,
              int64_t cycles, StringRef prefix) {
    Value cur = input;
    for (int64_t i = 0; i < cycles; ++i) {
      cur = createCompReg(builder, loc, cur, clk, prefix);
    }
    return cur;
  }

  FailureOr<Value> getSeqClock(Value clkI1) {
    if (isa<seq::ClockType>(clkI1.getType()))
      return clkI1;
    if (auto from = clkI1.getDefiningOp<seq::FromClockOp>())
      return from.getInput();
    return failure();
  }

  FailureOr<Value> lowerImplicationWithDelay(OpBuilder &builder, Location loc,
                                             Value antecedent, ltl::DelayOp delay,
                                             Value clkI1) {
    if (!antecedent.getType().isInteger(1))
      return failure();
    if (!delay.getInput().getType().isInteger(1))
      return failure();

    int64_t start = static_cast<int64_t>(delay.getDelay());
    if (start < 1)
      return failure();

    auto lengthAttr = delay.getLength();
    if (!lengthAttr)
      return failure();
    int64_t length = static_cast<int64_t>(*lengthAttr);
    if (length < 0)
      return failure();
    int64_t end = start + length;

    auto clkOr = getSeqClock(clkI1);
    if (failed(clkOr))
      return failure();
    Value clk = *clkOr;

    // Exact delay case: antecedent delayed by start cycles implies consequent now.
    if (length == 0) {
      Value delayed = shift(builder, loc, antecedent, clk, start, "ltl_delay");
      Value notDelayed = comb::createOrFoldNot(loc, delayed, builder);
      auto orOp = comb::OrOp::create(builder, loc, ValueRange{notDelayed, delay.getInput()}, false);
      return orOp.getResult();
    }

    // Bounded range case: if antecedent fires, consequent must hold at least
    // once in [start, end] cycles. Use a pending-request monitor that clears
    // when consequent is observed.
    Value notRight = comb::createOrFoldNot(loc, delay.getInput(), builder);

    // Age 1 pending is the antecedent delayed by 1 cycle.
    Value pending = shift(builder, loc, antecedent, clk, 1, "ltl_delay");
    Value prev = pending;

    for (int64_t age = 1; age <= end; ++age) {
      Value nextVal;
      if (age == 1) {
        nextVal = prev;
      } else if (age < start) {
        nextVal = prev;
      } else {
        auto andOp = comb::AndOp::create(builder, loc, ValueRange{prev, notRight}, false);
        nextVal = andOp.getResult();
      }
      prev = createCompReg(builder, loc, nextVal, clk, "ltl_window");
    }

    // violation = pending_at_end AND NOT(consequent)
    auto viol = comb::AndOp::create(builder, loc, ValueRange{prev, notRight}, false);
    Value notViol = comb::createOrFoldNot(loc, viol.getResult(), builder);
    return notViol;
  }

  FailureOr<Value> lowerClockedProperty(OpBuilder &builder, Location loc,
                                        Value prop, Value clkI1) {
    if (auto imp = prop.getDefiningOp<ltl::ImplicationOp>()) {
      auto delay = imp.getConsequent().getDefiningOp<ltl::DelayOp>();
      if (!delay)
        return failure();
      return lowerImplicationWithDelay(builder, loc, imp.getAntecedent(), delay,
                                       clkI1);
    }
    if (prop.getType().isInteger(1))
      return prop;
    return failure();
  }
};
} // namespace

void LowerLTLToBMCPass::runOnOperation() {
  hw::HWModuleOp hwModule = getOperation();
  OpBuilder builder(hwModule);
  bool failedLowering = false;

  auto lowerAssertLike = [&](Operation *op, Value prop) {
    auto clkOp = prop.getDefiningOp<ltl::ClockOp>();
    if (!clkOp)
      return;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);

    auto lowered = lowerClockedProperty(builder, op->getLoc(), clkOp.getInput(),
                                        clkOp.getClock());
    if (failed(lowered)) {
      op->emitError("unsupported LTL pattern for circt-bmc lowering");
      failedLowering = true;
      return;
    }
    op->setOperand(0, *lowered);
    if (clkOp->use_empty())
      clkOp.erase();
  };

  hwModule.walk([&](verif::AssertOp op) {
    if (failedLowering)
      return;
    lowerAssertLike(op, op.getProperty());
  });

  hwModule.walk([&](verif::AssumeOp op) {
    if (failedLowering)
      return;
    lowerAssertLike(op, op.getProperty());
  });

  if (failedLowering)
    signalPassFailure();
}
