//===- LTLToCore.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts LTL and Verif operations to Core operations
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LTLToCore.h"
#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {
#define GEN_PASS_DEF_LOWERLTLTOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
struct HasBeenResetOpConversion : OpConversionPattern<verif::HasBeenResetOp> {
  using OpConversionPattern<verif::HasBeenResetOp>::OpConversionPattern;

  // HasBeenReset generates a 1 bit register that is set to one once the reset
  // has been raised and lowered at at least once.
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    // Generate the constant used to set the register value
    Value constZero = seq::createConstantInitialValue(
        rewriter, op->getLoc(), rewriter.getIntegerAttr(i1, 0));

    // Generate the constant used to negate the reset value
    Value constOne = hw::ConstantOp::create(rewriter, op.getLoc(), i1, 1);

    // Create a backedge for the register to be used in the OrOp
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate an or between the reset and the register's value to store
    // whether or not the reset has been active at least once
    Value orReset =
        comb::OrOp::create(rewriter, op.getLoc(), adaptor.getReset(), reg);

    // This register should not be reset, so we give it dummy reset and resetval
    // operands to fit the build signature
    Value reset, resetval;

    // Finally generate the register to set the backedge
    reg.setValue(seq::CompRegOp::create(
        rewriter, op.getLoc(), orReset,
        rewriter.createOrFold<seq::ToClockOp>(op.getLoc(), adaptor.getClock()),
        rewriter.getStringAttr("hbr"), reset, resetval, constZero,
        InnerSymAttr{} // inner_sym
        ));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset = comb::XorOp::create(rewriter, op.getLoc(),
                                         adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
  }
};

struct LTLImplicationConversion
    : public OpConversionPattern<ltl::ImplicationOp> {
  using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean implications to comb ops
    if (!isa<IntegerType>(op.getAntecedent().getType()) ||
        !isa<IntegerType>(op.getConsequent().getType()))
      return failure();
    /// A -> B = !A || B
    auto loc = op.getLoc();
    auto notA = comb::createOrFoldNot(rewriter, loc, adaptor.getAntecedent());
    auto orOp =
        comb::OrOp::create(rewriter, loc, notA, adaptor.getConsequent());
    rewriter.replaceOp(op, orOp);
    return success();
  }
};

struct LTLNotConversion : public OpConversionPattern<ltl::NotOp> {
  using OpConversionPattern<ltl::NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean nots to comb ops
    if (!isa<IntegerType>(op.getInput().getType()))
      return failure();
    auto loc = op.getLoc();
    auto inverted = comb::createOrFoldNot(rewriter, loc, adaptor.getInput());
    rewriter.replaceOp(op, inverted);
    return success();
  }
};

struct LTLAndOpConversion : public OpConversionPattern<ltl::AndOp> {
  using OpConversionPattern<ltl::AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean ands to comb ops
    if (!isa<IntegerType>(op->getOperandTypes()[0]) ||
        !isa<IntegerType>(op->getOperandTypes()[1]))
      return failure();
    auto loc = op.getLoc();
    // Explicit twoState value to disambiguate builders
    auto andOp =
        comb::AndOp::create(rewriter, loc, adaptor.getOperands(), false);
    rewriter.replaceOp(op, andOp);
    return success();
  }
};

struct LTLOrOpConversion : public OpConversionPattern<ltl::OrOp> {
  using OpConversionPattern<ltl::OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean ors to comb ops
    if (!isa<IntegerType>(op->getOperandTypes()[0]) ||
        !isa<IntegerType>(op->getOperandTypes()[1]))
      return failure();
    auto loc = op.getLoc();
    // Explicit twoState value to disambiguate builders
    auto orOp = comb::OrOp::create(rewriter, loc, adaptor.getOperands(), false);
    rewriter.replaceOp(op, orOp);
    return success();
  }
};

struct LTLIntersectOpConversion : public OpConversionPattern<ltl::IntersectOp> {
  using OpConversionPattern<ltl::IntersectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::IntersectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean intersects to comb ops; booleans are
    // instantaneous matches, so intersection is conjunction.
    if (!isa<IntegerType>(op->getOperandTypes()[0]) ||
        !isa<IntegerType>(op->getOperandTypes()[1]))
      return failure();
    auto loc = op.getLoc();
    // Explicit twoState value to disambiguate builders
    auto andOp =
        comb::AndOp::create(rewriter, loc, adaptor.getOperands(), false);
    rewriter.replaceOp(op, andOp);
    return success();
  }
};

struct LTLPastOpConversion : public OpConversionPattern<ltl::PastOp> {
  using OpConversionPattern<ltl::PastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::PastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value clock =
        seq::ToClockOp::create(rewriter, op.getLoc(), adaptor.getClk());
    Value cur = adaptor.getInput();
    Value ce =
        hw::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI1Type(), 1);
    auto shiftreg =
        seq::ShiftRegOp::create(rewriter, op.getLoc(), op.getDelayAttr(), cur,
                                clock, ce, {}, {}, {}, {}, {});
    rewriter.replaceOp(op, shiftreg);
    return success();
  }
};

// Sample `input` on the requested clock edge.
static Value createRegister(Value input, Value clock, ltl::ClockEdge edge,
                            bool initialValue, OpBuilder &builder,
                            Operation *contextOp) {
  assert(edge != ltl::ClockEdge::Both && "both-edge clock not supported");
  auto clockSignal = clock;
  if (edge == ltl::ClockEdge::Neg)
    clockSignal =
        comb::createOrFoldNot(builder, contextOp->getLoc(), clockSignal);
  auto seqClock =
      builder.createOrFold<seq::ToClockOp>(contextOp->getLoc(), clockSignal);

  auto loc = contextOp->getLoc();
  auto initial = seq::createConstantInitialValue(
      builder, loc, builder.getIntegerAttr(builder.getI1Type(), initialValue));
  return seq::CompRegOp::create(builder, loc, input, seqClock,
                                /*reset=*/Value{},
                                /*rstValue=*/Value{}, initial)
      .getResult();
}

static void lowerTemporalLTLToCore(hw::HWModuleOp module) {
  SmallVector<Operation *> assertionsAndAssumptions;
  module->walk([&](Operation *op) {
    if (isa<verif::AssertOp, verif::AssumeOp>(op))
      assertionsAndAssumptions.push_back(op);
  });

  for (auto *op : assertionsAndAssumptions) {
    Value property = op->getOperand(0);
    // Only lower a clocked atom when it is the direct property of an assertion
    // or assumption. The startup `dontCare` guard is property-level and cannot
    // be composed correctly through nested LTL operations.
    auto atom = property.getDefiningOp<ltl::ClockedAtomOp>();
    if (!atom || atom.getEdge() == ltl::ClockEdge::Both)
      continue;

    OpBuilder builder(op);
    auto sampled =
        createRegister(atom.getInput(), atom.getClock(), atom.getEdge(),
                       /*initialValue=*/false, builder, atom);

    auto constFalse =
        hw::ConstantOp::create(builder, op->getLoc(), builder.getI1Type(), 0);
    // `dontCare` is true while `sampled` only contains its initial value. The
    // first real clock edge clears it and makes the sampled value observable.
    auto dontCare = createRegister(constFalse, atom.getClock(), atom.getEdge(),
                                   /*initialValue=*/true, builder, op);
    auto guardedProperty =
        comb::OrOp::create(builder, op->getLoc(), dontCare, sampled,
                           /*twoState=*/false);
    op->setOperand(0, guardedProperty);
  }
}
} // namespace

// Converts a `verif` clock edge into the equivalent `ltl` clock edge.
static ltl::ClockEdge verifToLTLClockEdge(verif::ClockEdge edge) {
  switch (edge) {
  case verif::ClockEdge::Pos:
    return ltl::ClockEdge::Pos;
  case verif::ClockEdge::Neg:
    return ltl::ClockEdge::Neg;
  case verif::ClockEdge::Both:
    return ltl::ClockEdge::Both;
  }
  llvm_unreachable("unknown clock edge");
}

// Matches the antecedent of a non-overlapping implication, `a[##N] |=> b` in
// SVA, and returns the boolean `a` together with the number of delay cycles N.
// We match 3 specific forms of antecedent:
//
// - simple boolean `a` (N is implicitly 0)
// - `concat(delay(a, N, 0), true)`, as emitted by ImportVerilog
// - `concat(a, delay(true, N, 0))`, as recognized by ExportVerilog
static std::optional<std::pair<Value, uint64_t>>
matchAntecedent(Value antecedent) {
  // match a simple bool antecedent
  if (antecedent.getType().isInteger(1))
    return std::make_pair(antecedent, uint64_t(0));

  // only other supported antecedents are in the form of a concat()
  auto concatOp = antecedent.getDefiningOp<ltl::ConcatOp>();
  if (!concatOp || concatOp.getInputs().size() != 2)
    return std::nullopt;
  Value lhs = concatOp.getInputs()[0];
  Value rhs = concatOp.getInputs()[1];

  // helper to match delays of at least one cycle
  auto getExactDelay = [](Value value) -> ltl::DelayOp {
    auto delayOp = value.getDefiningOp<ltl::DelayOp>();
    if (!delayOp || delayOp.getLength() != 0 || delayOp.getDelay() < 1)
      return {};
    return delayOp;
  };
  // helper to match constant true boolean.
  auto isConstantTrue = [](Value value) -> bool {
    auto constOp = value.getDefiningOp<hw::ConstantOp>();
    return constOp && constOp.getValue().isOne();
  };

  // match `concat(delay(a, N, 0), true)`
  if (auto delayOp = getExactDelay(lhs))
    if (isConstantTrue(rhs) && delayOp.getInput().getType().isInteger(1))
      return std::make_pair(delayOp.getInput(), delayOp.getDelay());
  // match `concat(a, delay(true, N, 0))`
  if (auto delayOp = getExactDelay(rhs))
    if (isConstantTrue(delayOp.getInput()) && lhs.getType().isInteger(1))
      return std::make_pair(lhs, delayOp.getDelay());

  return std::nullopt;
}

namespace {
// A consequent that is satisfied if `now` holds in the cycle it starts, or if
// `later` holds in any of the cycles `[lo, lo + len]` after that. `now` may be
// null.
struct BoundedConsequent {
  Value now;
  Value later;
  uint64_t lo = 0;
  uint64_t len = 0;
};
} // namespace

// Matches a consequent made of booleans and bounded delays:
//
// - `b`
// - `delay(b, lo, len)`, `##[lo:lo+len] b` in SVA
// - `or(x, delay(b, lo, len))` or `or(delay(b, lo, len), x)`
static std::optional<BoundedConsequent> matchConsequent(Value consequent) {
  // match `b`
  if (consequent.getType().isInteger(1))
    return BoundedConsequent{{}, consequent, 0, 0};

  auto matchBoundedDelay = [](Value value) -> std::optional<BoundedConsequent> {
    auto delayOp = value.getDefiningOp<ltl::DelayOp>();
    if (!delayOp || !delayOp.getLength() ||
        !delayOp.getInput().getType().isInteger(1))
      return std::nullopt;
    return BoundedConsequent{
        {}, delayOp.getInput(), delayOp.getDelay(), *delayOp.getLength()};
  };

  // match `delay(b, lo, len)`, `##[lo:lo+len] b`
  if (auto result = matchBoundedDelay(consequent))
    return result;

  auto orOp = consequent.getDefiningOp<ltl::OrOp>();
  if (!orOp || orOp.getInputs().size() != 2)
    return std::nullopt;
  Value first = orOp.getInputs()[0];
  Value second = orOp.getInputs()[1];

  auto matchOrWithDelayed =
      [&](Value now, Value delayed) -> std::optional<BoundedConsequent> {
    if (!now.getType().isInteger(1))
      return std::nullopt;
    if (auto result = matchBoundedDelay(delayed)) {
      result->now = now;
      return result;
    }
    return std::nullopt;
  };
  // match `or(x, delay(b, lo, len))`
  if (auto result = matchOrWithDelayed(first, second))
    return result;
  // match `or(delay(b, lo, len), x)`
  if (auto result = matchOrWithDelayed(second, first))
    return result;

  return std::nullopt;
}

// Lowers clocked assertions and assumptions whose property is an implication
// with a bounded temporal extent into a register chain and comb logic. The
// antecedent is a boolean `a`, optionally delayed by `d` cycles as in
// `a[##d] |=> b`, and the consequent is a boolean `b`, optionally delayed
// as in `or(now, ##[lo:lo+len] b)`.
// Simple bounded delays are treated as an implication with a true antecedent.
//
// The lowering tracks pending obligations through a chain of registers. An
// obligation is created when `a` holds, becomes due `d` cycles later, and is
// discharged by `now` in that cycle or by `b` (later) in any of the
// `[lo, lo+len]` cycles after it:
//
// If the op has an enable, which is how `disable iff` is represented, every
// register samples its input only while the enable holds. This cancels an
// obligation if the enable drops in any cycle between its creation and its
// check.
static void lowerDelayedImplications(hw::HWModuleOp module) {
  SmallVector<Operation *> clockedOps;
  module->walk([&](Operation *op) {
    if (isa<verif::ClockedAssertOp, verif::ClockedAssumeOp>(op))
      clockedOps.push_back(op);
  });

  for (auto *op : clockedOps) {
    verif::ClockEdge edge;
    Value clock, enable;
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<verif::ClockedAssertOp, verif::ClockedAssumeOp>(
            [&](auto clockedOp) {
              edge = clockedOp.getEdge();
              clock = clockedOp.getClock();
              enable = clockedOp.getEnable();
            })
        .Default([](Operation *) {
          llvm_unreachable("Only clocked assert and assume supported.");
        });
    if (edge == verif::ClockEdge::Both)
      continue;

    // Split the property into an antecedent and a consequent.
    Value property = op->getOperand(0);
    Value trigger;
    uint64_t delay = 0;
    std::optional<BoundedConsequent> consequent;
    if (auto implicationOp = property.getDefiningOp<ltl::ImplicationOp>()) {
      auto antecedent = matchAntecedent(implicationOp.getAntecedent());
      if (!antecedent)
        continue;
      std::tie(trigger, delay) = *antecedent;
      consequent = matchConsequent(implicationOp.getConsequent());
    } else if (property.getDefiningOp<ltl::DelayOp>()) {
      consequent = matchConsequent(property);
    } else {
      continue;
    }
    if (!consequent)
      continue;

    // Boolean implications without any delay are lowered to comb ops by the
    // conversion patterns.
    if (delay == 0 && consequent->lo == 0 && consequent->len == 0 &&
        !consequent->now)
      continue;

    OpBuilder builder(op);
    auto loc = op->getLoc();
    if (!trigger)
      trigger = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);

    // Delay a value by one cycle, cancelling it if the enable drops.
    auto delayByOne = [&](Value value) {
      if (enable)
        value = comb::AndOp::create(builder, loc, value, enable, false);
      return createRegister(value, clock, verifToLTLClockEdge(edge),
                            /*initialValue=*/false, builder, op);
    };
    auto andNot = [&](Value value, Value discharge) {
      return comb::AndOp::create(builder, loc, value,
                                 comb::createOrFoldNot(builder, loc, discharge),
                                 false);
    };

    // Create the obligation and move it to the cycle where it becomes due.
    Value pending = trigger;
    for (uint64_t i = 0; i < delay; ++i)
      pending = delayByOne(pending);
    if (consequent->now)
      pending = andNot(pending, consequent->now);
    for (uint64_t i = 0; i < consequent->lo; ++i)
      pending = delayByOne(pending);

    // Keep the obligation pending for each cycle of the window in which
    // `later` does not hold.
    for (uint64_t i = 0; i < consequent->len; ++i)
      pending = delayByOne(andNot(pending, consequent->later));

    // The obligation must be discharged by the last cycle of the window.
    Value notPending = comb::createOrFoldNot(builder, loc, pending);
    Value lowered = comb::OrOp::create(builder, loc, notPending,
                                       consequent->later, /*twoState=*/false);
    op->setOperand(0, lowered);
  }
}

//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerLTLToCorePass
    : public circt::impl::LowerLTLToCoreBase<LowerLTLToCorePass> {
  LowerLTLToCorePass() = default;
  void runOnOperation() override;
};
} // namespace

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {
  lowerTemporalLTLToCore(getOperation());
  lowerDelayedImplications(getOperation());

  // Preserve operations that require an LTL-aware downstream backend.
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<ltl::LTLDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addIllegalOp<verif::HasBeenResetOp>();
  target.addIllegalOp<ltl::PastOp>();

  auto isLegal = [](Operation *op) {
    auto hasNonAssertUsers = std::any_of(
        op->getUsers().begin(), op->getUsers().end(), [](Operation *user) {
          return !isa<verif::AssertOp, verif::ClockedAssertOp>(user);
        });
    auto hasIntegerResultTypes =
        std::all_of(op->getResultTypes().begin(), op->getResultTypes().end(),
                    [](Type type) { return isa<IntegerType>(type); });
    // If there are users other than asserts, we can't map it to comb (unless
    // the return type is already integer anyway)
    if (hasNonAssertUsers && !hasIntegerResultTypes)
      return true;

    // Otherwise illegal if operands are i1
    return std::any_of(
        op->getOperands().begin(), op->getOperands().end(),
        [](Value operand) { return !isa<IntegerType>(operand.getType()); });
  };
  target.addDynamicallyLegalOp<ltl::ImplicationOp>(isLegal);
  target.addDynamicallyLegalOp<ltl::NotOp>(isLegal);
  target.addDynamicallyLegalOp<ltl::AndOp>(isLegal);
  target.addDynamicallyLegalOp<ltl::OrOp>(isLegal);
  target.addDynamicallyLegalOp<ltl::IntersectOp>(isLegal);

  // Create type converters, mostly just to convert an ltl property to a bool
  mlir::TypeConverter converter;

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });

  // Basic materializations
  converter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  patterns.add<HasBeenResetOpConversion, LTLImplicationConversion,
               LTLNotConversion, LTLAndOpConversion, LTLOrOpConversion,
               LTLIntersectOpConversion, LTLPastOpConversion>(
      converter, patterns.getContext());
  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  // Clean up remaining unrealized casts by changing assert argument types
  getOperation().walk([&](Operation *op) {
    if (!isa<verif::AssertOp, verif::ClockedAssertOp, verif::AssumeOp,
             verif::ClockedAssumeOp>(op))
      return;
    Value prop = op->getOperand(0);
    if (auto cast = prop.getDefiningOp<UnrealizedConversionCastOp>()) {
      // Make sure that the cast is from an i1, not something random that was
      // in the input
      if (auto intType = dyn_cast<IntegerType>(cast.getOperandTypes()[0]);
          intType && intType.getWidth() == 1)
        op->setOperand(0, cast.getInputs()[0]);
    }
  });
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
