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
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

#include <tuple>

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
    // A -> B = !A || B
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

struct TimingStep {
  Value clock;
  ltl::ClockEdge edge;
  uint64_t cycles;

  bool operator==(const TimingStep &other) const {
    return clock == other.clock && edge == other.edge && cycles == other.cycles;
  }
};

// A timing path records the ordered clock events between the start and end of
// a sequence. For example:
//   [(clockA, posedge, 2), (clockB, negedge, 1)]
// represents:
//   clockA.posedge -> clockA.posedge -> clockB.negedge
// Adjacent steps always use different clock events; repeated adjacent events
// are coalesced into the step's cycle count.
using TimingPath = SmallVector<TimingStep>;

// Validity lists the clock histories required before a lowered value can be
// trusted. Each path is materialized by delaying a constant one, and all paths
// must be ready:
//   valid = AND(createPast(1, path) for path in validity)
// The lowered property is checked as `!valid || property`.
using Validity = SmallVector<TimingPath>;

struct SequenceMatch {
  TimingPath timing;
  Value value;
  Validity validity;
};

// A lowered sequence is evaluated at its end point. Each match records the
// clock events from the sequence start to that end point and the i1 signal
// which indicates a match there. The validity requirements track the sampled
// history needed by that signal.
using LoweredSequence = SmallVector<SequenceMatch>;

// A lowered property is evaluated after its timing path has elapsed.
struct LoweredProperty {
  TimingPath timing;
  Value value;
  Validity validity;
};

struct SampledAtom {
  Value value;
  Validity validity;
};

using SampledAtoms = DenseMap<Value, SampledAtom>;

// Cache one-cycle registers keyed by input and clock event so independently
// lowered timing paths share the same history registers.
using PastValues = DenseMap<std::tuple<Value, Value, unsigned>, Value>;

// Sample `input` on the requested clock edge with an initial value of false.
static Value createRegister(Value input, Value clock, ltl::ClockEdge edge,
                            OpBuilder &builder, Operation *contextOp) {
  assert(edge != ltl::ClockEdge::Both && "both-edge clock not supported");
  auto clockSignal = clock;
  if (edge == ltl::ClockEdge::Neg)
    clockSignal =
        comb::createOrFoldNot(builder, contextOp->getLoc(), clockSignal);
  auto seqClock =
      builder.createOrFold<seq::ToClockOp>(contextOp->getLoc(), clockSignal);

  auto loc = contextOp->getLoc();
  auto initial = seq::createConstantInitialValue(
      builder, loc, builder.getIntegerAttr(builder.getI1Type(), 0));
  return seq::CompRegOp::create(builder, loc, input, seqClock,
                                /*reset=*/Value{},
                                /*rstValue=*/Value{}, initial)
      .getResult();
}

// Delay `input` along a timing path by inserting one register per elapsed
// clock event.
static Value createPast(PastValues &pastValues, Value input,
                        ArrayRef<TimingStep> timing, OpBuilder &builder,
                        Operation *contextOp) {
  for (auto step : timing) {
    for (uint64_t i = 0; i < step.cycles; ++i) {
      auto [it, inserted] = pastValues.try_emplace(
          std::make_tuple(input, step.clock, static_cast<unsigned>(step.edge)));
      if (inserted)
        it->second =
            createRegister(input, step.clock, step.edge, builder, contextOp);
      input = it->second;
    }
  }
  return input;
}

// Append clock events to a timing path, coalescing adjacent events on the same
// clock edge.
static LogicalResult appendTiming(TimingPath &timing, Value clock,
                                  ltl::ClockEdge edge, uint64_t cycles) {
  if (cycles == 0)
    return success();
  if (timing.empty() || timing.back().clock != clock ||
      timing.back().edge != edge) {
    timing.push_back({clock, edge, cycles});
    return success();
  }
  auto combined =
      llvm::checkedAddUnsigned<uint64_t>(timing.back().cycles, cycles);
  if (!combined)
    return failure();
  timing.back().cycles = *combined;
  return success();
}

// Append a timing suffix, preserving the coalesced timing path form.
static LogicalResult appendTiming(TimingPath &timing,
                                  ArrayRef<TimingStep> suffix) {
  for (auto step : suffix)
    if (failed(appendTiming(timing, step.clock, step.edge, step.cycles)))
      return failure();
  return success();
}

// Return `suffix` such that:
//   prefix + suffix == timing
// where `+` denotes normalized timing-path concatenation. Return std::nullopt
// if `prefix` is not a prefix of `timing`. A prefix may end in the middle of
// the next timing step, but not after it.
static std::optional<TimingPath> getTimingSuffix(ArrayRef<TimingStep> prefix,
                                                 ArrayRef<TimingStep> timing) {
  if (prefix.empty())
    return TimingPath(timing);

  size_t prefixIndex = 0;
  size_t timingIndex = 0;
  while (prefixIndex < prefix.size()) {
    if (timingIndex == timing.size() ||
        prefix[prefixIndex].clock != timing[timingIndex].clock ||
        prefix[prefixIndex].edge != timing[timingIndex].edge)
      return std::nullopt;

    auto prefixCycles = prefix[prefixIndex].cycles;
    auto timingCycles = timing[timingIndex].cycles;
    if (prefixCycles > timingCycles)
      return std::nullopt;
    if (prefixCycles < timingCycles) {
      if (prefixIndex + 1 != prefix.size())
        return std::nullopt;
      TimingPath suffix;
      suffix.push_back({timing[timingIndex].clock, timing[timingIndex].edge,
                        timingCycles - prefixCycles});
      auto remaining = timing.drop_front(timingIndex + 1);
      suffix.append(remaining.begin(), remaining.end());
      return suffix;
    }
    ++prefixIndex;
    ++timingIndex;
  }
  return TimingPath(timing.drop_front(timingIndex));
}

// Move every validity requirement forward by the same timing suffix. If a
// match moves by `suffix`, each requirement `path` becomes `path + suffix`.
static LogicalResult appendTiming(Validity &validity,
                                  ArrayRef<TimingStep> suffix) {
  for (auto &timing : validity)
    if (failed(appendTiming(timing, suffix)))
      return failure();
  return success();
}

// Add a validity requirement, dropping requirements implied by stronger ones.
// If P is a prefix of Q, `valid(Q)` implies `valid(P)`, so only Q is kept.
// Requirements on incomparable paths are both kept.
static void addValidity(Validity &validity, TimingPath requirement) {
  for (size_t i = 0; i < validity.size();) {
    if (getTimingSuffix(requirement, validity[i]))
      return;
    if (getTimingSuffix(validity[i], requirement)) {
      validity.erase(validity.begin() + i);
      continue;
    }
    ++i;
  }
  validity.push_back(std::move(requirement));
}

// Find a common endpoint T such that every match timing M has a suffix S with:
//   M + S == T
// Each match value and its validity requirements are delayed by S before the
// final property is built.
static FailureOr<TimingPath> findCommonTiming(ArrayRef<SequenceMatch> matches) {
  if (matches.empty())
    return TimingPath{};
  for (auto &candidate : matches) {
    if (llvm::all_of(matches, [&](const SequenceMatch &match) {
          return getTimingSuffix(match.timing, candidate.timing).has_value();
        }))
      return candidate.timing;
  }
  return failure();
}

// Materialize the validity guard used to suppress assertions until all
// registers introduced for explicit-clock sampling contain real history.
static Value materializeValidity(PastValues &pastValues, Value &validStart,
                                 const Validity &validity, OpBuilder &builder,
                                 Operation *contextOp) {
  if (validity.empty())
    return hw::ConstantOp::create(builder, contextOp->getLoc(),
                                  builder.getI1Type(), 1);
  if (!validStart)
    validStart = hw::ConstantOp::create(builder, contextOp->getLoc(),
                                        builder.getI1Type(), 1);

  SmallVector<Value> values;
  for (auto &requirement : validity)
    values.push_back(
        createPast(pastValues, validStart, requirement, builder, contextOp));
  return comb::AndOp::create(builder, contextOp->getLoc(), values,
                             /*twoState=*/false);
}

// Lower an LTL sequence to one or more possible matches. Currently this only
// handles a single explicitly clocked atom; the match is visible immediately
// after sampling the atom.
static FailureOr<LoweredSequence>
lowerSequence( // NOLINT(misc-no-recursion): Walks an acyclic LTL expression.
    SampledAtoms &sampledAtoms, PastValues &pastValues, Value sequence,
    OpBuilder &builder) {
  if (auto atom = sequence.getDefiningOp<ltl::ClockedAtomOp>()) {
    if (atom.getEdge() == ltl::ClockEdge::Both)
      return failure();
    auto [it, inserted] = sampledAtoms.try_emplace(sequence);
    if (inserted) {
      it->second.value = createRegister(atom.getInput(), atom.getClock(),
                                        atom.getEdge(), builder, atom);
      it->second.validity = {{{atom.getClock(), atom.getEdge(), 1}}};
    }
    return LoweredSequence{
        SequenceMatch{{}, it->second.value, it->second.validity}};
  }

  return failure();
}

static FailureOr<LoweredProperty>
lowerSequenceAsProperty(SampledAtoms &sampledAtoms, PastValues &pastValues,
                        Value sequence, OpBuilder &builder,
                        Operation *contextOp) {
  auto lowered = lowerSequence(sampledAtoms, pastValues, sequence, builder);
  if (failed(lowered))
    return failure();

  auto timing = findCommonTiming(*lowered);
  if (failed(timing))
    return failure();

  SmallVector<Value> matches;
  Validity validity;
  for (const auto &match : *lowered) {
    auto suffix = getTimingSuffix(match.timing, *timing);
    assert(suffix && "common timing is compatible with every match");

    matches.push_back(
        createPast(pastValues, match.value, *suffix, builder, contextOp));

    Validity alignedValidity = match.validity;
    if (failed(appendTiming(alignedValidity, *suffix)))
      return failure();
    for (auto requirement : alignedValidity)
      addValidity(validity, std::move(requirement));
  }
  if (!timing->empty()) {
    TimingPath timingValid = *timing;
    auto cycles =
        llvm::checkedAddUnsigned<uint64_t>(timingValid.front().cycles, 1);
    if (!cycles)
      return failure();
    timingValid.front().cycles = *cycles;
    addValidity(validity, std::move(timingValid));
  }
  return LoweredProperty{*timing,
                         comb::OrOp::create(builder, contextOp->getLoc(),
                                            matches,
                                            /*twoState=*/false),
                         std::move(validity)};
}

static bool isSupportedTemporalLTLValue(Value value) {
  if (!value)
    return false;
  auto atom = value.getDefiningOp<ltl::ClockedAtomOp>();
  return atom && atom.getEdge() != ltl::ClockEdge::Both;
}

static Value getAssertLikeProperty(Operation *op) {
  if (auto assertOp = dyn_cast<verif::AssertOp>(op))
    return assertOp.getProperty();
  if (auto assumeOp = dyn_cast<verif::AssumeOp>(op))
    return assumeOp.getProperty();
  return {};
}

// Erase the LTL expression tree after replacing the assert-like property. This
// pass keeps LTL legal and does not run DCE, so the now-dead temporal ops
// would otherwise remain in the module.
// NOLINTNEXTLINE(misc-no-recursion): Walks an acyclic LTL expression.
static void eraseDeadLTLTree(Value value) {
  auto *op = value.getDefiningOp();
  if (!op || !op->use_empty() ||
      op->getDialect()->getNamespace() !=
          ltl::LTLDialect::getDialectNamespace())
    return;

  SmallVector<Value> operands(op->getOperands());
  op->erase();
  for (auto operand : operands)
    eraseDeadLTLTree(operand);
}

static void lowerTemporalLTLToCore(hw::HWModuleOp module) {
  SmallVector<Operation *> assertLikes;
  module->walk([&](Operation *op) {
    if (isa<verif::AssertOp, verif::AssumeOp>(op))
      assertLikes.push_back(op);
  });

  for (auto *op : assertLikes) {
    auto property = getAssertLikeProperty(op);
    if (!isSupportedTemporalLTLValue(property))
      continue;

    SampledAtoms sampledAtoms;
    PastValues pastValues;
    Value validStart;
    OpBuilder builder(op);
    auto lowered = lowerSequenceAsProperty(sampledAtoms, pastValues, property,
                                           builder, op);
    if (failed(lowered))
      continue;
    auto valid = materializeValidity(pastValues, validStart, lowered->validity,
                                     builder, op);
    auto guardedProperty = comb::OrOp::create(
        builder, op->getLoc(),
        comb::createOrFoldNot(builder, op->getLoc(), valid), lowered->value,
        /*twoState=*/false);
    op->setOperand(0, guardedProperty);
    eraseDeadLTLTree(property);
  }
}

} // namespace

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
    if (!isa<verif::AssertOp, verif::ClockedAssertOp>(op))
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
