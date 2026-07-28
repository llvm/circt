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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

struct LTLBooleanConstantConversion
    : public OpConversionPattern<ltl::BooleanConstantOp> {
  using OpConversionPattern<ltl::BooleanConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::BooleanConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, rewriter.getI1Type(),
                                                op.getValue() ? 1 : 0);
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

/// A timing path records the explicit clock events between the start and end of
/// a sequence. Adjacent steps always use different clock events.
struct TimingStep {
  Value clock;
  ltl::ClockEdge edge;
  uint64_t cycles;

  bool operator==(const TimingStep &other) const {
    return clock == other.clock && edge == other.edge && cycles == other.cycles;
  }
};

using TimingPath = SmallVector<TimingStep>;
using Validity = SmallVector<TimingPath>;

/// A lowered sequence is evaluated at its end point. Each match records the
/// clock events from the sequence start to that end point and the i1 signal
/// which indicates a match there.
struct SequenceMatch {
  TimingPath timing;
  Value value;
  Validity validity;
};

struct LoweredSequence {
  SmallVector<SequenceMatch> matches;
};

/// A lowered property is evaluated after its timing path has elapsed.
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
using PastValues = DenseMap<std::tuple<Value, Value, unsigned>, Value>;

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

static Value createPast(PastValues &pastValues, Value input, uint64_t cycles,
                        Value clock, ltl::ClockEdge edge, OpBuilder &builder,
                        Operation *contextOp) {
  Value current = input;
  for (uint64_t i = 0; i < cycles; ++i) {
    auto [it, inserted] = pastValues.try_emplace(
        std::make_tuple(current, clock, static_cast<unsigned>(edge)));
    if (inserted)
      it->second = createRegister(current, clock, edge, builder, contextOp);
    current = it->second;
  }
  return current;
}

static Value createPast(PastValues &pastValues, Value input,
                        ArrayRef<TimingStep> timing, OpBuilder &builder,
                        Operation *contextOp) {
  for (auto step : timing)
    input = createPast(pastValues, input, step.cycles, step.clock, step.edge,
                       builder, contextOp);
  return input;
}

static LogicalResult appendTiming(TimingPath &timing, Value clock,
                                  ltl::ClockEdge edge, uint64_t cycles,
                                  Operation *contextOp) {
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
    return contextOp->emitError("LTL sequence length overflows 64 bits");
  timing.back().cycles = *combined;
  return success();
}

static LogicalResult appendTiming(TimingPath &timing,
                                  ArrayRef<TimingStep> suffix,
                                  Operation *contextOp) {
  for (auto step : suffix)
    if (failed(appendTiming(timing, step.clock, step.edge, step.cycles,
                            contextOp)))
      return failure();
  return success();
}

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

static LogicalResult appendTiming(Validity &validity,
                                  ArrayRef<TimingStep> suffix,
                                  Operation *contextOp) {
  for (auto &timing : validity)
    if (failed(appendTiming(timing, suffix, contextOp)))
      return failure();
  return success();
}

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

static void appendValidity(Validity &validity, const Validity &suffix) {
  for (auto requirement : suffix)
    addValidity(validity, std::move(requirement));
}

static FailureOr<TimingPath>
findCommonTiming(ArrayRef<LoweredProperty> properties) {
  if (properties.empty())
    return TimingPath{};
  for (auto &candidate : properties) {
    if (llvm::all_of(properties, [&](const LoweredProperty &property) {
          return getTimingSuffix(property.timing, candidate.timing).has_value();
        }))
      return candidate.timing;
  }
  return failure();
}

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

static Value alignValue(PastValues &pastValues, Value value,
                        ArrayRef<TimingStep> from, ArrayRef<TimingStep> to,
                        OpBuilder &builder, Operation *contextOp) {
  auto suffix = getTimingSuffix(from, to);
  assert(suffix && "timing compatibility checked before alignment");
  return createPast(pastValues, value, *suffix, builder, contextOp);
}

static FailureOr<Validity> alignValidity(const Validity &validity,
                                         ArrayRef<TimingStep> from,
                                         ArrayRef<TimingStep> to,
                                         Operation *contextOp) {
  auto suffix = getTimingSuffix(from, to);
  assert(suffix && "timing compatibility checked before alignment");
  Validity result = validity;
  if (failed(appendTiming(result, *suffix, contextOp)))
    return failure();
  return result;
}

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

static void addMatch(LoweredSequence &sequence, TimingPath timing, Value value,
                     Validity validity, OpBuilder &builder, Location loc) {
  for (auto &match : sequence.matches) {
    if (match.timing != timing)
      continue;
    match.value = comb::OrOp::create(builder, loc, match.value, value,
                                     /*twoState=*/false);
    appendValidity(match.validity, validity);
    return;
  }
  sequence.matches.push_back({std::move(timing), value, std::move(validity)});
}

static FailureOr<LoweredSequence>
lowerSequence( // NOLINT(misc-no-recursion): Walks an acyclic LTL expression.
    SampledAtoms &sampledAtoms, PastValues &pastValues, Value sequence,
    OpBuilder &builder, Operation *contextOp) {
  auto loc = sequence.getLoc();
  if (auto atom = sequence.getDefiningOp<ltl::ClockedAtomOp>()) {
    if (atom.getEdge() == ltl::ClockEdge::Both) {
      atom.emitError("lower-ltl-to-core does not support both-edge LTL clocks");
      return failure();
    }
    auto [it, inserted] = sampledAtoms.try_emplace(sequence);
    if (inserted) {
      it->second.value = createRegister(atom.getInput(), atom.getClock(),
                                        atom.getEdge(), builder, atom);
      it->second.validity = {{{atom.getClock(), atom.getEdge(), 1}}};
    }
    return LoweredSequence{{{{}, it->second.value, it->second.validity}}};
  }

  if (auto delay = sequence.getDefiningOp<ltl::ClockedDelayOp>()) {
    auto length = delay.getLength();
    if (!length) {
      delay.emitError("lower-ltl-to-core only supports bounded LTL delays");
      return failure();
    }
    if (delay.getEdge() == ltl::ClockEdge::Both) {
      delay.emitError(
          "lower-ltl-to-core does not support both-edge LTL clocks");
      return failure();
    }

    auto lowered = lowerSequence(sampledAtoms, pastValues, delay.getInput(),
                                 builder, delay);
    if (failed(lowered))
      return failure();

    LoweredSequence result;
    for (const auto &match : lowered->matches) {
      for (uint64_t i = 0;; ++i) {
        auto matchLength =
            llvm::checkedAddUnsigned<uint64_t>(delay.getDelay(), i);
        if (!matchLength) {
          delay.emitError("LTL sequence length overflows 64 bits");
          return failure();
        }
        TimingPath timing;
        if (failed(appendTiming(timing, delay.getClock(), delay.getEdge(),
                                *matchLength, delay)) ||
            failed(appendTiming(timing, match.timing, delay)))
          return failure();
        addMatch(result, std::move(timing), match.value, match.validity,
                 builder, loc);
        if (i == *length)
          break;
      }
    }
    return result;
  }

  if (auto concat = sequence.getDefiningOp<ltl::ConcatOp>()) {
    auto current = lowerSequence(sampledAtoms, pastValues,
                                 concat.getInputs().front(), builder, concat);
    if (failed(current))
      return failure();

    for (auto input : concat.getInputs().drop_front()) {
      auto next =
          lowerSequence(sampledAtoms, pastValues, input, builder, concat);
      if (failed(next))
        return failure();

      LoweredSequence combined;
      for (const auto &lhs : current->matches) {
        for (const auto &rhs : next->matches) {
          auto lhsAtEnd =
              createPast(pastValues, lhs.value, rhs.timing, builder, concat);
          auto lhsValidityAtEnd =
              alignValidity(lhs.validity, {}, rhs.timing, concat);
          if (failed(lhsValidityAtEnd))
            return failure();
          appendValidity(*lhsValidityAtEnd, rhs.validity);
          TimingPath timing = lhs.timing;
          if (failed(appendTiming(timing, rhs.timing, concat)))
            return failure();
          addMatch(combined, std::move(timing),
                   comb::AndOp::create(builder, loc, lhsAtEnd, rhs.value,
                                       /*twoState=*/false),
                   std::move(*lhsValidityAtEnd), builder, loc);
        }
      }
      current = combined;
    }
    return *current;
  }

  if (auto *defOp = sequence.getDefiningOp())
    defOp->emitError("lower-ltl-to-core does not support temporal LTL "
                     "operation '")
        << defOp->getName() << "'";
  else
    contextOp->emitError("lower-ltl-to-core does not support block argument "
                         "LTL sequences");
  return failure();
}

static FailureOr<LoweredProperty>
lowerSequenceAsProperty(SampledAtoms &sampledAtoms, PastValues &pastValues,
                        Value sequence, OpBuilder &builder,
                        Operation *contextOp) {
  auto lowered =
      lowerSequence(sampledAtoms, pastValues, sequence, builder, contextOp);
  if (failed(lowered))
    return failure();

  auto timing = findCommonTiming(lowered->matches);
  if (failed(timing)) {
    contextOp->emitError(
        "lower-ltl-to-core cannot align temporal LTL sequence matches");
    return failure();
  }

  SmallVector<Value> matches;
  Validity validity;
  for (const auto &match : lowered->matches) {
    matches.push_back(alignValue(pastValues, match.value, match.timing, *timing,
                                 builder, contextOp));
    auto aligned =
        alignValidity(match.validity, match.timing, *timing, contextOp);
    if (failed(aligned))
      return failure();
    appendValidity(validity, *aligned);
  }
  if (!timing->empty()) {
    TimingPath timingValid = *timing;
    auto cycles =
        llvm::checkedAddUnsigned<uint64_t>(timingValid.front().cycles, 1);
    if (!cycles) {
      contextOp->emitError("LTL sequence length overflows 64 bits");
      return failure();
    }
    timingValid.front().cycles = *cycles;
    addValidity(validity, std::move(timingValid));
  }
  return LoweredProperty{*timing,
                         comb::OrOp::create(builder, contextOp->getLoc(),
                                            matches,
                                            /*twoState=*/false),
                         std::move(validity)};
}

static FailureOr<LoweredProperty>
lowerProperty( // NOLINT(misc-no-recursion): Walks an acyclic LTL expression.
    SampledAtoms &sampledAtoms, PastValues &pastValues, Value property,
    OpBuilder &builder, Operation *contextOp) {
  auto loc = property.getLoc();
  if (auto implication = property.getDefiningOp<ltl::ImplicationOp>()) {
    auto antecedent =
        lowerSequence(sampledAtoms, pastValues, implication.getAntecedent(),
                      builder, implication);
    auto consequent =
        lowerProperty(sampledAtoms, pastValues, implication.getConsequent(),
                      builder, implication);
    if (failed(antecedent) || failed(consequent))
      return failure();

    SmallVector<LoweredProperty> obligations;
    for (const auto &match : antecedent->matches) {
      TimingPath timing = match.timing;
      if (failed(appendTiming(timing, consequent->timing, implication)))
        return failure();
      auto antecedentAtEnd = createPast(
          pastValues, match.value, consequent->timing, builder, implication);
      auto validity =
          alignValidity(match.validity, {}, consequent->timing, implication);
      if (failed(validity))
        return failure();
      appendValidity(*validity, consequent->validity);
      obligations.push_back(
          {std::move(timing),
           comb::OrOp::create(
               builder, loc,
               comb::createOrFoldNot(builder, loc, antecedentAtEnd),
               consequent->value, /*twoState=*/false),
           std::move(*validity)});
    }

    auto timing = findCommonTiming(obligations);
    if (failed(timing)) {
      implication.emitError(
          "lower-ltl-to-core cannot align temporal LTL timing paths");
      return failure();
    }

    SmallVector<Value> values;
    Validity validity;
    for (auto &obligation : obligations) {
      values.push_back(alignValue(pastValues, obligation.value,
                                  obligation.timing, *timing, builder,
                                  implication));
      auto aligned = alignValidity(obligation.validity, obligation.timing,
                                   *timing, implication);
      if (failed(aligned))
        return failure();
      appendValidity(validity, *aligned);
    }
    return LoweredProperty{
        *timing, comb::AndOp::create(builder, loc, values, /*twoState=*/false),
        std::move(validity)};
  }

  if (auto notOp = property.getDefiningOp<ltl::NotOp>()) {
    auto input = lowerProperty(sampledAtoms, pastValues, notOp.getInput(),
                               builder, notOp);
    if (failed(input))
      return failure();
    return LoweredProperty{input->timing,
                           comb::createOrFoldNot(builder, loc, input->value),
                           input->validity};
  }

  if (auto andOp = property.getDefiningOp<ltl::AndOp>()) {
    SmallVector<LoweredProperty> inputs;
    for (auto input : andOp.getInputs()) {
      auto lowered =
          lowerProperty(sampledAtoms, pastValues, input, builder, andOp);
      if (failed(lowered))
        return failure();
      inputs.push_back(*lowered);
    }

    auto timing = findCommonTiming(inputs);
    if (failed(timing)) {
      andOp.emitError(
          "lower-ltl-to-core cannot align temporal LTL timing paths");
      return failure();
    }

    SmallVector<Value> values;
    Validity validity;
    for (auto &input : inputs) {
      values.push_back(alignValue(pastValues, input.value, input.timing,
                                  *timing, builder, andOp));
      auto aligned =
          alignValidity(input.validity, input.timing, *timing, andOp);
      if (failed(aligned))
        return failure();
      appendValidity(validity, *aligned);
    }
    return LoweredProperty{
        *timing, comb::AndOp::create(builder, loc, values, /*twoState=*/false),
        std::move(validity)};
  }

  if (auto orOp = property.getDefiningOp<ltl::OrOp>()) {
    SmallVector<LoweredProperty> inputs;
    for (auto input : orOp.getInputs()) {
      auto lowered =
          lowerProperty(sampledAtoms, pastValues, input, builder, orOp);
      if (failed(lowered))
        return failure();
      inputs.push_back(*lowered);
    }

    auto timing = findCommonTiming(inputs);
    if (failed(timing)) {
      orOp.emitError(
          "lower-ltl-to-core cannot align temporal LTL timing paths");
      return failure();
    }

    SmallVector<Value> values;
    Validity validity;
    for (auto &input : inputs) {
      values.push_back(alignValue(pastValues, input.value, input.timing,
                                  *timing, builder, orOp));
      auto aligned = alignValidity(input.validity, input.timing, *timing, orOp);
      if (failed(aligned))
        return failure();
      appendValidity(validity, *aligned);
    }
    return LoweredProperty{
        *timing, comb::OrOp::create(builder, loc, values, /*twoState=*/false),
        std::move(validity)};
  }

  if (isa<ltl::SequenceType>(property.getType()))
    return lowerSequenceAsProperty(sampledAtoms, pastValues, property, builder,
                                   contextOp);

  if (auto *defOp = property.getDefiningOp())
    defOp->emitError("lower-ltl-to-core does not support temporal LTL "
                     "operation '")
        << defOp->getName() << "'";
  else
    contextOp->emitError("lower-ltl-to-core does not support block argument "
                         "LTL properties");
  return failure();
}

static bool isTemporalLTLValue(Value value, DenseSet<Value> &visited);

static bool isTemporalLTLValue(Value value) {
  DenseSet<Value> visited;
  return isTemporalLTLValue(value, visited);
}

// NOLINTNEXTLINE(misc-no-recursion): Walks an acyclic LTL expression.
static bool isTemporalLTLValue(Value value, DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;

  auto *op = value.getDefiningOp();
  if (!op || op->getDialect()->getNamespace() !=
                 ltl::LTLDialect::getDialectNamespace())
    return false;

  if (isa<ltl::ClockOp, ltl::ClockedAtomOp, ltl::DelayOp, ltl::ClockedDelayOp,
          ltl::ConcatOp>(op))
    return true;

  if (auto implication = dyn_cast<ltl::ImplicationOp>(op))
    return isTemporalLTLValue(implication.getAntecedent(), visited) ||
           isTemporalLTLValue(implication.getConsequent(), visited);

  if (auto notOp = dyn_cast<ltl::NotOp>(op))
    return isTemporalLTLValue(notOp.getInput(), visited);

  if (auto andOp = dyn_cast<ltl::AndOp>(op)) {
    for (auto input : andOp.getInputs())
      if (isTemporalLTLValue(input, visited))
        return true;
    return false;
  }

  if (auto orOp = dyn_cast<ltl::OrOp>(op)) {
    for (auto input : orOp.getInputs())
      if (isTemporalLTLValue(input, visited))
        return true;
    return false;
  }

  for (auto operand : op->getOperands())
    if (isTemporalLTLValue(operand, visited))
      return true;
  return false;
}

static Value getAssertLikeProperty(Operation *op) {
  if (auto assertOp = dyn_cast<verif::AssertOp>(op))
    return assertOp.getProperty();
  if (auto assumeOp = dyn_cast<verif::AssumeOp>(op))
    return assumeOp.getProperty();
  return {};
}

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

static LogicalResult lowerTemporalLTLToCore(hw::HWModuleOp module) {
  SmallVector<Operation *> assertLikes;
  module->walk([&](Operation *op) {
    if (isa<verif::AssertOp, verif::AssumeOp>(op))
      assertLikes.push_back(op);
  });

  for (auto *op : assertLikes) {
    auto property = getAssertLikeProperty(op);
    if (!isTemporalLTLValue(property))
      continue;

    SampledAtoms sampledAtoms;
    PastValues pastValues;
    Value validStart;
    OpBuilder builder(op);
    auto lowered =
        lowerProperty(sampledAtoms, pastValues, property, builder, op);
    if (failed(lowered))
      return failure();
    auto valid = materializeValidity(pastValues, validStart, lowered->validity,
                                     builder, op);
    auto guardedProperty = comb::OrOp::create(
        builder, op->getLoc(),
        comb::createOrFoldNot(builder, op->getLoc(), valid), lowered->value,
        /*twoState=*/false);
    op->setOperand(0, guardedProperty);
    eraseDeadLTLTree(property);
  }

  return success();
}

static bool isSupportedBooleanPropertyUser(Operation *op) {
  return isa<verif::AssertOp, verif::AssumeOp, verif::ClockedAssertOp>(op);
}

static void cleanupBooleanLTLToCore(hw::HWModuleOp module) {
  // Clean up remaining unrealized casts at assertion and assumption users.
  SmallVector<UnrealizedConversionCastOp> deadCasts;
  module.walk([&](Operation *op) {
    if (!isSupportedBooleanPropertyUser(op))
      return;
    Value prop = op->getOperand(0);
    if (auto cast = prop.getDefiningOp<UnrealizedConversionCastOp>()) {
      // Make sure that the cast is from an i1, not something random that was
      // in the input
      if (auto intType = dyn_cast<IntegerType>(cast.getOperandTypes()[0]);
          intType && intType.getWidth() == 1) {
        op->setOperand(0, cast.getInputs()[0]);
        if (cast->use_empty())
          deadCasts.push_back(cast);
      }
    }
  });
  for (auto cast : deadCasts)
    cast.erase();
}

static LogicalResult checkNoLTLTypes(hw::HWModuleOp module) {
  auto containsLTLType = [](Type type) {
    return type
        .walk([](Type nestedType) {
          return isa<ltl::PropertyType, ltl::SequenceType>(nestedType)
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        })
        .wasInterrupted();
  };
  WalkResult result = module->walk([&](Operation *op) {
    for (auto type : op->getResultTypes()) {
      if (!containsLTLType(type))
        continue;
      op->emitError("lower-ltl-to-core cannot eliminate LTL type ") << type;
      return WalkResult::interrupt();
    }
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto argument : block.getArguments()) {
          auto type = argument.getType();
          if (!containsLTLType(type))
            continue;
          op->emitError("lower-ltl-to-core cannot eliminate LTL type ") << type;
          return WalkResult::interrupt();
        }
    for (const auto &namedAttr : op->getAttrs()) {
      Type ltlType;
      namedAttr.getValue().walk([&](Type type) {
        if (isa<ltl::PropertyType, ltl::SequenceType>(type))
          ltlType = type;
      });
      if (!ltlType)
        continue;
      op->emitError("lower-ltl-to-core cannot eliminate LTL type ")
          << ltlType << " in attribute '" << namedAttr.getName().getValue()
          << "'";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static LogicalResult lowerRemainingLTLToCore(hw::HWModuleOp module,
                                             ConversionTarget &target) {
  TypeConverter converter;
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type resultType,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });
  converter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type resultType,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  target.addIllegalOp<verif::HasBeenResetOp>();
  RewritePatternSet patterns(module.getContext());
  patterns.add<LTLBooleanConstantConversion, LTLImplicationConversion,
               LTLNotConversion, LTLAndOpConversion, LTLOrOpConversion,
               LTLIntersectOpConversion>(converter, patterns.getContext());
  patterns.add<HasBeenResetOpConversion, LTLPastOpConversion>(
      patterns.getContext());

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return failure();

  cleanupBooleanLTLToCore(module);
  return checkNoLTLTypes(module);
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

void LowerLTLToCorePass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addIllegalDialect<ltl::LTLDialect>();

  if (failed(lowerTemporalLTLToCore(module)) ||
      failed(lowerRemainingLTLToCore(module, target))) {
    signalPassFailure();
  }
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
