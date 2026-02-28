//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SOP (Sum-of-Products) balancing for delay optimization
// based on "Delay Optimization Using SOP Balancing" by Mishchenko et al.
// (ICCAD 2011).
//
// NOTE: Currently supports AIG but should be extended to other logic forms
// (MIG, comb or/and) in the future.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/TruthTable.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "synth-sop-balancing"

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SOPBALANCING
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt::synth;

//===----------------------------------------------------------------------===//
// SOP Cache
//===----------------------------------------------------------------------===//

namespace {

/// Expected maximum number of inputs for ISOP extraction, used as a hint for
/// SmallVector capacity to avoid reallocations in common cases.
constexpr unsigned expectedISOPInputs = 8;

/// Cache for SOP extraction results, keyed by truth table.
class SOPCache {
public:
  const SOPForm &getOrCompute(const APInt &truthTable, unsigned numVars) {
    auto it = cache.find(truthTable);
    if (it != cache.end())
      return it->second;
    return cache
        .try_emplace(truthTable, circt::extractISOP(truthTable, numVars))
        .first->second;
  }

private:
  DenseMap<APInt, SOPForm> cache;
};

//===----------------------------------------------------------------------===//
// Tree Building Helpers
//===----------------------------------------------------------------------===//

/// Simulate balanced tree and return output arrival time.
static DelayType simulateBalancedTree(ArrayRef<DelayType> arrivalTimes) {
  if (arrivalTimes.empty())
    return 0;
  return buildBalancedTreeWithArrivalTimes<DelayType>(
      arrivalTimes, [](auto a, auto b) { return std::max(a, b) + 1; });
}

/// Build balanced AND tree.
ValueWithArrivalTime
buildBalancedAndTree(OpBuilder &builder, Location loc,
                     SmallVectorImpl<ValueWithArrivalTime> &nodes) {
  assert(!nodes.empty());

  if (nodes.size() == 1)
    return nodes[0];

  size_t num = nodes.size();
  auto result = buildBalancedTreeWithArrivalTimes<ValueWithArrivalTime>(
      nodes, [&](const auto &n1, const auto &n2) {
        Value v = aig::AndInverterOp::create(builder, loc, n1.getValue(),
                                             n2.getValue(), n1.isInverted(),
                                             n2.isInverted());
        return ValueWithArrivalTime(
            v, std::max(n1.getArrivalTime(), n2.getArrivalTime()) + 1, false,
            num++);
      });
  return result;
}

/// Build balanced SOP structure.
Value buildBalancedSOP(OpBuilder &builder, Location loc, const SOPForm &sop,
                       ArrayRef<Value> inputs,
                       ArrayRef<DelayType> inputArrivalTimes) {
  SmallVector<ValueWithArrivalTime, expectedISOPInputs> productTerms, literals;

  size_t num = 0;
  for (const auto &cube : sop.cubes) {
    for (unsigned i = 0; i < sop.numVars; ++i) {
      if (cube.hasLiteral(i))
        literals.push_back(ValueWithArrivalTime(
            inputs[i], inputArrivalTimes[i], cube.isLiteralInverted(i), num++));
    }

    if (literals.empty())
      continue;

    // Get product term, and flip the inversion to construct OR afterwards.
    productTerms.push_back(
        buildBalancedAndTree(builder, loc, literals).flipInversion());

    literals.clear();
  }

  assert(!productTerms.empty() && "No product terms");

  auto andOfInverted =
      buildBalancedAndTree(builder, loc, productTerms).flipInversion();
  // Let's invert the output.
  if (andOfInverted.isInverted())
    return aig::AndInverterOp::create(builder, loc, andOfInverted.getValue(),
                                      true);
  return andOfInverted.getValue();
}

/// Compute SOP delays for cost estimation.
void computeSOPDelays(const SOPForm &sop, ArrayRef<DelayType> inputArrivalTimes,
                      SmallVectorImpl<DelayType> &delays) {
  SmallVector<DelayType, expectedISOPInputs> productArrivalTimes, literalTimes;
  for (const auto &cube : sop.cubes) {
    for (unsigned i = 0; i < sop.numVars; ++i)
      // No need to consider inverted literals separately for delay.
      if (cube.hasLiteral(i))
        literalTimes.push_back(inputArrivalTimes[i]);
    if (!literalTimes.empty()) {
      productArrivalTimes.push_back(simulateBalancedTree(literalTimes));
      literalTimes.clear();
    }
  }

  DelayType outputTime = simulateBalancedTree(productArrivalTimes);

  delays.resize(sop.numVars, 0);
  // Compute the delay contribution of each input to the output for cost
  // estimation. The CutRewriter framework requires per-input delays, even
  // though this is somewhat artificial for SOP balancing. This may be
  // improved in future framework improvements.
  //
  // First, determine which variables are actually used in the SOP by
  // collecting a bitmask from all cubes.
  uint64_t mask = 0;
  for (auto &cube : sop.cubes)
    mask |= cube.mask;

  // Compute delay for each used input variable.
  for (unsigned i = 0; i < sop.numVars; ++i)
    if (mask & (1u << i))
      delays[i] = outputTime - inputArrivalTimes[i];
}

//===----------------------------------------------------------------------===//
// SOP Balancing Pattern
//===----------------------------------------------------------------------===//

/// Pattern that performs SOP balancing on cuts.
struct SOPBalancingPattern : public CutRewritePattern {
  SOPBalancingPattern(MLIRContext *context) : CutRewritePattern(context) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    if (cut.isTrivialCut() || cut.getOutputSize() != 1)
      return std::nullopt;

    const auto &tt = cut.getTruthTable();
    const SOPForm &sop = sopCache.getOrCompute(tt.table, tt.numInputs);
    if (sop.cubes.empty())
      return std::nullopt;

    SmallVector<DelayType, expectedISOPInputs> arrivalTimes;
    if (failed(cut.getInputArrivalTimes(enumerator, arrivalTimes)))
      return std::nullopt;

    // Compute area estimate
    unsigned totalGates = 0;
    for (const auto &cube : sop.cubes)
      if (cube.size() > 1)
        totalGates += cube.size() - 1;
    if (sop.cubes.size() > 1)
      totalGates += sop.cubes.size() - 1;

    SmallVector<DelayType, expectedISOPInputs> delays;
    computeSOPDelays(sop, arrivalTimes, delays);

    MatchResult result;
    result.area = static_cast<double>(totalGates);
    result.setOwnedDelays(std::move(delays));
    return result;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &tt = cut.getTruthTable();
    const SOPForm &sop = sopCache.getOrCompute(tt.table, tt.numInputs);
    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting SOP:\n";
      sop.dump(llvm::dbgs());
    });

    SmallVector<DelayType, expectedISOPInputs> arrivalTimes;
    if (failed(cut.getInputArrivalTimes(enumerator, arrivalTimes)))
      return failure();
    // Construct the fused location.
    SetVector<Location> inputLocs;
    inputLocs.insert(cut.getRoot()->getLoc());
    for (auto input : cut.inputs)
      inputLocs.insert(input.getLoc());

    auto loc = builder.getFusedLoc(inputLocs.getArrayRef());

    Value result = buildBalancedSOP(builder, loc, sop, cut.inputs.getArrayRef(),
                                    arrivalTimes);

    auto *op = result.getDefiningOp();
    if (!op)
      op = aig::AndInverterOp::create(builder, loc, result, false);
    return op;
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override { return "sop-balancing"; }

private:
  // Cache for SOP extraction results. Hence the pattern is stateful and must
  // not be used in parallelly.
  mutable SOPCache sopCache;
};

} // namespace

//===----------------------------------------------------------------------===//
// SOP Balancing Pass
//===----------------------------------------------------------------------===//

struct SOPBalancingPass
    : public circt::synth::impl::SOPBalancingBase<SOPBalancingPass> {
  using SOPBalancingBase::SOPBalancingBase;

  void runOnOperation() override {
    auto module = getOperation();

    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = maxCutInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.allowNoMatch = true;

    SmallVector<std::unique_ptr<CutRewritePattern>, 1> patterns;
    patterns.push_back(
        std::make_unique<SOPBalancingPattern>(module->getContext()));

    CutRewritePatternSet patternSet(std::move(patterns));
    CutRewriter rewriter(options, patternSet);
    if (failed(rewriter.run(module)))
      return signalPassFailure();
  }
};
