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
#include <array>

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

struct SOPSignal {
  unsigned index = 0;
  bool isInput = true;
  bool inverted = false;
};

struct SOPAndNode {
  SOPSignal lhs;
  SOPSignal rhs;
};

struct SOPImplementation : PatternImplementation {
  SmallVector<DelayType, expectedISOPInputs> delays;
  SmallVector<DelayType, expectedISOPInputs> inputArrivalTimes;
};

struct SOPRewritePlan {
  SmallVector<SOPAndNode, 8> nodes;
  SOPSignal output;
};

struct SOPPlanNode {
  DelayType arrivalTime = 0;
  size_t valueNumbering = 0;
  uint64_t usedMask = 0;
  std::array<DelayType, maxTruthTableInputs> inputDelays = {};
  SOPSignal signal;

  SOPPlanNode flipInversion() const {
    SOPPlanNode result = *this;
    result.signal.inverted = !result.signal.inverted;
    return result;
  }

  bool operator>(const SOPPlanNode &other) const {
    return std::tie(arrivalTime, valueNumbering) >
           std::tie(other.arrivalTime, other.valueNumbering);
  }
};

static SOPPlanNode
combineSOPPlanNodes(const SOPPlanNode &lhs, const SOPPlanNode &rhs,
                    unsigned numVars, size_t &valueNumbering,
                    SmallVectorImpl<SOPAndNode> *implementationNodes) {
  SOPPlanNode result;
  result.arrivalTime = std::max(lhs.arrivalTime, rhs.arrivalTime) + 1;
  result.valueNumbering = valueNumbering++;
  result.usedMask = lhs.usedMask | rhs.usedMask;
  result.inputDelays.fill(0);

  if (implementationNodes) {
    implementationNodes->push_back({lhs.signal, rhs.signal});
    result.signal =
        SOPSignal{static_cast<unsigned>(implementationNodes->size() - 1),
                  /*isInput=*/false, /*inverted=*/false};
  }

  auto accumulateDelays = [&](const SOPPlanNode &source) {
    for (unsigned i = 0; i < numVars; ++i) {
      if (!(source.usedMask & (uint64_t{1} << i)))
        continue;
      result.inputDelays[i] =
          std::max(result.inputDelays[i], source.inputDelays[i] + 1);
    }
  };
  accumulateDelays(lhs);
  accumulateDelays(rhs);
  return result;
}

static unsigned getSOPArea(const SOPForm &sop) {
  unsigned totalGates = 0;
  for (const auto &cube : sop.cubes)
    if (cube.size() > 1)
      totalGates += cube.size() - 1;
  if (sop.cubes.size() > 1)
    totalGates += sop.cubes.size() - 1;
  return totalGates;
}

static SOPPlanNode
buildSOPTree(const SOPForm &sop, ArrayRef<DelayType> inputArrivalTimes,
             bool outputInverted,
             SmallVectorImpl<SOPAndNode> *implementationNodes) {
  SmallVector<SOPPlanNode, expectedISOPInputs> productTerms, literals;
  size_t valueNumbering = 0;

  for (const auto &cube : sop.cubes) {
    for (unsigned i = 0; i < sop.numVars; ++i)
      if (cube.hasLiteral(i)) {
        SOPPlanNode literal;
        literal.arrivalTime = inputArrivalTimes[i];
        literal.valueNumbering = valueNumbering++;
        literal.usedMask = uint64_t{1} << i;
        literal.inputDelays.fill(0);
        literal.signal =
            SOPSignal{i, /*isInput=*/true, cube.isLiteralInverted(i)};
        literals.push_back(std::move(literal));
      }

    if (!literals.empty()) {
      productTerms.push_back(
          buildBalancedTreeWithArrivalTimes<SOPPlanNode, expectedISOPInputs>(
              literals,
              [&](const SOPPlanNode &lhs, const SOPPlanNode &rhs) {
                return combineSOPPlanNodes(lhs, rhs, sop.numVars,
                                           valueNumbering, implementationNodes);
              })
              .flipInversion());
      literals.clear();
    }
  }

  assert(!productTerms.empty() && "No product terms");
  auto output =
      buildBalancedTreeWithArrivalTimes<SOPPlanNode, expectedISOPInputs>(
          productTerms,
          [&](const SOPPlanNode &lhs, const SOPPlanNode &rhs) {
            return combineSOPPlanNodes(lhs, rhs, sop.numVars, valueNumbering,
                                       implementationNodes);
          })
          .flipInversion();
  if (outputInverted)
    output = output.flipInversion();
  return output;
}

static SmallVector<DelayType, expectedISOPInputs>
computeSOPDelays(const SOPForm &sop, ArrayRef<DelayType> inputArrivalTimes,
                 bool outputInverted) {
  auto output = buildSOPTree(sop, inputArrivalTimes, outputInverted, nullptr);
  SmallVector<DelayType, expectedISOPInputs> delays(sop.numVars, 0);
  for (unsigned i = 0; i < sop.numVars; ++i)
    if (output.usedMask & (uint64_t{1} << i))
      delays[i] = output.inputDelays[i];
  return delays;
}

static SOPRewritePlan buildSOPRewritePlan(const SOPForm &sop,
                                          ArrayRef<DelayType> inputArrivalTimes,
                                          bool outputInverted) {
  SOPRewritePlan plan;
  auto output =
      buildSOPTree(sop, inputArrivalTimes, outputInverted, &plan.nodes);
  plan.output = output.signal;
  return plan;
}

//===----------------------------------------------------------------------===//
// SOP Balancing Pattern
//===----------------------------------------------------------------------===//

/// Pattern that performs SOP balancing on cuts.
struct SOPBalancingPattern : public CutRewritePattern {
  SOPBalancingPattern(MLIRContext *context, SOPCache &sopCache,
                      bool outputInverted)
      : CutRewritePattern(context), sopCache(sopCache),
        outputInverted(outputInverted) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator, const Cut &cut,
                                   const MatchBinding &binding) const override {
    const auto &network = enumerator.getLogicNetwork();
    (void)binding;
    if (cut.isTrivialCut() || cut.getOutputSize(network) != 1)
      return std::nullopt;

    const auto &tt = *cut.getTruthTable();
    APInt truthTable = outputInverted ? ~tt.table : tt.table;
    const SOPForm &sop = sopCache.getOrCompute(truthTable, tt.numInputs);
    if (sop.cubes.empty())
      return std::nullopt;

    // Constants should already be handled by cheaper trivial cuts.
    if (llvm::any_of(sop.cubes,
                     [](const Cube &cube) { return cube.size() == 0; }))
      return std::nullopt;

    SmallVector<DelayType, expectedISOPInputs> inputArrivalTimes;
    if (failed(cut.getInputArrivalTimes(enumerator, inputArrivalTimes)))
      return std::nullopt;

    auto implementation = std::make_shared<SOPImplementation>();
    implementation->delays =
        computeSOPDelays(sop, inputArrivalTimes, outputInverted);
    implementation->inputArrivalTimes.assign(inputArrivalTimes);

    MatchResult match;
    match.area = static_cast<double>(getSOPArea(sop));
    match.setDelayRef(implementation->delays);
    match.setImplementation(std::move(implementation));
    return match;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut,
                                 const MatchedPattern &matched) const override {
    const auto &network = enumerator.getLogicNetwork();
    const auto &tt = *cut.getTruthTable();
    APInt truthTable = outputInverted ? ~tt.table : tt.table;
    const SOPForm &sop = sopCache.getOrCompute(truthTable, tt.numInputs);
    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting SOP:\n";
      sop.dump(llvm::dbgs());
    });

    auto *implementation =
        static_cast<const SOPImplementation *>(matched.getImplementation());
    if (!implementation)
      return failure();
    SOPRewritePlan plan = buildSOPRewritePlan(
        sop, implementation->inputArrivalTimes, outputInverted);

    // Construct the fused location.
    SetVector<Location> inputLocs;
    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");
    inputLocs.insert(rootOp->getLoc());

    SmallVector<Value> inputValues;
    network.getValues(cut.inputs, inputValues);
    for (auto input : inputValues)
      inputLocs.insert(input.getLoc());

    auto loc = builder.getFusedLoc(inputLocs.getArrayRef());

    SmallVector<Value> nodeValues;
    nodeValues.reserve(plan.nodes.size());
    auto resolveSignal = [&](const SOPSignal &signal) -> Value {
      return signal.isInput ? inputValues[signal.index]
                            : nodeValues[signal.index];
    };

    for (const auto &node : plan.nodes) {
      Value lhs = resolveSignal(node.lhs);
      Value rhs = resolveSignal(node.rhs);
      nodeValues.push_back(aig::AndInverterOp::create(
          builder, loc, lhs, rhs, node.lhs.inverted, node.rhs.inverted));
    }

    Value result = resolveSignal(plan.output);
    if (plan.output.inverted)
      result = aig::AndInverterOp::create(builder, loc, result, true);

    auto *op = result.getDefiningOp();
    if (!op)
      op = aig::AndInverterOp::create(builder, loc, result, false);
    return op;
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override {
    return outputInverted ? "sop-balancing-inverted-output" : "sop-balancing";
  }

private:
  SOPCache &sopCache;
  bool outputInverted = false;
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

    SOPCache sopCache;
    SmallVector<std::unique_ptr<CutRewritePattern>, 2> patterns;
    patterns.push_back(std::make_unique<SOPBalancingPattern>(
        module->getContext(), sopCache, /*outputInverted=*/false));
    patterns.push_back(std::make_unique<SOPBalancingPattern>(
        module->getContext(), sopCache, /*outputInverted=*/true));

    CutRewritePatternSet patternSet(std::move(patterns));
    CutRewriter rewriter(options, patternSet);
    if (failed(rewriter.run(module)))
      return signalPassFailure();

    const auto &stats = rewriter.getStats();
    numCutsCreated += stats.numCutsCreated;
    numCutSetsCreated += stats.numCutSetsCreated;
    numCutsRewritten += stats.numCutsRewritten;
  }
};
