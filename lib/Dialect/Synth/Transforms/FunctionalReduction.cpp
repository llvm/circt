//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements FunctionalReduction (Functionally Reduced And-Inverter
// Graph) optimization using a built-in minimal CDCL SAT solver. It identifies
// and merges functionally equivalent nodes through simulation-based candidate
// detection followed by SAT-based verification.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <random>

#define DEBUG_TYPE "synth-functional-reduction"

static constexpr llvm::StringLiteral kTestClassAttrName =
    "synth.test.fc_equiv_class";

namespace circt {
namespace synth {
#define GEN_PASS_DEF_FUNCTIONALREDUCTION
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

namespace {
enum class EquivResult { Proved, Disproved, Unknown };

//===----------------------------------------------------------------------===//
// Core Functional Reduction Implementation
//===----------------------------------------------------------------------===//

class FunctionalReductionSolver {
public:
  FunctionalReductionSolver(hw::HWModuleOp module, unsigned numPatterns,
                            unsigned seed, bool testTransformation)
      : module(module), numPatterns(numPatterns), seed(seed),
        testTransformation(testTransformation) {}

  ~FunctionalReductionSolver() = default;

  /// Run the Functional Reduction algorithm and return statistics.
  struct Stats {
    unsigned numEquivClasses = 0;
    unsigned numProvedEquiv = 0;
    unsigned numDisprovedEquiv = 0;
    unsigned numUnknown = 0;
    unsigned numMergedNodes = 0;
  };
  mlir::FailureOr<Stats> run();

private:
  // Phase 1: Collect i1 values and run simulation
  void collectValues();
  void runSimulation();
  llvm::APInt simulateValue(Value v);

  // Phase 2: Build equivalence classes from simulation
  void buildEquivalenceClasses();

  // Phase 3: SAT-based verification with per-class solver
  void verifyCandidates();

  // Phase 4: Merge equivalent nodes
  void mergeEquivalentNodes();

  // Test transformation helpers.
  static Attribute getTestEquivClass(Value value);
  static bool matchesTestEquivClass(Value lhs, Value rhs);
  EquivResult verifyEquivalence(Value lhs, Value rhs);

  // Module being processed
  hw::HWModuleOp module;

  // Configuration
  unsigned numPatterns;
  unsigned seed;
  bool testTransformation;

  // Primary inputs (block arguments or results of unknown operations treated as
  // inputs)
  SmallVector<Value> primaryInputs;

  // All i1 values in topological order
  SmallVector<Value> allValues;

  // Simulation signatures: value -> APInt simulation result
  llvm::DenseMap<Value, llvm::APInt> simSignatures;

  // Equivalence candidates: groups of values with identical simulation
  // signatures
  SmallVector<SmallVector<Value>> equivCandidates;

  // Proven equivalences: representative -> proven equivalent members.
  llvm::MapVector<Value, SmallVector<Value>> provenEquivalences;

  Stats stats;
};

Attribute FunctionalReductionSolver::getTestEquivClass(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return {};
  return op->getAttr(kTestClassAttrName);
}

bool FunctionalReductionSolver::matchesTestEquivClass(Value lhs, Value rhs) {
  Attribute lhsClass = getTestEquivClass(lhs);
  Attribute rhsClass = getTestEquivClass(rhs);
  return lhsClass && rhsClass && lhsClass == rhsClass;
}

EquivResult FunctionalReductionSolver::verifyEquivalence(Value lhs, Value rhs) {
  if (testTransformation) {
    if (matchesTestEquivClass(lhs, rhs))
      return EquivResult::Proved;
    return EquivResult::Unknown;
  }

  // TODO: Implement actual SAT-based verification here. For now, we return
  // Unknown.
  return EquivResult::Unknown;
}

//===----------------------------------------------------------------------===//
// Phase 1: Collect values and run simulation
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::collectValues() {
  // Collect block arguments (primary inputs) that are i1
  for (auto arg : module.getBodyBlock()->getArguments()) {
    if (arg.getType().isInteger(1)) {
      primaryInputs.push_back(arg);
      allValues.push_back(arg);
    }
  }

  // Walk operations and collect i1 results
  // - AIG/MIG operations: add to allValues for simulation
  // - Unknown operations: treat as inputs (assign random patterns)
  module.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!result.getType().isInteger(1))
        continue;

      allValues.push_back(result);
      if (!isa<aig::AndInverterOp>(op)) {
        // Unknown operations - treat as primary inputs
        primaryInputs.push_back(result);
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Collected "
                          << primaryInputs.size()
                          << " primary inputs (including unknown ops) and "
                          << allValues.size() << " total i1 values\n");
}

void FunctionalReductionSolver::runSimulation() {
  // Calculate number of 64-bit words needed for numPatterns bits
  unsigned numWords = numPatterns / 64;

  // Create seeded random number generator for deterministic patterns
  std::mt19937_64 rng(seed);

  for (auto input : primaryInputs) {
    // Generate random words using seeded RNG
    SmallVector<uint64_t> words(numWords);
    for (auto &word : words)
      word = rng();

    // Construct APInt directly from words
    llvm::APInt pattern(numPatterns, words);
    simSignatures[input] = pattern;
  }

  // Propagate simulation through the circuit in topological order
  for (auto value : allValues) {
    if (simSignatures.count(value))
      continue; // Already computed (primary input)

    simSignatures[value] = simulateValue(value);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "FunctionalReduction: Simulation complete with "
                 << numPatterns << " patterns\n";
  });
}

llvm::APInt FunctionalReductionSolver::simulateValue(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return simSignatures.at(v);
  return llvm::TypeSwitch<Operation *, llvm::APInt>(op)
      .Case<aig::AndInverterOp>([&](auto op) {
        SmallVector<llvm::APInt> inputSigs;
        for (auto input : op.getInputs())
          inputSigs.push_back(simSignatures.at(input));
        return op.evaluate(inputSigs);
      })
      .Default([&](Operation *) {
        // Unknown operation - treat as input (already assigned a random
        // pattern)
        return simSignatures.at(v);
      });
}

//===----------------------------------------------------------------------===//
// Phase 2: Build equivalence classes from simulation
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::buildEquivalenceClasses() {
  // Map from signature to list of values
  llvm::MapVector<llvm::APInt, SmallVector<Value>> sigGroups;

  for (auto value : allValues)
    sigGroups[simSignatures.at(value)].push_back(value);

  // Build equivalence candidates for groups with >1 member.
  for (auto &[hash, members] : sigGroups) {
    if (members.size() <= 1)
      continue;
    equivCandidates.push_back(std::move(members));
  }
  stats.numEquivClasses = equivCandidates.size();

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Built "
                          << equivCandidates.size()
                          << " equivalence candidates\n");
}

//===----------------------------------------------------------------------===//
// Phase 3: SAT-based verification with per-class solvers
//
// For each equivalence class candidates, verify each member against the
// representative using a SAT solver.
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::verifyCandidates() {
  LLVM_DEBUG(
      llvm::dbgs() << "FunctionalReduction: Starting SAT verification with "
                   << equivCandidates.size() << " equivalence classes\n");

  for (auto &members : equivCandidates) {
    if (members.empty())
      continue;
    auto representative = members.front();
    auto &provenMembers = provenEquivalences[representative];
    // Representative is the canonical node for this class.
    for (auto member : llvm::ArrayRef<Value>(members).drop_front()) {
      EquivResult result = verifyEquivalence(representative, member);
      if (result == EquivResult::Proved) {
        stats.numProvedEquiv++;
        provenMembers.push_back(member);
      } else if (result == EquivResult::Disproved) {
        stats.numDisprovedEquiv++;
        // TODO: Refine equivalence classes based on counterexamples from SAT
        // solver
      } else {
        stats.numUnknown++;
      }
    }
  }

  LLVM_DEBUG(
      llvm::dbgs() << "FunctionalReduction: SAT verification complete. Proved "
                   << stats.numProvedEquiv << " equivalences\n");
}

//===----------------------------------------------------------------------===//
// Phase 4: Merge equivalent nodes
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::mergeEquivalentNodes() {
  if (provenEquivalences.empty())
    return;

  mlir::OpBuilder builder(module.getContext());
  for (auto &provenEquivSet : provenEquivalences) {
    auto &[representative, members] = provenEquivSet;
    if (members.empty())
      continue;
    SmallVector<Value> operands;
    operands.reserve(members.size() + 1);
    operands.push_back(representative);
    operands.append(members);
    builder.setInsertionPointAfterValue(members.back());
    auto choice = synth::ChoiceOp::create(builder, representative.getLoc(),
                                          representative.getType(), operands);
    stats.numMergedNodes += members.size() + 1;
    representative.replaceAllUsesExcept(choice, choice);
    for (auto value : provenEquivSet.second)
      value.replaceAllUsesExcept(choice, choice);
  }

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Merged "
                          << stats.numMergedNodes << " nodes\n");
}

//===----------------------------------------------------------------------===//
// Main Functional Reduction algorithm
//===----------------------------------------------------------------------===//

mlir::FailureOr<FunctionalReductionSolver::Stats>
FunctionalReductionSolver::run() {
  LLVM_DEBUG(
      llvm::dbgs() << "FunctionalReduction: Starting functional reduction with "
                   << numPatterns << " simulation patterns\n");
  // Topologically sort the values

  if (failed(circt::synth::topologicallySortLogicNetwork(module))) {
    module->emitError()
        << "FunctionalReduction: Failed to topologically sort logic network";
    return failure();
  }

  // Phase 1: Collect values and run simulation
  collectValues();
  if (allValues.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "FunctionalReduction: No i1 values to process\n");
    return stats;
  }

  runSimulation();

  // Phase 2: Build equivalence classes
  buildEquivalenceClasses();
  if (equivCandidates.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "FunctionalReduction: No equivalence candidates found\n");
    return stats;
  }

  // Phase 3: SAT-based verification
  verifyCandidates();

  // Phase 4: Merge equivalent nodes
  mergeEquivalentNodes();

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Complete. Stats:\n"
                          << "  Equivalence classes: " << stats.numEquivClasses
                          << "\n"
                          << "  Proved: " << stats.numProvedEquiv << "\n"
                          << "  Disproved: " << stats.numDisprovedEquiv << "\n"
                          << "  Unknown (limit): " << stats.numUnknown << "\n"
                          << "  Merged: " << stats.numMergedNodes << "\n");

  return stats;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct FunctionalReductionPass
    : public circt::synth::impl::FunctionalReductionBase<
          FunctionalReductionPass> {
  using FunctionalReductionBase::FunctionalReductionBase;
  void updateStats(const FunctionalReductionSolver::Stats &stats) {
    numEquivClasses += stats.numEquivClasses;
    numProvedEquiv += stats.numProvedEquiv;
    numDisprovedEquiv += stats.numDisprovedEquiv;
    numUnknown += stats.numUnknown;
    numMergedNodes += stats.numMergedNodes;
  }

  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Running FunctionalReduction pass on "
                            << module.getName() << "\n");

    if (numRandomPatterns == 0 || (numRandomPatterns & 63U) != 0) {
      module.emitError()
          << "'num-random-patterns' must be a positive multiple of 64";
      return signalPassFailure();
    }

    FunctionalReductionSolver fcSolver(module, numRandomPatterns, seed,
                                      testTransformation);
    auto stats = fcSolver.run();
    if (failed(stats))
      return signalPassFailure();
    updateStats(*stats);
    if (stats->numMergedNodes == 0)
      markAllAnalysesPreserved();
  }
};

} // namespace
