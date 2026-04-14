//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements FunctionalReduction (Functionally Reduced And-Inverter
// Graph) optimization. It identifies and merges functionally equivalent nodes
// through simulation-based candidate detection followed by SAT-based
// verification.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOpInterfaces.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SATSolver.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
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

std::unique_ptr<IncrementalSATSolver>
createFunctionalReductionSATSolver(llvm::StringRef backend) {
  if (backend == "auto") {
    if (auto solver = createCadicalSATSolver())
      return solver;
    return createZ3SATSolver();
  }
  if (backend == "cadical")
    return createCadicalSATSolver();
  if (backend == "z3")
    return createZ3SATSolver();
  return {};
}

class FunctionalReductionSATBuilder {
public:
  FunctionalReductionSATBuilder(IncrementalSATSolver &solver,
                                llvm::DenseMap<Value, int> &satVars,
                                llvm::DenseSet<Value> &encodedValues,
                                int &nextFreshVar);

  // If inverted, negates rhs in the SAT encoding to check lhs == NOT(rhs).
  EquivResult verify(Value lhs, Value rhs, bool inverted);

private:
  int getOrCreateVar(Value value);
  // Create a fresh SAT variable for an intermediate Boolean subexpression that
  // does not correspond to an MLIR value.
  int createAuxVar();
  SmallVector<int> getOperandVars(ValueRange operands);
  void encodeValue(Value value);

  IncrementalSATSolver &solver;
  llvm::DenseMap<Value, int> &satVars;
  llvm::DenseSet<Value> &encodedValues;
  int &nextFreshVar;
};

static bool isFunctionalReductionSimulatableOp(Operation *op) {
  return isa<BooleanLogicOpInterface, comb::AndOp, comb::OrOp, comb::XorOp>(op);
}

EquivResult FunctionalReductionSATBuilder::verify(Value lhs, Value rhs,
                                                  bool inverted) {
  encodeValue(lhs);
  encodeValue(rhs);

  int lhsVar = getOrCreateVar(lhs);
  int rhsVar = getOrCreateVar(rhs);

  if (inverted)
    rhsVar = -rhsVar;
  // Check the two halves of the XOR miter separately. If either assignment is
  // satisfiable, the solver found a distinguishing input pattern.
  solver.assume(lhsVar);
  solver.assume(-rhsVar);
  auto result = solver.solve();
  if (result == IncrementalSATSolver::kSAT)
    return EquivResult::Disproved;
  if (result != IncrementalSATSolver::kUNSAT)
    return EquivResult::Unknown;

  solver.assume(-lhsVar);
  solver.assume(rhsVar);
  result = solver.solve();
  if (result == IncrementalSATSolver::kSAT)
    return EquivResult::Disproved;
  if (result != IncrementalSATSolver::kUNSAT)
    return EquivResult::Unknown;

  return EquivResult::Proved;
}

int FunctionalReductionSATBuilder::getOrCreateVar(Value value) {
  auto it = satVars.find(value);
  assert(it != satVars.end() && "SAT variable must be preallocated");
  return it->second;
}

int FunctionalReductionSATBuilder::createAuxVar() {
  int freshVar = ++nextFreshVar;
  solver.reserveVars(freshVar);
  return freshVar;
}

SmallVector<int>
FunctionalReductionSATBuilder::getOperandVars(ValueRange operands) {
  SmallVector<int> vars;
  vars.reserve(operands.size());
  for (auto operand : operands)
    vars.push_back(getOrCreateVar(operand));
  return vars;
}

void FunctionalReductionSATBuilder::encodeValue(Value value) {
  SmallVector<std::pair<Value, bool>> worklist;
  worklist.push_back({value, false});

  while (!worklist.empty()) {
    auto [current, readyToEncode] = worklist.pop_back_val();
    if (encodedValues.contains(current))
      continue;

    Operation *op = current.getDefiningOp();
    if (!op) {
      encodedValues.insert(current);
      continue;
    }

    APInt constantValue;
    if (matchPattern(current, mlir::m_ConstantInt(&constantValue))) {
      encodedValues.insert(current);
      solver.addClause({constantValue.isZero() ? -getOrCreateVar(current)
                                               : getOrCreateVar(current)});
      continue;
    }

    if (!isFunctionalReductionSimulatableOp(op)) {
      // Unsupported operations remain unconstrained, just like block
      // arguments. Since we only prove equivalence from UNSAT, omitting these
      // clauses may miss a proof but cannot create a false proof.
      encodedValues.insert(current);
      continue;
    }

    if (!readyToEncode) {
      worklist.push_back({current, true});
      for (auto input : op->getOperands()) {
        assert(input.getType().isInteger(1) &&
               "only i1 inputs should be simulated or encoded");
        if (!encodedValues.contains(input))
          worklist.push_back({input, false});
      }
      continue;
    }

    encodedValues.insert(current);
    int outVar = getOrCreateVar(current);
    auto addClause = [&](llvm::ArrayRef<int> clause) {
      solver.addClause(clause);
    };

    TypeSwitch<Operation *>(op)
        .Case<BooleanLogicOpInterface>([&](auto logicOp) {
          auto inputVars = getOperandVars(logicOp.getInputs());
          logicOp.emitCNF(outVar, inputVars, addClause,
                          [&]() { return createAuxVar(); });
        })
        .Case<comb::AndOp>([&](auto andOp) {
          auto inputLits = getOperandVars(andOp.getInputs());
          circt::addAndClauses(outVar, inputLits, addClause);
        })
        .Case<comb::OrOp>([&](auto orOp) {
          auto inputLits = getOperandVars(orOp.getInputs());
          circt::addOrClauses(outVar, inputLits, addClause);
        })
        .Case<comb::XorOp>([&](auto xorOp) {
          auto inputLits = getOperandVars(xorOp.getInputs());
          circt::addParityClauses(outVar, inputLits, addClause,
                                  [&]() { return createAuxVar(); });
        })
        .Default(
            [](Operation *) { llvm_unreachable("unexpected supported op"); });
  }
}

//===----------------------------------------------------------------------===//
// Core Functional Reduction Implementation
//===----------------------------------------------------------------------===//

class FunctionalReductionSolver {
public:
  FunctionalReductionSolver(hw::HWModuleOp module, unsigned numPatterns,
                            unsigned seed, bool testTransformation,
                            std::unique_ptr<IncrementalSATSolver> satSolver)
      : module(module), numPatterns(numPatterns), seed(seed),
        testTransformation(testTransformation),
        satSolver(std::move(satSolver)) {}

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
  void initializeSATState();

  // Phase 4: Merge equivalent nodes
  void mergeEquivalentNodes();

  // Test transformation helpers.
  static Attribute getTestEquivClass(Value value);
  static bool matchesTestEquivClass(Value lhs, Value rhs);
  EquivResult verifyEquivalence(Value lhs, Value rhs, bool inverted);

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

  // Equivalence candidates: groups of values with identical or inverted
  // simulation signatures, tracked with an inversion flag
  SmallVector<SmallVector<std::pair<Value, bool>>> equivCandidates;

  // Proven equivalences: representative -> proven equivalent members with
  // inversion flag indicating whether the member is inverted relative to
  // representative
  llvm::MapVector<Value, SmallVector<std::pair<Value, bool>>>
      provenEquivalences;

  std::unique_ptr<IncrementalSATSolver> satSolver;
  std::unique_ptr<FunctionalReductionSATBuilder> satBuilder;
  llvm::DenseMap<Value, int> satVars;
  llvm::DenseSet<Value> encodedValues;
  // Monotonic counter for auxiliary SAT variables introduced by definitional
  // CNF encodings, currently used for variadic XOR.
  int nextFreshVar = 0;
  Stats stats;
};

FunctionalReductionSATBuilder::FunctionalReductionSATBuilder(
    IncrementalSATSolver &solver, llvm::DenseMap<Value, int> &satVars,
    llvm::DenseSet<Value> &encodedValues, int &nextFreshVar)
    : solver(solver), satVars(satVars), encodedValues(encodedValues),
      nextFreshVar(nextFreshVar) {}

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

EquivResult FunctionalReductionSolver::verifyEquivalence(Value lhs, Value rhs,
                                                         bool inverted) {

  if (testTransformation) {
    if (matchesTestEquivClass(lhs, rhs))
      return EquivResult::Proved;
    return EquivResult::Unknown;
  }
  assert(satBuilder && "SAT builder must be initialized before verification");
  // SAT-based equivalence checking builds a miter for the two candidate nodes
  // and proves that no input assignment can make them differ.
  return satBuilder->verify(lhs, rhs, inverted);
}

void FunctionalReductionSolver::initializeSATState() {
  assert(satSolver && "SAT solver must be initialized before SAT state setup");

  satVars.clear();
  encodedValues.clear();
  satVars.reserve(allValues.size());
  for (auto [index, value] : llvm::enumerate(allValues))
    satVars[value] = index + 1;
  nextFreshVar = allValues.size();
  satSolver->reserveVars(allValues.size());

  satBuilder = std::make_unique<FunctionalReductionSATBuilder>(
      *satSolver, satVars, encodedValues, nextFreshVar);
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
  // - AIG operations: add to allValues for simulation
  // - Unknown operations: treat as inputs (assign random patterns)
  module.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!result.getType().isInteger(1))
        continue;

      allValues.push_back(result);
      if (!op->hasTrait<OpTrait::ConstantLike>() &&
          !isFunctionalReductionSimulatableOp(op)) {
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
      .Case<BooleanLogicOpInterface>([&](auto op) {
        return op.evaluateBooleanLogic([&](unsigned i) -> const APInt & {
          return simSignatures.at(op.getInput(i));
        });
      })
      .Case<comb::AndOp>([&](auto op) {
        APInt result = APInt::getAllOnes(numPatterns);
        for (auto input : op.getInputs())
          result &= simSignatures.at(input);
        return result;
      })
      .Case<comb::OrOp>([&](auto op) {
        APInt result = APInt::getZero(numPatterns);
        for (auto input : op.getInputs())
          result |= simSignatures.at(input);
        return result;
      })
      .Case<comb::XorOp>([&](auto op) {
        APInt result = APInt::getZero(numPatterns);
        for (auto input : op.getInputs())
          result ^= simSignatures.at(input);
        return result;
      })
      .Case([&](hw::ConstantOp op) {
        return op.getValue().isZero() ? APInt::getZero(numPatterns)
                                      : APInt::getAllOnes(numPatterns);
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
  // Map from canonical signature to list of {value, inverted pairs}
  // Inverted signals share the same canonical signature since inversion
  // is zero cost in synthesis
  llvm::MapVector<llvm::APInt, SmallVector<std::pair<Value, bool>>> sigGroups;
  for (auto value : allValues) {
    auto signature = simSignatures.at(value);
    bool inverted = false;
    if (signature.isNegative()) {
      inverted = true;
      signature.flipAllBits();
    }
    sigGroups[signature].push_back({value, inverted});
  }

  // Build equivalence candidates for groups with >1 member.
  // Re-normalize so inverted is relative to representative (first member)
  for (auto &[hash, members] : sigGroups) {
    if (members.size() <= 1)
      continue;
    bool repInverted = members.front().second;
    for (auto &[_, inv] : members)
      inv ^= repInverted;
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
    auto [representative, repInversion] = members.front();
    assert(!repInversion && "representative must not be inverted");
    (void)repInversion;
    auto &provenMembers = provenEquivalences[representative];
    // Representative is the canonical node for this class. Members can be
    // inverted relative to the representative, tracked by the inversion flag
    for (auto [member, inverted] :
         llvm::ArrayRef<std::pair<Value, bool>>(members).drop_front()) {
      EquivResult result = verifyEquivalence(representative, member, inverted);
      if (result == EquivResult::Proved) {
        stats.numProvedEquiv++;
        provenMembers.push_back({member, inverted});
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

    builder.setInsertionPointAfterValue(members.back().first);

    SmallVector<Value> operands;
    operands.reserve(members.size() + 1);
    operands.push_back(representative);

    llvm::SmallPtrSet<Operation *, 8> createdInverters;
    for (auto [member, inverted] : members) {
      if (!inverted) {
        operands.push_back(member);
        continue;
      }
      // If the member is inverted relative to the representative, we
      // create an inverter for the choice operand
      auto inverter =
          aig::AndInverterOp::create(builder, member.getLoc(), member, true);
      createdInverters.insert(inverter);
      operands.push_back(inverter.getResult());
    }

    auto choice = synth::ChoiceOp::create(builder, representative.getLoc(),
                                          representative.getType(), operands);

    // If there is an inverted member, we need to create an inverter for the
    // choice result as well
    auto choiceNot = createdInverters.empty()
                         ? nullptr
                         : aig::AndInverterOp::create(builder, choice.getLoc(),
                                                      choice, true);

    stats.numMergedNodes += members.size() + 1;

    auto replaceValue = [&](Value value, bool inverted) {
      if (inverted)
        value.replaceUsesWithIf(choiceNot, [&](OpOperand &use) {
          // Only replace uses that are not the inverters we just created. This
          // is necessary to avoid creatng an immediate cycle when merging an
          // inverted node into its representative.
          return !createdInverters.contains(use.getOwner());
        });
      else
        value.replaceAllUsesExcept(choice, choice);
    };

    representative.replaceAllUsesExcept(choice, choice);
    for (auto [value, inverted] : members)
      replaceValue(value, inverted);
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

  if (!testTransformation && !satSolver) {
    module->emitError()
        << "FunctionalReduction requires a SAT solver, but none is "
           "available in this build";
    return failure();
  }

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
  if (!testTransformation)
    initializeSATState();
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
    if (conflictLimit < -1) {
      module.emitError()
          << "'conflict-limit' must be greater than or equal to -1";
      return signalPassFailure();
    }

    std::unique_ptr<IncrementalSATSolver> satSolver;
    if (!testTransformation) {
      satSolver = createFunctionalReductionSATSolver(this->satSolver);
      if (!satSolver) {
        module.emitError() << "unsupported or unavailable SAT solver '"
                           << this->satSolver
                           << "' (expected auto, z3, or cadical)";
        return signalPassFailure();
      }
      satSolver->setConflictLimit(static_cast<int>(conflictLimit));
    }

    FunctionalReductionSolver fcSolver(module, numRandomPatterns, seed,
                                       testTransformation,
                                       std::move(satSolver));
    auto stats = fcSolver.run();
    if (failed(stats))
      return signalPassFailure();
    updateStats(*stats);
    if (stats->numMergedNodes == 0)
      markAllAnalysesPreserved();
  }
};

} // namespace
