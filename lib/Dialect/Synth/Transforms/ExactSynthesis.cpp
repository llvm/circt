//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SAT-based exact synthesis for small Boolean truth
// tables.
//
// References:
//  "Practical exact synthesis", M. Soeken et al., DATE 2018
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Naming.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include <array>
#include <optional>
#include <string>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_EXACTSYNTHESIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-exact-synthesis"

namespace {

static constexpr unsigned kMaxExactSynthesisInputs = 6;
static constexpr unsigned kMaxExactSearchArea = 32;

static SmallString<32> formatTruthTable(const APInt &truthTable) {
  SmallString<32> text;
  truthTable.toStringUnsigned(text, /*Radix=*/16);
  return text;
}

using BooleanLogicConcept =
    circt::synth::detail::BooleanLogicOpInterfaceInterfaceTraits::Concept;

struct ExactCandidatePolicy {
  bool mayUseConstantSource = true;
  bool enumerateInputInversions = true;
};

//===----------------------------------------------------------------------===//
// Exact network model
//===----------------------------------------------------------------------===//

class ExactNodeInfo;

struct ExactSignalRef {
  // Source 0 is the constant false value. Sources 1..numInputs are primary
  // inputs. Later sources are the synthesized steps, in order.
  unsigned source = 0;
  bool inverted = false;
};

struct ExactNetworkStep {
  const ExactNodeInfo *info = nullptr;
  SmallVector<ExactSignalRef, 3> fanins;

  unsigned getInversionMask() const;
  bool operator<(const ExactNetworkStep &rhs) const;
};

struct ExactNetwork {
  unsigned numInputs = 0;
  SmallVector<ExactNetworkStep, 4> steps;
  ExactSignalRef output;
};

using ExactCandidate = ExactNetworkStep;

/// One primitive that the exact search may use.
class ExactNodeInfo {
public:
  ExactNodeInfo(OperationName opName, unsigned arity, bool commutative,
                const BooleanLogicConcept *iface,
                ExactCandidatePolicy candidatePolicy)
      : opName(opName), arity(arity), commutative(commutative), iface(iface),
        candidatePolicy(candidatePolicy) {}

  OperationName getOperationName() const { return opName; }
  unsigned getArity() const { return arity; }
  bool isCommutative() const { return commutative; }
  bool mayUseConstantSource() const {
    return candidatePolicy.mayUseConstantSource;
  }
  bool shouldEnumerateInputInversions() const {
    return candidatePolicy.enumerateInputInversions;
  }

  /// Emit clauses for `selector => outLit == f(fanins...)`.
  void emitConditionedCNF(
      IncrementalSATSolver &solver, int selector, int outLit,
      const ExactCandidate &candidate, unsigned minterm,
      llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
          getSourceLiteral) const;
  Value materialize(OpBuilder &builder, Location loc, ArrayRef<Value> operands,
                    ArrayRef<bool> inverted) const {
    return iface->createBooleanLogicOp(builder, loc, operands, inverted);
  }

private:
  OperationName opName;
  unsigned arity;
  bool commutative;
  const BooleanLogicConcept *iface;
  ExactCandidatePolicy candidatePolicy;
};

struct ExactSynthesisPolicy {
  SmallVector<ExactNodeInfo, 4> primitiveInfos;
};

static std::unique_ptr<IncrementalSATSolver>
createExactSynthesisSATSolver(StringRef backend) {
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

static SmallString<64>
formatPrimitiveSummary(const ExactSynthesisPolicy &policy) {
  SmallString<64> text;
  llvm::raw_svector_ostream os(text);
  llvm::interleaveComma(policy.primitiveInfos, os, [&](const auto &info) {
    os << info.getOperationName().getStringRef() << ":" << info.getArity();
  });
  return text;
}

unsigned ExactNetworkStep::getInversionMask() const {
  unsigned mask = 0;
  for (auto [index, fanin] : llvm::enumerate(fanins))
    if (fanin.inverted)
      mask |= 1u << index;
  return mask;
}

bool ExactNetworkStep::operator<(const ExactNetworkStep &rhs) const {
  if (fanins.size() != rhs.fanins.size())
    return fanins.size() < rhs.fanins.size();
  for (size_t i = 0, e = fanins.size(); i != e; ++i) {
    if (fanins[i].source != rhs.fanins[i].source)
      return fanins[i].source < rhs.fanins[i].source;
  }
  if (info != rhs.info) {
    if (info->getOperationName() != rhs.info->getOperationName())
      return info->getOperationName().getStringRef() <
             rhs.info->getOperationName().getStringRef();
  }
  return getInversionMask() < rhs.getInversionMask();
}

static std::optional<ExactNetwork> synthesizeDirect(unsigned numInputs,
                                                    const APInt &target) {
  ExactNetwork network;
  network.numInputs = numInputs;
  if (target.isZero()) {
    network.output = {0, false};
    return network;
  }
  if (target.isAllOnes()) {
    network.output = {0, true};
    return network;
  }

  for (unsigned input = 0; input != numInputs; ++input) {
    APInt mask = circt::createVarMask(numInputs, input, true);
    if (target == mask) {
      network.output = {1 + input, false};
      return network;
    }
    APInt invertedMask = mask;
    invertedMask.flipAllBits();
    if (target == invertedMask) {
      network.output = {1 + input, true};
      return network;
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Enumeration
//===----------------------------------------------------------------------===//

/// Enumerates the primitive instances that may be placed at one SAT step.
class ExactCandidateEnumerator {
public:
  void enumerate(const ExactSynthesisPolicy &policy, unsigned availableSources,
                 SmallVectorImpl<ExactCandidate> &candidates) const;

private:
  static void enumerateCommutativeOperandSources(
      unsigned availableSources, unsigned arity, unsigned currentArity,
      unsigned nextSource, SmallVectorImpl<unsigned> &sources,
      llvm::function_ref<void(ArrayRef<unsigned>)> emit);

  static void enumerateOrderedOperandSources(
      unsigned availableSources, unsigned arity, unsigned firstSource,
      unsigned currentArity, SmallVectorImpl<unsigned> &sources,
      llvm::function_ref<void(ArrayRef<unsigned>)> emit);

  static void
  enumerateNodeCandidates(const ExactNodeInfo &info, unsigned availableSources,
                          SmallVectorImpl<ExactCandidate> &candidates);
};

//===----------------------------------------------------------------------===//
// Materialization
//===----------------------------------------------------------------------===//

/// Lowers a solved exact network back into current Synth IR.
class ExactNetworkMaterializer {
public:
  ExactNetworkMaterializer(OpBuilder &builder, Location loc,
                           ArrayRef<Value> inputs);

  Value materialize(const ExactNetwork &network);

private:
  Value getConstant(bool value);
  Value getRawSignal(ExactSignalRef signal, ArrayRef<Value> stepValues);
  Value materializeInverter(Value input);

  OpBuilder &builder;
  Location loc;
  ArrayRef<Value> inputs;
  std::array<Value, 2> constValues;
};

//===----------------------------------------------------------------------===//
// SAT search
//===----------------------------------------------------------------------===//

class GenericExactSATProblem {
public:
  GenericExactSATProblem(const ExactSynthesisPolicy &policy,
                         IncrementalSATSolver &solver, unsigned numInputs,
                         const APInt &target, unsigned numSteps);

  std::optional<ExactNetwork> solve();

private:
  int newVar();
  /// Return the SAT variable for one source under one concrete input pattern.
  int getSourceValueVar(unsigned source, unsigned minterm) const;
  /// Return that same variable as a literal, optionally negated.
  int getSourceLiteral(unsigned source, unsigned minterm, bool inverted) const;

  /// Build the SAT model for "can this truth table be implemented with exactly
  /// `numSteps` internal nodes?".
  bool buildEncoding();

  /// Say what each step output must be for each candidate and each input
  /// pattern.
  void addCandidateSemanticsConstraints();

  /// Force every step except the root to feed some later selected step.
  void addUseAllStepsConstraints();

  ExactNetwork decodeModel() const;

  const ExactSynthesisPolicy &policy;
  IncrementalSATSolver &solver;
  unsigned numInputs;
  APInt target;
  unsigned numSteps;
  unsigned numMinterms;
  unsigned totalSources;
  int rootInvertVar = 0;
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  SmallVector<SmallVector<ExactCandidate, 64>, 8> stepCandidates;
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
  ExactCandidateEnumerator enumerator;
};

void ExactNodeInfo::emitConditionedCNF(
    IncrementalSATSolver &solver, int selector, int outLit,
    const ExactCandidate &candidate, unsigned minterm,
    llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
        getSourceLiteral) const {
  auto addConditionedClause = [&](ArrayRef<int> literals) {
    SmallVector<int, 8> clause;
    clause.reserve(literals.size() + 1);
    clause.push_back(-selector);
    clause.append(literals.begin(), literals.end());
    solver.addClause(clause);
  };

  // Apply this candidate's edge inversions in the literals we pass to the
  // primitive. Guard every primitive clause with the selector so unselected
  // candidates do not constrain the step output.
  SmallVector<int, 4> inputLits;
  inputLits.reserve(getArity());
  for (unsigned operand = 0; operand != getArity(); ++operand) {
    const auto &fanin = candidate.fanins[operand];
    inputLits.push_back(
        getSourceLiteral(fanin.source, minterm, fanin.inverted));
  }
  iface->emitCNFWithoutInversion(outLit, inputLits, addConditionedClause,
                                 [&] { return solver.newVar(); });
}

void ExactCandidateEnumerator::enumerate(
    const ExactSynthesisPolicy &policy, unsigned availableSources,
    SmallVectorImpl<ExactCandidate> &candidates) const {
  candidates.clear();
  for (const auto &info : policy.primitiveInfos)
    enumerateNodeCandidates(info, availableSources, candidates);
  llvm::sort(candidates);
  LDBG() << "Enumerated " << candidates.size()
         << " candidates with availableSources=" << availableSources << "\n";
}

void ExactCandidateEnumerator::enumerateCommutativeOperandSources(
    unsigned availableSources, unsigned arity, unsigned currentArity,
    unsigned nextSource, SmallVectorImpl<unsigned> &sources,
    llvm::function_ref<void(ArrayRef<unsigned>)> emit) {
  if (currentArity == arity) {
    emit(sources);
    return;
  }

  for (unsigned source = nextSource; source < availableSources; ++source) {
    sources.push_back(source);
    enumerateCommutativeOperandSources(
        availableSources, arity, currentArity + 1, source + 1, sources, emit);
    sources.pop_back();
  }
}

void ExactCandidateEnumerator::enumerateOrderedOperandSources(
    unsigned availableSources, unsigned arity, unsigned firstSource,
    unsigned currentArity, SmallVectorImpl<unsigned> &sources,
    llvm::function_ref<void(ArrayRef<unsigned>)> emit) {
  if (currentArity == arity) {
    emit(sources);
    return;
  }

  // Ordered nodes such as DOT must keep operand order. They may also reuse the
  // same source more than once.
  for (unsigned source = firstSource; source < availableSources; ++source) {
    sources.push_back(source);
    enumerateOrderedOperandSources(availableSources, arity, firstSource,
                                   currentArity + 1, sources, emit);
    sources.pop_back();
  }
}

void ExactCandidateEnumerator::enumerateNodeCandidates(
    const ExactNodeInfo &info, unsigned availableSources,
    SmallVectorImpl<ExactCandidate> &candidates) {
  SmallVector<unsigned, 3> sources;
  auto emitCandidate = [&](ArrayRef<unsigned> operandSources) {
    unsigned numInversionMasks =
        info.shouldEnumerateInputInversions() ? (1u << info.getArity()) : 1;
    for (unsigned invMask = 0; invMask != numInversionMasks; ++invMask) {
      ExactCandidate candidate;
      candidate.info = &info;
      for (auto [index, source] : llvm::enumerate(operandSources))
        candidate.fanins.push_back(
            {source, static_cast<bool>(invMask & (1u << index))});
      candidates.push_back(std::move(candidate));
    }
  };

  // At this point we only choose source numbers and edge inversions. The
  // primitive's Boolean behavior is added later as CNF.
  unsigned firstSource = info.mayUseConstantSource() ? 0 : 1;
  if (info.isCommutative()) {
    // Commutative nodes use sorted, distinct sources. This removes operand
    // permutations and skips repeated-source cases. For the current
    // commutative Synth primitives, repeated sources reduce to constants,
    // projections, inversions, or distinct-source candidates with constants.
    // FIXME: Derive this repeated-source pruning from the primitive truth table
    // or make it an explicit primitive policy before adding more commutative
    // primitives here.
    enumerateCommutativeOperandSources(availableSources, info.getArity(),
                                       /*currentArity=*/0, firstSource, sources,
                                       emitCandidate);
    return;
  }

  enumerateOrderedOperandSources(availableSources, info.getArity(), firstSource,
                                 /*currentArity=*/0, sources, emitCandidate);
}

ExactNetworkMaterializer::ExactNetworkMaterializer(OpBuilder &builder,
                                                   Location loc,
                                                   ArrayRef<Value> inputs)
    : builder(builder), loc(loc), inputs(inputs) {}

Value ExactNetworkMaterializer::materialize(const ExactNetwork &network) {
  SmallVector<Value, 4> stepValues;
  stepValues.reserve(network.steps.size());
  for (const auto &step : network.steps) {
    assert(step.info && "network step must carry node info");
    const auto &info = *step.info;
    SmallVector<Value, 3> operands;
    SmallVector<bool, 3> inverted;
    operands.reserve(info.getArity());
    inverted.reserve(info.getArity());
    for (const auto &fanin : step.fanins) {
      operands.push_back(getRawSignal(fanin, stepValues));
      inverted.push_back(fanin.inverted);
    }
    stepValues.push_back(info.materialize(builder, loc, operands, inverted));
  }

  if (network.output.source == 0)
    return getConstant(network.output.inverted);

  Value result = getRawSignal(network.output, stepValues);
  if (!network.output.inverted)
    return result;
  return materializeInverter(result);
}

Value ExactNetworkMaterializer::getConstant(bool value) {
  if (constValues[value])
    return constValues[value];
  return constValues[value] =
             hw::ConstantOp::create(builder, loc, APInt(1, value));
}

Value ExactNetworkMaterializer::getRawSignal(ExactSignalRef signal,
                                             ArrayRef<Value> stepValues) {
  if (signal.source == 0)
    return getConstant(false);
  if (signal.source <= inputs.size())
    return inputs[signal.source - 1];

  unsigned stepIndex = signal.source - (inputs.size() + 1);
  assert(stepIndex < stepValues.size() && "invalid synthesized step index");
  return stepValues[stepIndex];
}

Value ExactNetworkMaterializer::materializeInverter(Value input) {
  return aig::AndInverterOp::create(builder, loc, input, true);
}

GenericExactSATProblem::GenericExactSATProblem(
    const ExactSynthesisPolicy &policy, IncrementalSATSolver &solver,
    unsigned numInputs, const APInt &target, unsigned numSteps)
    : policy(policy), solver(solver), numInputs(numInputs), target(target),
      numSteps(numSteps), numMinterms(1u << numInputs),
      totalSources(1 + numInputs + numSteps) {}

std::optional<ExactNetwork> GenericExactSATProblem::solve() {
  LDBG() << "SAT solve start: inputs=" << numInputs << " steps=" << numSteps
         << " minterms=" << numMinterms << " target=0x"
         << formatTruthTable(target) << "\n";
  if (!buildEncoding())
    return std::nullopt;
  auto result = solver.solve();
  LDBG() << "SAT solve result: "
         << (result == IncrementalSATSolver::kSAT     ? "SAT"
             : result == IncrementalSATSolver::kUNSAT ? "UNSAT"
                                                      : "UNKNOWN")
         << "\n";
  if (result != IncrementalSATSolver::kSAT)
    return std::nullopt;
  return decodeModel();
}

int GenericExactSATProblem::newVar() { return solver.newVar(); }

int GenericExactSATProblem::getSourceValueVar(unsigned source,
                                              unsigned minterm) const {
  return sourceValueVars[source][minterm];
}

int GenericExactSATProblem::getSourceLiteral(unsigned source, unsigned minterm,
                                             bool inverted) const {
  int lit = getSourceValueVar(source, minterm);
  return inverted ? -lit : lit;
}

bool GenericExactSATProblem::buildEncoding() {
  // A minterm is one input assignment. For every source and every minterm,
  // create one SAT variable that means "this source is true for this minterm".
  sourceValueVars.resize(totalSources);
  for (unsigned source = 0; source != totalSources; ++source) {
    sourceValueVars[source].reserve(numMinterms);
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      sourceValueVars[source].push_back(newVar());
  }

  // Source 0 is always false, for every input pattern.
  for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
    solver.addClause({-getSourceValueVar(0, minterm)});

  // Fix each primary input source to the matching bit of the minterm number.
  // For example, minterm 5 (0b101) makes input 0 and input 2 true.
  for (unsigned input = 0; input != numInputs; ++input)
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({((minterm >> input) & 1)
                            ? getSourceValueVar(1 + input, minterm)
                            : -getSourceValueVar(1 + input, minterm)});

  // Each internal step chooses exactly one candidate primitive instance.
  stepCandidates.resize(numSteps);
  stepSelectionVars.resize(numSteps);
  for (unsigned step = 0; step != numSteps; ++step) {
    unsigned availableSources = 1 + numInputs + step;
    enumerator.enumerate(policy, availableSources, stepCandidates[step]);
    LDBG() << "  step " << step << ": availableSources=" << availableSources
           << " candidates=" << stepCandidates[step].size() << "\n";
    if (stepCandidates[step].empty())
      return false;

    auto &selectionVars = stepSelectionVars[step];
    selectionVars.reserve(stepCandidates[step].size());
    for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
      selectionVars.push_back(newVar());
    addExactlyOneClauses(
        selectionVars, [&](ArrayRef<int> clause) { solver.addClause(clause); },
        [&] { return newVar(); });
  }

  // Add the primitive semantics and require every non-root step to feed a later
  // selected step.
  // TODO: Add symmetry breaking constraints to reduce the search space.
  addCandidateSemanticsConstraints();
  addUseAllStepsConstraints();

  // The root is the last internal step with one optional output inversion. The
  // inversion bit is shared by all minterms, so it chooses one global polarity.
  unsigned rootSource = totalSources - 1;
  rootInvertVar = newVar();
  for (unsigned minterm = 0; minterm != numMinterms; ++minterm) {
    int rootLit = getSourceValueVar(rootSource, minterm);
    if (target[minterm]) {
      // target = root xor rootInvert, with target fixed to true.
      solver.addClause({rootLit, rootInvertVar});
      solver.addClause({-rootLit, -rootInvertVar});
    } else {
      // target = root xor rootInvert, with target fixed to false.
      solver.addClause({rootLit, -rootInvertVar});
      solver.addClause({-rootLit, rootInvertVar});
    }
  }
  return true;
}

void GenericExactSATProblem::addCandidateSemanticsConstraints() {
  for (unsigned step = 0; step != numSteps; ++step) {
    unsigned outSource = 1 + numInputs + step;
    const auto &selectionVars = stepSelectionVars[step];
    // If a candidate is selected, its output must match its primitive
    // semantics for every minterm.
    for (auto [candidateIndex, candidate] :
         llvm::enumerate(stepCandidates[step]))
      for (unsigned minterm = 0; minterm != numMinterms; ++minterm) {
        assert(candidate.info && "candidate must carry node info");
        candidate.info->emitConditionedCNF(
            solver, selectionVars[candidateIndex],
            getSourceValueVar(outSource, minterm), candidate, minterm,
            [&](unsigned source, unsigned currentMinterm, bool inverted) {
              return getSourceLiteral(source, currentMinterm, inverted);
            });
      }
  }
}

void GenericExactSATProblem::addUseAllStepsConstraints() {
  for (unsigned step = 0; step + 1 < numSteps; ++step) {
    unsigned source = 1 + numInputs + step;
    SmallVector<int, 32> users;
    for (unsigned userStep = step + 1; userStep != numSteps; ++userStep)
      for (auto [candidateIndex, candidate] :
           llvm::enumerate(stepCandidates[userStep]))
        if (llvm::any_of(candidate.fanins, [&](const ExactSignalRef &fanin) {
              return fanin.source == source;
            }))
          users.push_back(stepSelectionVars[userStep][candidateIndex]);

    // Without this, an area-bounded search could satisfy the target with dead
    // logic that never reaches the root.
    solver.addClause(users);
  }
}

ExactNetwork GenericExactSATProblem::decodeModel() const {
  ExactNetwork network;
  network.numInputs = numInputs;
  network.steps.reserve(numSteps);
  for (unsigned step = 0; step != numSteps; ++step) {
    const auto &selectionVars = stepSelectionVars[step];
    const auto &candidates = stepCandidates[step];
    for (size_t i = 0, e = selectionVars.size(); i != e; ++i) {
      if (solver.val(selectionVars[i]) != selectionVars[i])
        continue;
      network.steps.push_back(candidates[i]);
      break;
    }
  }
  network.output = {1 + numInputs + numSteps - 1,
                    solver.val(rootInvertVar) == rootInvertVar};
  LDBG() << "Decoded network with " << network.steps.size()
         << " steps, rootSource=" << network.output.source
         << " rootInvert=" << network.output.inverted << "\n";
  return network;
}

static FailureOr<Value>
exactSynthesizeAreaMinimized(OpBuilder &builder, Location loc, APInt truthTable,
                             ArrayRef<Value> operands,
                             const ExactSynthesisPolicy &policy,
                             StringRef satSolver) {
  ExactNetworkMaterializer materializer(builder, loc, operands);
  unsigned numInputs = operands.size();
  LDBG() << "Exact synthesis request: inputs=" << numInputs << " truthTable=0x"
         << formatTruthTable(truthTable)
         << " allowed-primitives=" << formatPrimitiveSummary(policy)
         << " sat-solver=" << satSolver << "\n";

  if (policy.primitiveInfos.empty())
    return failure();

  LDBG() << "Trying direct synthesis for target=0x"
         << formatTruthTable(truthTable) << "\n";
  auto network = synthesizeDirect(numInputs, truthTable);
  if (network) {
    LDBG() << "Using direct synthesis result\n";
    return materializer.materialize(*network);
  }

  for (unsigned area = 1; area <= kMaxExactSearchArea; ++area) {
    LDBG() << "Trying area=" << area << "\n";
    auto solver = createExactSynthesisSATSolver(satSolver);
    if (!solver)
      return failure();
    GenericExactSATProblem problem(policy, *solver, numInputs, truthTable,
                                   area);
    auto solved = problem.solve();
    if (!solved) {
      LDBG() << "Area " << area << " has no solution\n";
      continue;
    }
    LDBG() << "Found solution at area=" << area << "\n";
    return materializer.materialize(*solved);
  }
  LDBG() << "No exact solution found up to area limit " << kMaxExactSearchArea
         << "\n";
  return failure();
}

//===----------------------------------------------------------------------===//
// Rewrite Pass
//===----------------------------------------------------------------------===//

struct ExactSynthesisPattern : public OpRewritePattern<comb::TruthTableOp> {
  ExactSynthesisPattern(MLIRContext *context,
                        const ExactSynthesisPolicy &policy, StringRef satSolver)
      : OpRewritePattern(context), policy(policy), satSolver(satSolver.str()) {}

  LogicalResult matchAndRewrite(comb::TruthTableOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() > kMaxExactSynthesisInputs)
      return failure();

    SmallVector<Value> operands;
    operands.reserve(op.getInputs().size());
    // comb.truth_table indexes the first operand as the most significant input
    // bit. The exact synthesis truth-table utilities use input 0 as the least
    // significant bit.
    for (Value operand : llvm::reverse(op.getInputs()))
      operands.push_back(operand);

    APInt truthTable(op.getLookupTable().size(), 0);
    for (size_t index = 0, e = op.getLookupTable().size(); index != e; ++index)
      truthTable.setBitVal(index, op.getLookupTable()[index]);
    auto result = exactSynthesizeAreaMinimized(
        rewriter, op.getLoc(), truthTable, operands, policy, satSolver);
    if (failed(result))
      return failure();

    replaceOpAndCopyNamehint(rewriter, op, *result);
    return success();
  }

private:
  const ExactSynthesisPolicy &policy;
  std::string satSolver;
};

struct ExactSynthesisPass
    : public circt::synth::impl::ExactSynthesisBase<ExactSynthesisPass> {
  using ExactSynthesisBase::ExactSynthesisBase;

  FailureOr<ExactSynthesisPolicy> parsePolicy(MLIRContext *context) const {
    ExactSynthesisPolicy policy;

    if (allowedOps.empty()) {
      emitError(UnknownLoc::get(context))
          << "synth-exact-synthesis requires at least one "
             "'allowed-ops=name:arity' entry";
      return failure();
    }

    for (const std::string &allowedOp : allowedOps) {
      StringRef spelling = allowedOp;
      auto parts = spelling.split(':');
      StringRef name = parts.first.trim();
      StringRef arityText = parts.second.trim();
      auto registeredInfo = RegisteredOperationName::lookup(name, context);
      if (!registeredInfo) {
        emitError(UnknownLoc::get(context))
            << "unknown allowed exact-synthesis op '" << name << "'";
        return failure();
      }
      auto *iface = registeredInfo->getInterface<BooleanLogicOpInterface>();
      if (!iface) {
        emitError(UnknownLoc::get(context))
            << "op '" << name << "' does not implement BooleanLogicOpInterface";
        return failure();
      }

      unsigned arity = 0;
      if (arityText.empty() || arityText.getAsInteger(10, arity)) {
        emitError(UnknownLoc::get(context))
            << "expected allowed exact-synthesis op in 'name:arity' form, "
               "e.g. '"
            << name << ":3'";
        return failure();
      }
      if (arity < 2 || arity > kMaxExactSynthesisInputs) {
        emitError(UnknownLoc::get(context))
            << "unsupported arity " << arity << " for '" << name << "'";
        return failure();
      }
      if (!iface->supportsNumInputs(arity)) {
        emitError(UnknownLoc::get(context))
            << "op '" << name << "' does not support exact-synthesis arity "
            << arity;
        return failure();
      }
      OperationName opName(*registeredInfo);
      ExactCandidatePolicy candidatePolicy;
      if (name == XorInverterOp::getOperationName() && arity == 2) {
        // For binary XOR, constants and input inversions only change the result
        // into a constant, projection, or complemented XOR. Direct synthesis,
        // edge inversions, and root inversion already cover those cases.
        candidatePolicy.mayUseConstantSource = false;
        candidatePolicy.enumerateInputInversions = false;
      }
      if (llvm::any_of(policy.primitiveInfos, [&](const ExactNodeInfo &info) {
            return info.getOperationName() == opName &&
                   info.getArity() == arity;
          })) {
        emitError(UnknownLoc::get(context))
            << "duplicate allowed exact-synthesis op '" << spelling << "'";
        return failure();
      }
      policy.primitiveInfos.emplace_back(opName, arity,
                                         iface->areInputsPermutationInvariant(),
                                         iface, candidatePolicy);
    }
    return policy;
  }

  LogicalResult initialize(MLIRContext *context) override {
    auto parsedPolicy = parsePolicy(context);
    if (failed(parsedPolicy))
      return failure();
    policy = *parsedPolicy;

    if (!createExactSynthesisSATSolver(satSolver)) {
      emitError(UnknownLoc::get(context))
          << "unsupported or unavailable SAT solver '" << satSolver
          << "' (expected auto, z3, or cadical)";

      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExactSynthesisPattern>(&getContext(), policy, satSolver);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }

private:
  ExactSynthesisPolicy policy;
};

} // namespace
