//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a DAG-based boolean matching cut rewriting algorithm for
// applications like technology/LUT mapping and combinational logic
// optimization. The algorithm uses priority cuts and NPN
// (Negation-Permutation-Negation) canonical forms to efficiently match cuts
// against rewriting patterns.
//
// References:
//  "Combinational and Sequential Mapping with Priority Cuts", Alan Mishchenko,
//  Sungmin Cho, Satrajit Chatterjee and Robert Brayton, ICCAD 2007
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/CutRewriter.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/TruthTable.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#define DEBUG_TYPE "synth-cut-rewriter"

using namespace circt;
using namespace circt::synth;

static bool isSupportedLogicOp(mlir::Operation *op) {
  // Check if the operation is a combinational operation that can be simulated
  // TODO: Extend this to allow comb.and/xor/or as well.
  return isa<aig::AndInverterOp>(op);
}

static void simulateLogicOp(Operation *op, DenseMap<Value, llvm::APInt> &eval) {
  assert(isSupportedLogicOp(op) &&
         "Operation must be a supported logic operation for simulation");

  // Simulate the operation by evaluating its inputs and computing the output
  // This is a simplified simulation for demonstration purposes
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    SmallVector<llvm::APInt, 2> inputs;
    inputs.reserve(andOp.getInputs().size());
    for (auto input : andOp.getInputs()) {
      auto it = eval.find(input);
      if (it == eval.end())
        llvm::report_fatal_error("Input value not found in evaluation map");
      inputs.push_back(it->second);
    }
    // Evaluate the and inverter
    eval[andOp.getResult()] = andOp.evaluate(inputs);
    return;
  }

  llvm::report_fatal_error(
      "Unsupported operation for simulation. isSupportedLogicOp should "
      "be used to check if the operation can be simulated.");
}

// Return true if the value is always a cut input.
static bool isAlwaysCutInput(Value value) {
  auto *op = value.getDefiningOp();
  // If the value has no defining operation, it is an input
  if (!op)
    return true;

  if (op->hasTrait<OpTrait::ConstantLike>()) {
    // Constant values are never cut inputs.
    return false;
  }

  return !isSupportedLogicOp(op);
}

// Return true if the new area/delay is better than the old area/delay in the
// context of the given strategy.
static bool compareDelayAndArea(OptimizationStrategy strategy, double newArea,
                                ArrayRef<DelayType> newDelay, double oldArea,
                                ArrayRef<DelayType> oldDelay) {
  if (OptimizationStrategyArea == strategy) {
    // Compare by area first.
    return newArea < oldArea || (newArea == oldArea && newDelay < oldDelay);
  }
  if (OptimizationStrategyTiming == strategy) {
    // Compare by delay first.
    return newDelay < oldDelay || (newDelay == oldDelay && newArea < oldArea);
  }
  llvm_unreachable("Unknown mapping strategy");
}

LogicalResult
circt::synth::topologicallySortLogicNetwork(mlir::Operation *topOp) {

  auto isOperationReady = [](Value value, Operation *op) -> bool {
    // Topologically sort simulatable ops and purely
    // dataflow ops. Other operations can be scheduled.
    return !(isSupportedLogicOp(op) ||
             isa<comb::ExtractOp, comb::ReplicateOp, comb::ConcatOp>(op));
  };

  auto result = topologicallySortGraphRegionBlocks(topOp, isOperationReady);
  if (failed(result))
    return mlir::emitError(topOp->getLoc(),
                           "failed to sort operations topologically");
  return success();
}

/// Get the truth table for an op.
template <typename OpRange>
FailureOr<BinaryTruthTable> static computeTruthTable(
    mlir::ValueRange values, const OpRange &ops,
    const llvm::SmallSetVector<mlir::Value, 4> &inputArgs) {
  // Create a truth table for the operation
  int64_t numInputs = inputArgs.size();
  int64_t numOutputs = values.size();
  if (LLVM_UNLIKELY(numOutputs != 1 || numInputs >= maxTruthTableInputs)) {
    if (numOutputs == 0)
      return BinaryTruthTable(numInputs, 0);
    if (numInputs >= maxTruthTableInputs)
      return mlir::emitError(values.front().getLoc(),
                             "Truth table is too large");
    return mlir::emitError(values.front().getLoc(),
                           "Multiple outputs are not supported yet");
  }

  // Create a truth table with the given number of inputs and outputs
  BinaryTruthTable truthTable(numInputs, numOutputs);
  // The truth table size is 2^numInputs
  // Create a map to evaluate the operation
  DenseMap<Value, APInt> eval;
  for (uint32_t i = 0; i < numInputs; ++i)
    eval[inputArgs[i]] = circt::createVarMask(numInputs, i, true);
  // Simulate the operation
  for (auto *op : ops) {
    if (op->getNumResults() == 0)
      continue; // Skip operations with no results
    if (!isSupportedLogicOp(op))
      return op->emitError("Unsupported operation for truth table simulation");

    // Simulate the operation
    simulateLogicOp(op, eval);
  }
  // TODO: Currently numOutputs is always 1, so we can just return the first
  // one.
  return BinaryTruthTable(numInputs, 1, eval[values[0]]);
}

FailureOr<BinaryTruthTable> circt::synth::getTruthTable(ValueRange values,
                                                        Block *block) {
  // Get the input arguments from the block
  llvm::SmallSetVector<Value, 4> inputs;
  for (auto arg : block->getArguments())
    inputs.insert(arg);

  // If there are no inputs, return an empty truth table
  if (inputs.empty())
    return BinaryTruthTable();

  return computeTruthTable(values, llvm::make_pointer_range(*block), inputs);
}

//===----------------------------------------------------------------------===//
// Cut
//===----------------------------------------------------------------------===//

bool Cut::isTrivialCut() const {
  // A cut is a trival cut if it has no operations and only one input
  return operations.empty() && inputs.size() == 1;
}

mlir::Operation *Cut::getRoot() const {
  return operations.empty()
             ? nullptr
             : operations.back(); // The last operation is the root
}

const NPNClass &Cut::getNPNClass() const {
  // If the NPN is already computed, return it
  if (npnClass)
    return *npnClass;

  auto truthTable = getTruthTable();

  // Compute the NPN canonical form
  auto canonicalForm = NPNClass::computeNPNCanonicalForm(truthTable);

  npnClass.emplace(std::move(canonicalForm));
  return *npnClass;
}

void Cut::getPermutatedInputs(const NPNClass &patternNPN,
                              SmallVectorImpl<Value> &permutedInputs) const {
  auto npnClass = getNPNClass();
  SmallVector<unsigned> idx;
  npnClass.getInputPermutation(patternNPN, idx);
  permutedInputs.reserve(idx.size());
  for (auto inputIndex : idx) {
    assert(inputIndex < inputs.size() && "Input index out of bounds");
    permutedInputs.push_back(inputs[inputIndex]);
  }
}

void Cut::dump(llvm::raw_ostream &os) const {
  os << "// === Cut Dump ===\n";
  os << "Cut with " << getInputSize() << " inputs and " << operations.size()
     << " operations:\n";
  if (isTrivialCut()) {
    os << "Primary input cut: " << *inputs.begin() << "\n";
    return;
  }

  os << "Inputs: \n";
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    os << "  Input " << idx << ": " << input << "\n";
  }
  os << "\nOperations: \n";
  for (auto *op : operations) {
    op->print(os);
    os << "\n";
  }
  auto &npnClass = getNPNClass();
  npnClass.dump(os);

  os << "// === Cut End ===\n";
}

unsigned Cut::getInputSize() const { return inputs.size(); }

unsigned Cut::getOutputSize() const { return getRoot()->getNumResults(); }

const BinaryTruthTable &Cut::getTruthTable() const {
  if (truthTable)
    return *truthTable;

  if (isTrivialCut()) {
    // For a trivial cut, a truth table is simply the identity function.
    // 0 -> 0, 1 -> 1
    truthTable = BinaryTruthTable(1, 1, {llvm::APInt(2, 2)});
    return *truthTable;
  }

  // Create a truth table with the given number of inputs and outputs
  truthTable = *computeTruthTable(getRoot()->getResults(), operations, inputs);

  return *truthTable;
}

static Cut getAsTrivialCut(mlir::Value value) {
  // Create a cut with a single root operation
  Cut cut;
  // There is no input for the primary input cut.
  cut.inputs.insert(value);

  return cut;
}

[[maybe_unused]] static bool isCutDerivedFromOperand(const Cut &cut,
                                                     Operation *op) {
  if (auto *root = cut.getRoot())
    return llvm::any_of(op->getOperands(),
                        [&](Value v) { return v.getDefiningOp() == root; });

  assert(cut.isTrivialCut());
  // If the cut is trivial, it has no operations, so it must be a primary input.
  // In this case, the only operation that can be derived from it is the
  // primary input itself.
  return cut.inputs.size() == 1 &&
         llvm::any_of(op->getOperands(),
                      [&](Value v) { return v == *cut.inputs.begin(); });
}

Cut Cut::mergeWith(const Cut &other, Operation *root) const {
  assert(isCutDerivedFromOperand(*this, root) &&
         isCutDerivedFromOperand(other, root) &&
         "The operation must be a child of the current root operation");

  // Create a new cut that combines this cut and the other cut
  Cut newCut;
  // Topological sort the operations in the new cut.
  // TODO: Merge-sort `operations` and `other.operations` by operation index
  // (since it's already topo-sorted, we can use a simple merge).
  std::function<void(Operation *)> populateOperations = [&](Operation *op) {
    // If the operation is already in the cut, skip it
    if (newCut.operations.contains(op))
      return;

    // Add its operands to the worklist
    for (auto value : op->getOperands()) {
      if (isAlwaysCutInput(value))
        continue;

      // If the value is in *both* cuts inputs, it is an input. So skip
      // it.
      bool isInput = inputs.contains(value);
      bool isOtherInput = other.inputs.contains(value);
      // If the value is in this cut inputs, it is an input. So skip it
      if (isInput && isOtherInput)
        continue;

      auto *defOp = value.getDefiningOp();

      assert(defOp && "Value must have a defining operation since block"
                      "arguments are treated as inputs");

      // Otherwise, check if the operation is in the other cut.
      if (isInput)
        if (!other.operations.contains(defOp)) // op is in the other cut.
          continue;
      if (isOtherInput)
        if (!operations.contains(defOp)) // op is in this cut.
          continue;
      populateOperations(defOp);
    }

    // Add the operation to the cut
    newCut.operations.insert(op);
  };

  populateOperations(root);

  // Construct inputs.
  for (auto *operation : newCut.operations) {
    for (auto value : operation->getOperands()) {
      if (isAlwaysCutInput(value)) {
        newCut.inputs.insert(value);
        continue;
      }

      auto *defOp = value.getDefiningOp();
      assert(defOp && "Value must have a defining operation");

      // If the operation is not in the cut, it is an input
      if (!newCut.operations.contains(defOp))
        // Add the input to the cut
        newCut.inputs.insert(value);
    }
  }

  // TODO: Sort the inputs by their defining operation.
  // TODO: Update area and delay based on the merged cuts.

  return newCut;
}

// Reroot the cut with a new root operation.
// This is used to create a new cut with the same inputs and operations, but a
// different root operation.
Cut Cut::reRoot(Operation *root) const {
  assert(isCutDerivedFromOperand(*this, root) &&
         "The operation must be a child of the current root operation");
  Cut newCut;
  newCut.inputs = inputs;
  newCut.operations = operations;
  // Add the new root operation to the cut
  newCut.operations.insert(root);
  return newCut;
}

//===----------------------------------------------------------------------===//
// MatchedPattern
//===----------------------------------------------------------------------===//

ArrayRef<DelayType> MatchedPattern::getArrivalTimes() const {
  assert(pattern && "Pattern must be set to get arrival time");
  return arrivalTimes;
}

DelayType MatchedPattern::getArrivalTime(unsigned index) const {
  assert(pattern && "Pattern must be set to get arrival time");
  return arrivalTimes[index];
}

const CutRewritePattern *MatchedPattern::getPattern() const {
  assert(pattern && "Pattern must be set to get the pattern");
  return pattern;
}

double MatchedPattern::getArea() const {
  assert(pattern && "Pattern must be set to get area");
  return area;
}

//===----------------------------------------------------------------------===//
// CutSet
//===----------------------------------------------------------------------===//

Cut *CutSet::getBestMatchedCut() const { return bestCut; }

unsigned CutSet::size() const { return cuts.size(); }

void CutSet::addCut(Cut cut) {
  assert(!isFrozen && "Cannot add cuts to a frozen cut set");
  cuts.push_back(std::move(cut));
}

ArrayRef<Cut> CutSet::getCuts() const { return cuts; }

// Remove duplicate cuts and non-minimal cuts. A cut is non-minimal if there
// exists another cut that is a subset of it. We use a bitset to represent the
// inputs of each cut for efficient subset checking.
static void removeDuplicateAndNonMinimalCuts(SmallVectorImpl<Cut> &cuts) {
  // First sort the cuts by input size (ascending). This ensures that when we
  // iterate through the cuts, we always encounter smaller cuts first, allowing
  // us to efficiently check for non-minimality. Stable sort to maintain
  // relative order of cuts with the same input size.
  std::stable_sort(cuts.begin(), cuts.end(), [](const Cut &a, const Cut &b) {
    return a.getInputSize() < b.getInputSize();
  });

  llvm::SmallVector<llvm::Bitset<64>, 4> inputBitMasks;
  DenseMap<Value, unsigned> inputIndices;
  auto getIndex = [&](Value v) -> unsigned {
    auto it = inputIndices.find(v);
    if (it != inputIndices.end())
      return it->second;
    unsigned index = inputIndices.size();
    if (LLVM_UNLIKELY(index >= 64))
      llvm::report_fatal_error(
          "Too many unique inputs across cuts. Max 64 supported. Consider "
          "increasing the compile-time constant.");
    inputIndices[v] = index;
    return index;
  };

  for (unsigned i = 0; i < cuts.size(); ++i) {
    auto &cut = cuts[i];
    // Create a unique identifier for the cut based on its inputs.
    llvm::Bitset<64> inputsMask;
    for (auto input : cut.inputs.getArrayRef())
      inputsMask.set(getIndex(input));

    bool isUnique = llvm::all_of(
        inputBitMasks, [&](const llvm::Bitset<64> &existingCutInputMask) {
          // If the bitset is a subset of the current inputsMask, it is not
          // unique
          return (existingCutInputMask & inputsMask) != existingCutInputMask;
        });

    if (!isUnique)
      continue;

    // If the cut is unique, keep it
    size_t uniqueCount = inputBitMasks.size();
    if (i != uniqueCount)
      cuts[uniqueCount] = std::move(cut);
    inputBitMasks.push_back(inputsMask);
  }

  unsigned uniqueCount = inputBitMasks.size();

  LLVM_DEBUG(llvm::dbgs() << "Original cuts: " << cuts.size()
                          << " Unique cuts: " << uniqueCount << "\n");

  // Resize the cuts vector to the number of unique cuts found
  cuts.resize(uniqueCount);
}

void CutSet::finalize(
    const CutRewriterOptions &options,
    llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut) {

  // Step 1: Remove duplicate and non-minimal cuts to reduce the search space
  // This eliminates cuts that are strictly dominated by others
  removeDuplicateAndNonMinimalCuts(cuts);

  // Step 2: Match each remaining cut against available patterns
  // This computes timing and area information needed for prioritization
  for (auto &cut : cuts) {
    // Verify cut doesn't exceed input size limits
    assert(cut.getInputSize() <= options.maxCutInputSize &&
           "Cut input size exceeds maximum allowed size");

    // Attempt to match the cut against available patterns
    auto matched = matchCut(cut);
    if (!matched)
      continue; // No matching pattern found for this cut

    // Store the matched pattern with the cut for later evaluation
    cut.setMatchedPattern(std::move(*matched));
  }

  // Step 3: Sort cuts by priority to select the best ones
  // Priority is determined by the optimization strategy:
  // - Trivial cuts (direct connections) have highest priority
  // - Among matched cuts, compare by area/delay based on the strategy
  // - Matched cuts are preferred over unmatched cuts
  // See "Combinational and Sequential Mapping with Priority Cuts" by Mishchenko
  // et al., ICCAD 2007 for more details.
  // TODO: Use a priority queue instead of sorting for better performance.

  // Partition the cuts into trivial and non-trivial cuts.
  auto *trivialCutsEnd =
      std::stable_partition(cuts.begin(), cuts.end(),
                            [](const Cut &cut) { return cut.isTrivialCut(); });

  std::stable_sort(trivialCutsEnd, cuts.end(),
                   [&options](const Cut &a, const Cut &b) -> bool {
                     assert(!a.isTrivialCut() && !b.isTrivialCut() &&
                            "Trivial cuts should have been excluded");
                     const auto &aMatched = a.getMatchedPattern();
                     const auto &bMatched = b.getMatchedPattern();

                     // Both cuts have matched patterns.
                     if (aMatched && bMatched)
                       return compareDelayAndArea(
                           options.strategy, aMatched->getArea(),
                           aMatched->getArrivalTimes(), bMatched->getArea(),
                           bMatched->getArrivalTimes());

                     // Prefer cuts with matched patterns over those without
                     if (aMatched && !bMatched)
                       return true;
                     if (!aMatched && bMatched)
                       return false;

                     // Both cuts are unmatched - prefer smaller input size
                     return a.getInputSize() < b.getInputSize();
                   });

  // Step 4: Limit the number of cuts to prevent exponential growth
  // After sorting, keep only the best cuts up to the specified limit
  if (cuts.size() > options.maxCutSizePerRoot)
    cuts.resize(options.maxCutSizePerRoot);

  // Select the best cut from the remaining candidates
  for (auto &cut : cuts) {
    const auto &currentMatch = cut.getMatchedPattern();
    if (!currentMatch)
      continue; // Skip cuts without matched patterns

    // This is already sorted, so the first matched cut is the best.
    bestCut = &cut;
    break;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Finalized cut set with " << cuts.size() << " cuts and "
                 << (bestCut
                         ? "matched pattern to " + bestCut->getMatchedPattern()
                                                       ->getPattern()
                                                       ->getPatternName()
                         : "no matched pattern")
                 << "\n";
  });

  isFrozen = true; // Mark the cut set as frozen
}

//===----------------------------------------------------------------------===//
// CutRewritePattern
//===----------------------------------------------------------------------===//

bool CutRewritePattern::useTruthTableMatcher(
    SmallVectorImpl<NPNClass> &matchingNPNClasses) const {
  return false;
}

//===----------------------------------------------------------------------===//
// CutRewritePatternSet
//===----------------------------------------------------------------------===//

CutRewritePatternSet::CutRewritePatternSet(
    llvm::SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns)
    : patterns(std::move(patterns)) {
  // Initialize the NPN to pattern map
  for (auto &pattern : this->patterns) {
    SmallVector<NPNClass, 2> npnClasses;
    auto result = pattern->useTruthTableMatcher(npnClasses);
    if (result) {
      for (auto npnClass : npnClasses) {
        // Create a NPN class from the truth table
        npnToPatternMap[{npnClass.truthTable.table,
                         npnClass.truthTable.numInputs}]
            .push_back(std::make_pair(std::move(npnClass), pattern.get()));
      }
    } else {
      // If the pattern does not provide NPN classes, we use a special key
      // to indicate that it should be considered for all cuts.
      nonNPNPatterns.push_back(pattern.get());
    }
  }
}

//===----------------------------------------------------------------------===//
// CutEnumerator
//===----------------------------------------------------------------------===//

CutEnumerator::CutEnumerator(const CutRewriterOptions &options)
    : options(options) {}

CutSet *CutEnumerator::createNewCutSet(Value value) {
  auto [cutSetPtr, inserted] =
      cutSets.try_emplace(value, std::make_unique<CutSet>());
  assert(inserted && "Cut set already exists for this value");
  return cutSetPtr->second.get();
}

llvm::MapVector<Value, std::unique_ptr<CutSet>> CutEnumerator::takeVector() {
  return std::move(cutSets);
}

void CutEnumerator::clear() { cutSets.clear(); }

LogicalResult CutEnumerator::visit(Operation *op) {
  if (isSupportedLogicOp(op))
    return visitLogicOp(op);

  // Skip other operations. If the operation is not a supported logic
  // operation, we create a trivial cut lazily.
  return success();
}

LogicalResult CutEnumerator::visitLogicOp(Operation *logicOp) {
  assert(logicOp->getNumResults() == 1 &&
         "Logic operation must have a single result");

  Value result = logicOp->getResult(0);
  unsigned numOperands = logicOp->getNumOperands();

  // Validate operation constraints
  // TODO: Variadic operations and non-single-bit results can be supported
  if (numOperands > 2)
    return logicOp->emitError("Cut enumeration supports at most 2 operands, "
                              "found: ")
           << numOperands;
  if (!logicOp->getOpResult(0).getType().isInteger(1))
    return logicOp->emitError()
           << "Supported logic operations must have a single bit "
              "result type but found: "
           << logicOp->getResult(0).getType();

  SmallVector<const CutSet *, 2> operandCutSets;
  operandCutSets.reserve(numOperands);
  // Collect cut sets for each operand
  for (unsigned i = 0; i < numOperands; ++i) {
    auto *operandCutSet = getCutSet(logicOp->getOperand(i));
    if (!operandCutSet)
      return logicOp->emitError("Failed to get cut set for operand ")
             << i << ": " << logicOp->getOperand(i);
    operandCutSets.push_back(operandCutSet);
  }

  // Create the singleton cut (just this operation)
  Cut primaryInputCut = getAsTrivialCut(result);

  auto *resultCutSet = createNewCutSet(result);

  // Add the singleton cut first
  resultCutSet->addCut(primaryInputCut);

  // Schedule cut set finalization when exiting this scope
  llvm::scope_exit prune([&]() {
    // Finalize cut set: remove duplicates, limit size, and match patterns
    resultCutSet->finalize(options, matchCut);
  });

  // Handle unary operations
  if (numOperands == 1) {
    const auto &inputCutSet = operandCutSets[0];

    // Try to extend each input cut by including this operation
    for (const Cut &inputCut : inputCutSet->getCuts()) {
      Cut extendedCut = inputCut.reRoot(logicOp);
      // Skip cuts that exceed input size limit
      if (extendedCut.getInputSize() > options.maxCutInputSize)
        continue;

      resultCutSet->addCut(std::move(extendedCut));
    }
    return success();
  }

  // Handle binary operations (like AND, OR, XOR gates)
  assert(numOperands == 2 && "Expected binary operation");

  const auto *lhsCutSet = operandCutSets[0];
  const auto *rhsCutSet = operandCutSets[1];

  // Combine cuts from both inputs to create larger cuts
  for (const Cut &lhsCut : lhsCutSet->getCuts()) {
    for (const Cut &rhsCut : rhsCutSet->getCuts()) {
      Cut mergedCut = lhsCut.mergeWith(rhsCut, logicOp);
      // Skip cuts that exceed input size limit
      if (mergedCut.getInputSize() > options.maxCutInputSize)
        continue;

      resultCutSet->addCut(std::move(mergedCut));
    }
  }

  return success();
}

LogicalResult CutEnumerator::enumerateCuts(
    Operation *topOp,
    llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts for module: " << topOp->getName()
                          << "\n");
  // Topologically sort the logic network
  if (failed(topologicallySortLogicNetwork(topOp)))
    return failure();

  // Store the pattern matching function for use during cut finalization
  this->matchCut = matchCut;

  // Walk through all operations in the module in a topological manner
  auto result = topOp->walk([&](Operation *op) {
    if (failed(visit(op)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Cut enumeration completed successfully\n");
  return success();
}

const CutSet *CutEnumerator::getCutSet(Value value) {
  // Check if cut set already exists
  auto *it = cutSets.find(value);
  if (it == cutSets.end()) {
    // Create new cut set for an unprocessed value
    auto cutSet = std::make_unique<CutSet>();
    cutSet->addCut(getAsTrivialCut(value));
    auto [newIt, inserted] = cutSets.insert({value, std::move(cutSet)});
    assert(inserted && "Cut set already exists for this value");
    (void)newIt;
    it = newIt;
  }

  return it->second.get();
}

/// Generate a human-readable name for a value used in test output.
/// This function creates meaningful names for values to make debug output
/// and test results more readable and understandable.
static StringRef
getTestVariableName(Value value, DenseMap<OperationName, unsigned> &opCounter) {
  if (auto *op = value.getDefiningOp()) {
    // Handle values defined by operations
    // First, check if the operation already has a name hint attribute
    if (auto name = op->getAttrOfType<StringAttr>("sv.namehint"))
      return name.getValue();

    // For single-result operations, generate a unique name based on operation
    // type
    if (op->getNumResults() == 1) {
      auto opName = op->getName();
      auto count = opCounter[opName]++;

      // Create a unique name by appending a counter to the operation name
      SmallString<16> nameStr;
      nameStr += opName.getStringRef();
      nameStr += "_";
      nameStr += std::to_string(count);

      // Store the generated name as a hint attribute for future reference
      auto nameAttr = StringAttr::get(op->getContext(), nameStr);
      op->setAttr("sv.namehint", nameAttr);
      return nameAttr;
    }

    // Multi-result operations or other cases get a generic name
    return "<unknown>";
  }

  // Handle block arguments
  auto blockArg = cast<BlockArgument>(value);
  auto hwOp =
      dyn_cast<circt::hw::HWModuleOp>(blockArg.getOwner()->getParentOp());
  if (!hwOp)
    return "<unknown>";

  // Return the formal input name from the hardware module
  return hwOp.getInputName(blockArg.getArgNumber());
}

void CutEnumerator::dump() const {
  DenseMap<OperationName, unsigned> opCounter;
  for (auto &[value, cutSetPtr] : cutSets) {
    auto &cutSet = *cutSetPtr;
    llvm::outs() << getTestVariableName(value, opCounter) << " "
                 << cutSet.getCuts().size() << " cuts:";
    for (const Cut &cut : cutSet.getCuts()) {
      llvm::outs() << " {";
      llvm::interleaveComma(cut.inputs, llvm::outs(), [&](Value input) {
        llvm::outs() << getTestVariableName(input, opCounter);
      });
      auto &pattern = cut.getMatchedPattern();
      llvm::outs() << "}"
                   << "@t" << cut.getTruthTable().table.getZExtValue() << "d";
      if (pattern) {
        llvm::outs() << *std::max_element(pattern->getArrivalTimes().begin(),
                                          pattern->getArrivalTimes().end());
      } else {
        llvm::outs() << "0";
      }
    }
    llvm::outs() << "\n";
  }
  llvm::outs() << "Cut enumeration completed successfully\n";
}

//===----------------------------------------------------------------------===//
// CutRewriter
//===----------------------------------------------------------------------===//

LogicalResult CutRewriter::run(Operation *topOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "Starting Cut Rewriter\n";
    llvm::dbgs() << "Mode: "
                 << (OptimizationStrategyArea == options.strategy ? "area"
                                                                  : "timing")
                 << "\n";
    llvm::dbgs() << "Max input size: " << options.maxCutInputSize << "\n";
    llvm::dbgs() << "Max cut size: " << options.maxCutSizePerRoot << "\n";
    llvm::dbgs() << "Max cuts per node: " << options.maxCutSizePerRoot << "\n";
  });

  // Currrently we don't support patterns with multiple outputs.
  // So check that.
  // TODO: This must be removed when we support multiple outputs.
  for (auto &pattern : patterns.patterns) {
    if (pattern->getNumOutputs() > 1) {
      return mlir::emitError(pattern->getLoc(),
                             "Cut rewriter does not support patterns with "
                             "multiple outputs yet");
    }
  }

  // First sort the operations topologically to ensure we can process them
  // in a valid order.
  if (failed(topologicallySortLogicNetwork(topOp)))
    return failure();

  // Enumerate cuts for all nodes
  if (failed(enumerateCuts(topOp)))
    return failure();

  // Dump cuts if testing priority cuts.
  if (options.testPriorityCuts) {
    cutEnumerator.dump();
    return success();
  }

  // Select best cuts and perform mapping
  if (failed(runBottomUpRewrite(topOp)))
    return failure();

  return success();
}

LogicalResult CutRewriter::enumerateCuts(Operation *topOp) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts...\n");

  return cutEnumerator.enumerateCuts(
      topOp, [&](const Cut &cut) -> std::optional<MatchedPattern> {
        // Match the cut against the patterns
        return patternMatchCut(cut);
      });
}

ArrayRef<std::pair<NPNClass, const CutRewritePattern *>>
CutRewriter::getMatchingPatternsFromTruthTable(const Cut &cut) const {
  if (patterns.npnToPatternMap.empty())
    return {};

  auto &npnClass = cut.getNPNClass();
  auto it = patterns.npnToPatternMap.find(
      {npnClass.truthTable.table, npnClass.truthTable.numInputs});
  if (it == patterns.npnToPatternMap.end())
    return {};
  return it->getSecond();
}

std::optional<MatchedPattern> CutRewriter::patternMatchCut(const Cut &cut) {
  if (cut.isTrivialCut())
    return {};

  const CutRewritePattern *bestPattern = nullptr;
  SmallVector<DelayType, 4> inputArrivalTimes;
  SmallVector<DelayType, 1> bestArrivalTimes;
  double bestArea = 0.0;
  inputArrivalTimes.reserve(cut.getInputSize());
  bestArrivalTimes.reserve(cut.getOutputSize());

  // Compute arrival times for each input.
  for (auto input : cut.inputs) {
    assert(input.getType().isInteger(1));
    if (isAlwaysCutInput(input)) {
      // If the input is a primary input, it has no delay.
      // TODO: This doesn't consider a global delay. Need to capture
      // `arrivalTime` on the IR to make the primary input delays visible.
      inputArrivalTimes.push_back(0);
      continue;
    }
    auto *cutSet = cutEnumerator.getCutSet(input);
    assert(cutSet && "Input must have a valid cut set");

    // If there is no matching pattern, it means it's not possible to use the
    // input in the cut rewriting. So abort early.
    auto *bestCut = cutSet->getBestMatchedCut();
    if (!bestCut)
      return {};

    const auto &matchedPattern = *bestCut->getMatchedPattern();

    // Otherwise, the cut input is an op result. Get the arrival time
    // from the matched pattern.
    inputArrivalTimes.push_back(matchedPattern.getArrivalTime(
        cast<mlir::OpResult>(input).getResultNumber()));
  }

  auto computeArrivalTimeAndPickBest =
      [&](const CutRewritePattern *pattern, const MatchResult &matchResult,
          llvm::function_ref<unsigned(unsigned)> mapIndex) {
        SmallVector<DelayType, 1> outputArrivalTimes;
        // Compute the maximum delay for each output from inputs.
        for (unsigned outputIndex = 0, outputSize = cut.getOutputSize();
             outputIndex < outputSize; ++outputIndex) {
          // Compute the arrival time for this output.
          DelayType outputArrivalTime = 0;
          auto delays = matchResult.getDelays();
          for (unsigned inputIndex = 0, inputSize = cut.getInputSize();
               inputIndex < inputSize; ++inputIndex) {
            // Map pattern input i to cut input through NPN transformations
            unsigned cutOriginalInput = mapIndex(inputIndex);
            outputArrivalTime =
                std::max(outputArrivalTime,
                         delays[outputIndex * inputSize + inputIndex] +
                             inputArrivalTimes[cutOriginalInput]);
          }

          outputArrivalTimes.push_back(outputArrivalTime);
        }

        // Update the arrival time
        if (!bestPattern ||
            compareDelayAndArea(options.strategy, matchResult.area,
                                outputArrivalTimes, bestArea,
                                bestArrivalTimes)) {
          LLVM_DEBUG({
            llvm::dbgs() << "== Matched Pattern ==============\n";
            llvm::dbgs() << "Matching cut: \n";
            cut.dump(llvm::dbgs());
            llvm::dbgs() << "Found better pattern: "
                         << pattern->getPatternName();
            llvm::dbgs() << " with area: " << matchResult.area;
            llvm::dbgs() << " and input arrival times: ";
            for (unsigned i = 0; i < inputArrivalTimes.size(); ++i) {
              llvm::dbgs() << " " << inputArrivalTimes[i];
            }
            llvm::dbgs() << " and arrival times: ";

            for (auto arrivalTime : outputArrivalTimes) {
              llvm::dbgs() << " " << arrivalTime;
            }
            llvm::dbgs() << "\n";
            llvm::dbgs() << "== Matched Pattern End ==============\n";
          });

          bestArrivalTimes = std::move(outputArrivalTimes);
          bestArea = matchResult.area;
          bestPattern = pattern;
        }
      };

  for (auto &[patternNPN, pattern] : getMatchingPatternsFromTruthTable(cut)) {
    assert(patternNPN.truthTable.numInputs == cut.getInputSize() &&
           "Pattern input size must match cut input size");
    auto matchResult = pattern->match(cutEnumerator, cut);
    if (!matchResult)
      continue;
    auto &cutNPN = cut.getNPNClass();

    // Get the input mapping from pattern's NPN class to cut's NPN class
    SmallVector<unsigned> inputMapping;
    cutNPN.getInputPermutation(patternNPN, inputMapping);
    computeArrivalTimeAndPickBest(pattern, *matchResult,
                                  [&](unsigned i) { return inputMapping[i]; });
  }

  for (const CutRewritePattern *pattern : patterns.nonNPNPatterns) {
    if (auto matchResult = pattern->match(cutEnumerator, cut))
      computeArrivalTimeAndPickBest(pattern, *matchResult,
                                    [&](unsigned i) { return i; });
  }

  if (!bestPattern)
    return {}; // No matching pattern found

  return MatchedPattern(bestPattern, std::move(bestArrivalTimes), bestArea);
}

LogicalResult CutRewriter::runBottomUpRewrite(Operation *top) {
  LLVM_DEBUG(llvm::dbgs() << "Performing cut-based rewriting...\n");
  const auto &cutVector = cutEnumerator.getCutSets();
  // Note: Don't clear cutEnumerator yet - we need it during rewrite
  UnusedOpPruner pruner;
  PatternRewriter rewriter(top->getContext());
  for (auto &[value, cutSet] : llvm::reverse(cutVector)) {
    if (value.use_empty()) {
      if (auto *op = value.getDefiningOp())
        pruner.eraseNow(op);
      continue;
    }

    if (isAlwaysCutInput(value)) {
      // If the value is a primary input, skip it
      LLVM_DEBUG(llvm::dbgs() << "Skipping inputs: " << value << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Cut set for value: " << value << "\n");
    auto *bestCut = cutSet->getBestMatchedCut();
    if (!bestCut) {
      if (options.allowNoMatch)
        continue; // No matching pattern found, skip this value
      return emitError(value.getLoc(), "No matching cut found for value: ")
             << value;
    }

    rewriter.setInsertionPoint(bestCut->getRoot());
    const auto &matchedPattern = bestCut->getMatchedPattern();
    auto result = matchedPattern->getPattern()->rewrite(rewriter, cutEnumerator,
                                                        *bestCut);
    if (failed(result))
      return failure();

    rewriter.replaceOp(bestCut->getRoot(), *result);

    if (options.attachDebugTiming) {
      auto array = rewriter.getI64ArrayAttr(matchedPattern->getArrivalTimes());
      (*result)->setAttr("test.arrival_times", array);
    }
  }

  // Clear the enumerator after rewriting is complete
  cutEnumerator.clear();
  return success();
}
