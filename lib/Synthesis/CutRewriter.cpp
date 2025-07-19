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
//===----------------------------------------------------------------------===//

#include "circt/Synthesis/CutRewriter.h"

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/NPNClass.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "synthesis-cut-rewriter"

using namespace circt;
using namespace circt::synthesis;

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

static bool isAlwaysCutInput(Value value) {
  auto *op = value.getDefiningOp();
  // If the value has no defining operation, it is a primary input
  if (!op)
    return true;

  if (op->hasTrait<OpTrait::ConstantLike>()) {
    // Constant values are never cut inputs.
    return false;
  }

  // TODO: Extend this to allow comb.and/xor/or as well.
  return !isa<aig::AndInverterOp>(op);
}

//===----------------------------------------------------------------------===//
// TruthTable
//===----------------------------------------------------------------------===//
// Cut
//===----------------------------------------------------------------------===//

bool Cut::isPrimaryInput() const {
  // A cut is a primary input if it has no operations and only one input
  return operations.empty() && inputs.size() == 1;
}

mlir::Operation *Cut::getRoot() const {
  return operations.empty()
             ? nullptr
             : operations.back(); // The last operation is the root
}

const mlir::FailureOr<NPNClass> &Cut::getNPNClass() const {
  // If the NPN is already computed, return it
  if (npnClass)
    return *npnClass;

  auto truthTable = getTruthTable();
  if (failed(truthTable)) {
    npnClass = failure();
    return *npnClass;
  }

  // Compute the NPN canonical form
  auto canonicalForm = NPNClass::computeNPNCanonicalForm(*truthTable);

  npnClass.emplace(std::move(canonicalForm));
  return *npnClass;
}

void Cut::dump(llvm::raw_ostream &os) const {
  os << "// === Cut Dump ===\n";

  os << "Cut with " << getInputSize() << " inputs and " << getCutSize()
     << " operations:\n";
  if (isPrimaryInput()) {
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
  npnClass->dump(os);

  os << "// === Cut End ===\n";
}

unsigned Cut::getInputSize() const { return inputs.size(); }

unsigned Cut::getCutSize() const { return operations.size(); }

unsigned Cut::getOutputSize() const { return getRoot()->getNumResults(); }

const llvm::FailureOr<BinaryTruthTable> &Cut::getTruthTable() const {
  assert(!isPrimaryInput() && "Primary input cuts do not have truth tables");

  if (truthTable)
    return *truthTable;

  int64_t numInputs = getInputSize();
  int64_t numOutputs = getOutputSize();
  assert(numInputs < 20 && "Truth table is too large");

  // Simulate the IR.
  uint32_t tableSize = 1 << numInputs;
  DenseMap<Value, APInt> eval;
  for (uint32_t i = 0; i < numInputs; ++i) {
    APInt value(tableSize, 0);
    for (uint32_t j = 0; j < tableSize; ++j) {
      // Make sure the order of the bits is correct.
      value.setBitVal(j, (j >> i) & 1);
    }
    // Set the input value for the truth table
    eval[inputs[i]] = std::move(value);
  }

  // Simulate the operations in the cut
  for (auto *op : operations) {
    auto result = simulateOp(op, eval);
    if (failed(result)) {
      mlir::emitError(op->getLoc(), "Failed to simulate operation") << *op;
      truthTable = failure();
      return *truthTable;
    }
    // Set the output value for the truth table
    for (unsigned j = 0; j < op->getNumResults(); ++j) {
      auto outputValue = op->getResult(j);
      if (!outputValue.getType().isInteger(1)) {
        mlir::emitError(op->getLoc(), "Output value is not a single bit: ")
            << *op;
        truthTable = failure();
        return *truthTable;
      }
    }
  }

  // Extract the truth table from the root operation
  auto rootResults = getRoot()->getResults();
  assert(rootResults.size() == 1 &&
         "For now we only support single output cuts");
  auto result = rootResults[0];

  // Cache the truth table
  truthTable = BinaryTruthTable(numInputs, numOutputs, eval[result]);
  return *truthTable;
}

static Cut getAsPrimaryInput(mlir::Value value) {
  // Create a cut with a single root operation
  Cut cut;
  // There is no input for the primary input cut.
  cut.inputs.insert(value);

  return cut;
}

static Cut getSingletonCut(mlir::Operation *op) {
  // Create a cut with a single input value
  Cut cut;
  cut.operations.insert(op);
  for (auto value : op->getOperands()) {
    // Only consider integer values. No integer type such as seq.clock,
    // aggregate are pass through the cut.
    assert(value.getType().isInteger(1));

    cut.inputs.insert(value);
  }
  return cut;
}

Cut Cut::mergeWith(const Cut &other, Operation *root) const {
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
      if (isAlwaysCutInput(value)) {
        // If the value is a primary input, add it to the cut inputs
        continue;
      }

      // If the value is in *both* cuts inputs, it is an input. So skip
      // it.
      bool isInput = inputs.contains(value);
      bool isOtherInput = other.inputs.contains(value);
      // If the value is in this cut inputs, it is an input. So skip it
      if (isInput && isOtherInput) {
        continue;
      }
      auto *defOp = value.getDefiningOp();

      assert(defOp);

      if (isInput) {
        if (!other.operations.contains(defOp)) // op is in the other cut.
          continue;
      }

      if (isOtherInput) {
        if (!operations.contains(defOp)) // op is in this cut.
          continue;
      }

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

  // Update area and delay based on the merged cuts
  return newCut;
}

LogicalResult Cut::simulateOp(Operation *op,
                              DenseMap<Value, APInt> &values) const {
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    auto inputs = andOp.getInputs();
    SmallVector<APInt, 2> operands;
    for (auto input : inputs) {
      auto it = values.find(input);
      if (it == values.end()) {
        op->emitError("Input value not found: ") << input;
        return failure();
      }
      operands.push_back(it->second);
    }

    // Simulate the AND operation
    values[andOp.getResult()] = andOp.evaluate(operands);
    return llvm::success();
  }
  // Add more operation types as needed
  return failure();
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

Cut *MatchedPattern::getCut() const {
  assert(cut && "Cut must be set to get the cut");
  return cut;
}

double MatchedPattern::getArea() const {
  assert(pattern && "Pattern must be set to get area");
  return pattern->getArea(*cut);
}

DelayType MatchedPattern::getDelay(unsigned inputIndex,
                                   unsigned outputIndex) const {
  assert(pattern && "Pattern must be set to get delay");
  return pattern->getDelay(inputIndex, outputIndex);
}

//===----------------------------------------------------------------------===//
// CutSet
//===----------------------------------------------------------------------===//

bool CutSet::isMatched() const {
  return matchedPattern.has_value() && matchedPattern->getPattern();
}

std::optional<MatchedPattern> CutSet::getMatchedPattern() const {
  return matchedPattern;
}

Cut *CutSet::getMatchedCut() {
  assert(isMatched() &&
         "Matched pattern must be set before getting matched cut");
  return matchedPattern->getCut();
}

unsigned CutSet::size() const { return cuts.size(); }

void CutSet::addCut(Cut cut) {
  assert(!isFrozen && "Cannot add cuts to a frozen cut set");
  cuts.push_back(std::move(cut));
}

ArrayRef<Cut> CutSet::getCuts() const { return cuts; }

void CutSet::finalize(
    const CutRewriterOptions &options,
    llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut) {
  DenseSet<std::pair<ArrayRef<Value>, Operation *>> uniqueCuts;
  unsigned uniqueCount = 0;
  for (unsigned i = 0; i < cuts.size(); ++i) {
    auto &cut = cuts[i];
    // Create a unique identifier for the cut based on its inputs.
    auto inputs = cut.inputs.getArrayRef();

    // If the cut is a duplicate, skip it.
    if (uniqueCuts.contains({inputs, cut.getRoot()}))
      continue;

    if (i != uniqueCount) {
      // Move the unique cut to the front of the vector
      // This maintains the order of cuts while removing duplicates
      // by swapping with the last unique cut found.
      cuts[uniqueCount] = std::move(cuts[i]);
    }

    // Beaware of lifetime of ArrayRef. `cuts[uniqueCount]` is always valid
    // after this point.
    uniqueCuts.insert(
        {cuts[uniqueCount].inputs.getArrayRef(), cuts[uniqueCount].getRoot()});
    ++uniqueCount;
  }

  LLVM_DEBUG(llvm::dbgs() << "Original cuts: " << cuts.size()
                          << " Unique cuts: " << uniqueCount << "\n");
  // Resize the cuts vector to the number of unique cuts found
  cuts.resize(uniqueCount);

  // Maintain size limit by removing worst cuts
  if (cuts.size() > options.maxCutSizePerRoot) {
    // Sort by priority using heuristic.
    // TODO: Make this configurable.
    std::sort(cuts.begin(), cuts.end(), [](const Cut &a, const Cut &b) {
      return a.getCutSize() < b.getCutSize();
    });

    // TODO: Implement pruning based on dominance.

    // TODO: Pririty cuts may prune all matching cuts, so we may need to
    //       keep the matching cut before pruning.
    cuts.resize(options.maxCutSizePerRoot);
  }

  // Find the best matching pattern for this cut set
  for (auto &cut : cuts) {
    // Match the cut against the pattern set
    auto matchResult = matchCut(cut);
    if (!matchResult)
      continue;

    if (!matchedPattern ||
        compareDelayAndArea(options.strategy, matchResult->getArea(),
                            matchResult->getArrivalTimes(),
                            matchedPattern->getArea(),
                            matchedPattern->getArrivalTimes())) {
      // Found a better matching pattern
      matchedPattern = matchResult;
    }
  }

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
    if (pattern->useTruthTableMatcher(npnClasses)) {
      for (auto npnClass : npnClasses) {
        // Create a NPN class from the truth table
        npnToPatternMap[npnClass.truthTable.table].push_back(
            std::make_pair(std::move(npnClass), pattern.get()));
      }
    } else {
      // If the pattern does not use truth table matcher, add it to the
      // non-truth table patterns
      nonTruthTablePatterns.push_back(pattern.get());
    }
  }
}

//===----------------------------------------------------------------------===//
// CutEnumerator
//===----------------------------------------------------------------------===//

CutEnumerator::CutEnumerator(const CutRewriterOptions &options)
    : options(options) {}

CutSet *CutEnumerator::lookup(Value value) const {
  const auto *it = cutSets.find(value);
  if (it != cutSets.end())
    return it->second.get();
  return nullptr;
}

CutSet *CutEnumerator::createNewCutSet(Value value) {
  assert(!cutSets.contains(value) && "Cut set already exists for this value");
  auto cutSet = std::make_unique<CutSet>();
  auto *cutSetPtr = cutSet.get();
  cutSets[value] = std::move(cutSet);
  return cutSetPtr;
}

llvm::MapVector<Value, std::unique_ptr<CutSet>> CutEnumerator::takeVector() {
  return std::move(cutSets);
}

void CutEnumerator::clear() { cutSets.clear(); }

LogicalResult CutEnumerator::visit(Operation *op) {
  // For now, delegate to visitLogicOp for combinational operations
  if (isa<aig::AndInverterOp>(op))
    return visitLogicOp(op);

  // Skip non-combinational operations
  return success();
}

LogicalResult CutEnumerator::visitLogicOp(Operation *logicOp) {
  assert(logicOp->getNumResults() == 1 &&
         "Logic operation must have a single result");

  Value result = logicOp->getResult(0);
  unsigned numOperands = logicOp->getNumOperands();

  // Validate operation constraints
  if (numOperands > 2) {
    return logicOp->emitError("Cut enumeration supports at most 2 operands, "
                              "found: ")
           << numOperands;
  }

  if (!logicOp->getOpResult(0).getType().isInteger(1)) {
    return logicOp->emitError("Result type must be a single bit integer");
  }

  // Create the singleton cut (just this operation)
  Cut singletonCut = getSingletonCut(logicOp);
  auto *resultCutSet = createNewCutSet(result);

  // Add the singleton cut first
  resultCutSet->addCut(singletonCut);

  // Schedule cut set finalization when exiting this scope
  auto prune = llvm::make_scope_exit([&]() {
    // Finalize cut set: remove duplicates, limit size, and match patterns
    resultCutSet->finalize(options, matchCut);
  });

  // Handle unary operations (like NOT gates)
  if (numOperands == 1) {
    const auto &inputCutSet = getCutSet(logicOp->getOperand(0));

    // Try to extend each input cut by including this operation
    for (const Cut &inputCut : inputCutSet.getCuts()) {
      Cut extendedCut = inputCut.mergeWith(singletonCut, logicOp);

      // Skip cuts that exceed input size limit
      if (extendedCut.getInputSize() > options.maxCutInputSize)
        continue;

      resultCutSet->addCut(std::move(extendedCut));
    }
    return success();
  }

  // Handle binary operations (like AND, OR, XOR gates)
  assert(numOperands == 2 && "Expected binary operation");

  const auto &lhsCutSet = getCutSet(logicOp->getOperand(0));
  const auto &rhsCutSet = getCutSet(logicOp->getOperand(1));

  // Combine cuts from both inputs to create larger cuts
  for (const Cut &lhsCut : lhsCutSet.getCuts()) {
    for (const Cut &rhsCut : rhsCutSet.getCuts()) {
      Cut mergedCut = lhsCut.mergeWith(rhsCut, logicOp);

      // Skip cuts that exceed size limits
      if (mergedCut.getCutSize() > options.maxCutSizePerRoot)
        continue;
      if (mergedCut.getInputSize() > options.maxCutInputSize)
        continue;

      resultCutSet->addCut(std::move(mergedCut));
    }
  }

  return success();
}

LogicalResult CutEnumerator::enumerateCuts(
    Operation *topOp,
    llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts for module: " << topOp->getName()
                          << "\n");

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

const CutSet &CutEnumerator::getCutSet(Value value) {
  // Check if cut set already exists
  if (!cutSets.contains(value)) {
    // Create new cut set for primary input or unprocessed value
    cutSets[value] = std::make_unique<CutSet>();

    // Primary inputs get a trivial cut containing just themselves
    cutSets[value]->addCut(getAsPrimaryInput(value));

    LLVM_DEBUG(llvm::dbgs()
               << "Created primary input cut for: " << value << "\n");
  }

  return *cutSets.find(value)->second;
}

//===----------------------------------------------------------------------===//
// CutRewriter
//===----------------------------------------------------------------------===//

LogicalResult CutRewriter::sortOperationsTopologically(Operation *topOp) {
  // Sort the operations topologically
  if (topOp
          ->walk([&](Region *region) {
            auto regionKindOp =
                dyn_cast<mlir::RegionKindInterface>(region->getParentOp());
            if (!regionKindOp ||
                regionKindOp.hasSSADominance(region->getRegionNumber()))
              return WalkResult::advance();

            // Graph region.
            for (auto &block : *region) {
              if (!mlir::sortTopologically(
                      &block, [&](Value value, Operation *op) -> bool {
                        // Topologically sort AND-inverters and purely dataflow
                        // ops. Other operations can be scheduled.
                        return !(isa<aig::AndInverterOp, comb::ExtractOp,
                                     comb::ReplicateOp, comb::ConcatOp>(op));
                      }))
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return mlir::emitError(topOp->getLoc(),
                           "failed to sort operations topologically");
  return success();
}

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
  if (failed(sortOperationsTopologically(topOp)))
    return failure();

  // Enumerate cuts for all nodes
  if (failed(enumerateCuts(topOp)))
    return failure();

  // Select best cuts and perform mapping
  if (failed(runBottomUpRewrite(topOp)))
    return failure();

  return success();
}

LogicalResult CutRewriter::enumerateCuts(Operation *topOp) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts...\n");

  return cutEnumerator.enumerateCuts(
      topOp, [&](Cut &cut) -> std::optional<MatchedPattern> {
        // Match the cut against the patterns
        return patternMatchCut(cut);
      });
}

ArrayRef<std::pair<NPNClass, CutRewritePattern *>>
CutRewriter::getMatchingPatternFromTruthTable(const Cut &cut) const {
  if (patterns.npnToPatternMap.empty())
    return {};

  auto &npnClass = cut.getNPNClass();
  auto it = patterns.npnToPatternMap.find(npnClass->truthTable.table);
  if (it == patterns.npnToPatternMap.end())
    return {};
  return it->getSecond();
}

std::optional<MatchedPattern> CutRewriter::patternMatchCut(Cut &cut) {
  if (cut.isPrimaryInput())
    return {};

  CutRewritePattern *bestPattern = nullptr;
  SmallVector<DelayType, 4> inputArrivalTimes;
  SmallVector<DelayType, 2> bestArrivalTimes;
  inputArrivalTimes.reserve(cut.getInputSize());
  bestArrivalTimes.reserve(cut.getOutputSize());

  // Compute arrival times for each input.
  for (auto input : cut.inputs) {
    assert(input.getType().isInteger(1));
    if (isAlwaysCutInput(input)) {
      // If the input is a primary input, it has no delay.
      // TODO: This doesn't consider a global delay. Need to capture
      // `arrivalTime` on the IR to make the primary input delays visible.
      inputArrivalTimes.push_back(0.0);
      continue;
    }
    auto *cutSet = cutEnumerator.lookup(input);
    assert(cutSet && "Input must have a valid cut set");

    auto matchedPattern = cutSet->getMatchedPattern();
    // If there is no matching pattern, it means it's not possilbe to use the
    // input in the cut rewriting. So abort early.
    if (!matchedPattern)
      return {};

    // This must be a block argument must have been a cut input.
    auto resultNumber = cast<mlir::OpResult>(input);
    inputArrivalTimes.push_back(
        matchedPattern->getArrivalTime(resultNumber.getResultNumber()));
  }

  auto computeArrivalTimeAndPickBest =
      [&](CutRewritePattern *pattern,
          llvm::function_ref<unsigned(unsigned)> mapIndex) {
        SmallVector<DelayType, 2> outputArrivalTimes;
        // Compute the maximum delay for each output from inputs.
        for (unsigned outputIndex = 0; outputIndex < cut.getOutputSize();
             ++outputIndex) {
          // Compute the arrival time for this outpu.
          DelayType outputArrivalTime = 0;
          for (unsigned inputIndex = 0; inputIndex < cut.getInputSize();
               ++inputIndex) {
            // Map pattern input i to cut input through NPN transformations
            unsigned cutOriginalInput = mapIndex(inputIndex);
            outputArrivalTime =
                std::max(outputArrivalTime,
                         pattern->getDelay(cutOriginalInput, outputIndex) +
                             inputArrivalTimes[cutOriginalInput]);
          }

          outputArrivalTimes.push_back(outputArrivalTime);
        }

        // Update the arrival time
        if (!bestPattern ||
            compareDelayAndArea(options.strategy, pattern->getArea(cut),
                                outputArrivalTimes, bestPattern->getArea(cut),
                                bestArrivalTimes)) {
          LLVM_DEBUG({
            llvm::dbgs() << "== Matched Pattern ==============\n";
            llvm::dbgs() << "Matching cut: \n";
            cut.dump(llvm::dbgs());
            llvm::dbgs() << "Found better pattern: "
                         << pattern->getPatternName();
            llvm::dbgs() << " with area: " << pattern->getArea(cut);
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
          bestPattern = pattern;
        }
      };

  for (auto &[patternNPN, pattern] : getMatchingPatternFromTruthTable(cut)) {
    if (!pattern->match(cut))
      continue;
    auto &cutNPN = cut.getNPNClass();

    // Get the input mapping from pattern's NPN class to cut's NPN class
    auto inputMapping = cutNPN->getInputMappingTo(patternNPN);
    computeArrivalTimeAndPickBest(pattern,
                                  [&](unsigned i) { return inputMapping[i]; });
  }

  for (CutRewritePattern *pattern : patterns.nonTruthTablePatterns)
    if (pattern->match(cut))
      computeArrivalTimeAndPickBest(pattern, [&](unsigned i) { return i; });

  if (!bestPattern)
    return std::nullopt; // No matching pattern found

  return MatchedPattern(bestPattern, &cut, std::move(bestArrivalTimes));
}

LogicalResult CutRewriter::runBottomUpRewrite(Operation *top) {
  LLVM_DEBUG(llvm::dbgs() << "Performing cut-based rewriting...\n");
  auto cutVector = cutEnumerator.takeVector();
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
      LLVM_DEBUG(llvm::dbgs() << "Skipping primary input: " << value << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Cut set for value: " << value << "\n");
    auto matchedPattern = cutSet->getMatchedPattern();
    if (!matchedPattern) {
      if (options.allowNoMatch)
        continue; // No matching pattern found, skip this value
      return emitError(value.getLoc(), "No matching cut found for value: ")
             << value;
    }

    auto *cut = matchedPattern->getCut();
    rewriter.setInsertionPoint(cut->getRoot());
    auto result = matchedPattern->getPattern()->rewrite(rewriter, *cut);
    if (failed(result))
      return failure();

    rewriter.replaceOp(cut->getRoot(), *result);

    if (options.attachDebugTiming) {
      auto array = rewriter.getI64ArrayAttr(matchedPattern->getArrivalTimes());
      (*result)->setAttr("test.arrival_times", array);
    }
  }

  return success();
}
