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

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/NPNClass.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
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
#include <memory>
#include <optional>

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

  assert(false &&
         "Unsupported operation for simulation. isSupportedLogicOp should "
         "be used to check if the operation can be simulated.");
}

//  Return true if the value is always a cut input.
static bool isAlwaysCutInput(Value value) {
  auto *op = value.getDefiningOp();
  // If the value has no defining operation, it is a primary input
  if (!op)
    return true;

  if (op->hasTrait<OpTrait::ConstantLike>()) {
    // Constant values are never cut inputs.
    return false;
  }

  return !isSupportedLogicOp(op);
}

LogicalResult
circt::synth::topologicallySortLogicNetwork(mlir::Operation *topOp) {

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
                        // Topologically sort simulatable ops and purely
                        // dataflow ops. Other operations can be scheduled.
                        return !(isSupportedLogicOp(op) ||
                                 isa<comb::ExtractOp, comb::ReplicateOp,
                                     comb::ConcatOp>(op));
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

//===----------------------------------------------------------------------===//
// Cut
//===----------------------------------------------------------------------===//

bool Cut::isTrivialCut() const {
  // A cut is a primary input if it has no operations and only one input
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
                              SmallVectorImpl<Value> &permuatedInputs) const {
  auto npnClass = getNPNClass();
  SmallVector<Value> permutedInputs;
  SmallVector<unsigned> idx;
  npnClass.getInputPermutation(patternNPN, idx);
  permuatedInputs.reserve(idx.size());
  for (auto inputIndex : idx) {
    assert(inputIndex < inputs.size() && "Input index out of bounds");
    permuatedInputs.push_back(inputs[inputIndex]);
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
  for (auto *op : operations)
    simulateLogicOp(op, eval);

  // Extract the truth table from the root operation
  auto rootResults = getRoot()->getResults();
  BinaryTruthTable result(numInputs, numOutputs);
  assert(numOutputs == 1 &&
         "Multiple outputs are not supported yet, must be rejected earlier");
  result.table = eval.at(rootResults[0]);
  truthTable = std::move(result);
  return *truthTable;
}

static Cut getAsTrivialCut(mlir::Value value) {
  // Create a cut with a single root operation
  Cut cut;
  // There is no input for the primary input cut.
  cut.inputs.insert(value);

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

  // A map for constructing the depth of each operation.
  DenseMap<Operation *, unsigned> depthMap;

  // Construct inputs.
  for (auto *operation : newCut.operations) {
    depthMap[operation] = 1;
    for (auto value : operation->getOperands()) {
      if (isAlwaysCutInput(value)) {
        newCut.inputs.insert(value);
        continue;
      }

      auto *defOp = value.getDefiningOp();
      assert(defOp && "Value must have a defining operation");

      depthMap[operation] = std::max(depthMap[operation], depthMap[defOp] + 1);
      // If the operation is not in the cut, it is an input
      if (!newCut.operations.contains(defOp))
        // Add the input to the cut
        newCut.inputs.insert(value);
    }
    unsigned depth = depthMap[operation];
    newCut.depth = std::max(depth, newCut.depth);
  }

  // TODO: Sort the inputs by their defining operation.
  // TODO: Update area and delay based on the merged cuts.

  return newCut;
}

// Reroot the cut with a new root operation.
// This is used to create a new cut with the same inputs and operations, but a
// different root operation.
Cut Cut::reRoot(Operation *root) const {
  Cut newCut;
  newCut.inputs = inputs;
  newCut.operations = operations;
  // Add the new root operation to the cut
  newCut.operations.insert(root);
  newCut.depth = depth + 1; // Increment depth for new root
  return newCut;
}

//===----------------------------------------------------------------------===//
// CutSet
//===----------------------------------------------------------------------===//

unsigned CutSet::size() const { return cuts.size(); }

void CutSet::addCut(Cut cut) {
  assert(!isFrozen && "Cannot add cuts to a frozen cut set");
  cuts.push_back(std::move(cut));
}

ArrayRef<Cut> CutSet::getCuts() const { return cuts; }

void CutSet::finalize(const CutRewriterOptions &options) {
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
    // Sort by priority using heuristic. Currently we sort by (depth, input
    // size).
    // TODO: Make this configurable.
    // TODO: Implement pruning based on dominance.

    std::sort(cuts.begin(), cuts.end(), [](const Cut &a, const Cut &b) {
      if (a.getDepth() == b.getDepth())
        return a.getInputSize() < b.getInputSize();
      return a.getDepth() < b.getDepth();
    });

    cuts.resize(options.maxCutSizePerRoot);
  }

  isFrozen = true; // Mark the cut set as frozen
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
  if (isSupportedLogicOp(op)) {
    // Support only binary operations with a single bit
    // TODO: This can be extended to support variadic multiplication.
    if (op->getNumOperands() > 2)
      return op->emitError() << "Cut enumeration supports at most 2 operands, "
                             << "found: " << op->getNumOperands();
    if (!op->getResult(0).getType().isInteger(1))
      return op->emitError()
             << "Supported logic operations must have a single bit "
                "result type but found: "
             << op->getResult(0).getType();

    return visitLogicOp(op);
  }

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
    return logicOp->emitError("Result type must be a single bit integer");

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
  auto prune = llvm::make_scope_exit([&]() {
    // Finalize cut set: remove duplicates, limit size, and match patterns
    resultCutSet->finalize(options);
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
