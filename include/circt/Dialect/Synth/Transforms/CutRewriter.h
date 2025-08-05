//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a general cut-based rewriting framework for
// combinational logic optimization. The framework uses NPN-equivalence matching
// with area and delay metrics to rewrite cuts (subgraphs) in combinational
// circuits with optimal patterns.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_CUT_REWRITER_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_CUT_REWRITER_H

#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/NPNClass.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>

namespace circt {
namespace synth {

// This is a helper function to sort operations topologically in a logic
// network. This is necessary for cut rewriting to ensure that operations are
// processed in the correct order, respecting dependencies.
LogicalResult topologicallySortLogicNetwork(mlir::Operation *op);

//===----------------------------------------------------------------------===//
// Cut Data Structures
//===----------------------------------------------------------------------===//

// Forward declarations
class CutRewriter;
struct CutRewriterOptions;

/// Represents a cut in the combinational logic network.
///
/// A cut is a subset of nodes in the combinational logic that forms a complete
/// subgraph with a single output. It represents a portion of the circuit that
/// can potentially be replaced with a single library gate or pattern.
///
/// The cut contains:
/// - Input values: The boundary between the cut and the rest of the circuit
/// - Operations: The logic operations within the cut boundary
/// - Root operation: The output-driving operation of the cut
///
/// Cuts are used in combinational logic optimization to identify regions that
/// can be optimized and replaced with more efficient implementations.
class Cut {
  /// Cached truth table for this cut.
  /// Computed lazily when first accessed to avoid unnecessary computation.
  mutable std::optional<BinaryTruthTable> truthTable;

  /// Cached NPN canonical form for this cut.
  /// Computed lazily from the truth table when first accessed.
  mutable std::optional<NPNClass> npnClass;

  unsigned depth = 0; ///< Depth of this cut in the logic network

public:
  /// External inputs to this cut (cut boundary).
  /// These are the values that flow into the cut from outside.
  llvm::SmallSetVector<mlir::Value, 4> inputs;

  /// Operations contained within this cut.
  /// Stored in topological order with the root operation at the end.
  llvm::SmallSetVector<mlir::Operation *, 4> operations;

  /// Check if this cut represents a trivial cut.
  /// A trivial cut has no internal operations and exactly one input.
  bool isTrivialCut() const;

  /// Get the root operation of this cut.
  unsigned getDepth() const;

  /// Get the root operation of this cut.
  /// The root operation produces the output of the cut.
  mlir::Operation *getRoot() const;

  void dump(llvm::raw_ostream &os) const;

  /// Merge this cut with another cut to form a new cut.
  /// The new cut combines the operations from both cuts with the given root.
  Cut mergeWith(const Cut &other, Operation *root) const;
  Cut reRoot(Operation *root) const;

  /// Get the number of inputs to this cut.
  unsigned getInputSize() const;

  /// Get the number of operations in this cut.
  unsigned getCutSize() const;

  /// Get the number of outputs from root operation.
  unsigned getOutputSize() const;

  /// Get the truth table for this cut.
  /// The truth table represents the boolean function computed by this cut.
  const BinaryTruthTable &getTruthTable() const;

  /// Get the NPN canonical form for this cut.
  /// This is used for efficient pattern matching against library components.
  const NPNClass &getNPNClass() const;

  /// Get the permutated inputs for this cut based on the given pattern NPN.
  void getPermutatedInputs(const NPNClass &patternNPN,
                           SmallVectorImpl<Value> &permutedInputs) const;
};

/// Represents a cut that has been successfully matched to a rewriting pattern.
///
/// This class encapsulates the result of matching a cut against a rewriting
/// pattern during optimization. It stores the matched pattern, the
/// cut that was matched, and timing information needed for optimization.
/// TODO: Currently only added as a placeholder.
class MatchedPattern {
public:
};

/// Manages a collection of cuts for a single logic node using priority cuts
/// algorithm.
///
/// Each node in the combinational logic network can have multiple cuts
/// representing different ways to group it with surrounding logic. The CutSet
/// manages these cuts and selects the best one based on the optimization
/// strategy (area or timing).
///
/// The priority cuts algorithm maintains a bounded set of the most promising
/// cuts to avoid exponential explosion while ensuring good optimization
/// results.
class CutSet {
private:
  llvm::SmallVector<Cut, 4> cuts; ///< Collection of cuts for this node
  std::optional<MatchedPattern> matchedPattern; ///< Best matched pattern found
  bool isFrozen = false; ///< Whether cut set is finalized

public:
  /// Finalize the cut set by removing duplicates and selecting the best
  /// pattern.
  ///
  /// This method:
  /// 1. Removes duplicate cuts based on inputs and root operation
  /// 2. Limits the number of cuts to prevent exponential growth
  /// 3. Matches each cut against available patterns
  /// 4. Selects the best pattern based on the optimization strategy
  void finalize(const CutRewriterOptions &options);

  /// Get the number of cuts in this set.
  unsigned size() const;

  /// Add a new cut to this set.
  /// NOTE: The cut set must not be frozen
  void addCut(Cut cut);

  /// Get read-only access to all cuts in this set.
  ArrayRef<Cut> getCuts() const;
};

/// Configuration options for the cut-based rewriting algorithm.
///
/// These options control various aspects of the rewriting process including
/// optimization strategy, resource limits, and algorithmic parameters.
struct CutRewriterOptions {
  /// Optimization strategy (area vs. timing).
  OptimizationStrategy strategy;

  /// Maximum number of inputs allowed for any cut.
  /// Larger cuts provide more optimization opportunities but increase
  /// computational complexity exponentially.
  unsigned maxCutInputSize;

  /// Maximum number of cuts to maintain per logic node.
  /// The priority cuts algorithm keeps only the most promising cuts
  /// to prevent exponential explosion.
  unsigned maxCutSizePerRoot;
};

//===----------------------------------------------------------------------===//
// Cut Enumeration Engine
//===----------------------------------------------------------------------===//

/// Cut enumeration engine for combinational logic networks.
///
/// The CutEnumerator is responsible for generating cuts for each node in a
/// combinational logic network. It uses a priority cuts algorithm to maintain a
/// bounded set of promising cuts while avoiding exponential explosion.
///
/// The enumeration process works by:
/// 1. Visiting nodes in topological order
/// 2. For each node, combining cuts from its inputs
/// 3. Matching generated cuts against available patterns
/// 4. Maintaining only the most promising cuts per node
class CutEnumerator {
public:
  /// Constructor for cut enumerator.
  explicit CutEnumerator(const CutRewriterOptions &options);

  /// Enumerate cuts for all nodes in the given module.
  ///
  /// This is the main entry point that orchestrates the cut enumeration
  /// process. It visits all operations in the module and generates cuts
  /// for combinational logic operations.
  LogicalResult enumerateCuts(
      Operation *topOp,
      llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut =
          [](Cut &) {
            return std::nullopt; // Default no-op matcher
          });

  /// Create a new cut set for a value.
  /// The value must not already have a cut set.
  CutSet *createNewCutSet(Value value);

  /// Get the cut set for a specific value.
  /// If not found, it means no cuts have been generated for this value yet.
  /// In that case return a trivial cut set.
  const CutSet *getCutSet(Value value);

  /// Move ownership of all cut sets to caller.
  /// After calling this, the enumerator is left in an empty state.
  llvm::MapVector<Value, std::unique_ptr<CutSet>> takeVector();

  /// Clear all cut sets and reset the enumerator.
  void clear();

private:
  /// Visit a single operation and generate cuts for it.
  LogicalResult visit(Operation *op);

  /// Visit a combinational logic operation and generate cuts.
  /// This handles the core cut enumeration logic for operations
  /// like AND, OR, XOR, etc.
  LogicalResult visitLogicOp(Operation *logicOp);

  /// Maps values to their associated cut sets.
  llvm::MapVector<Value, std::unique_ptr<CutSet>> cutSets;

  /// Configuration options for cut enumeration.
  const CutRewriterOptions &options;

  /// Function to match cuts against available patterns.
  /// Set during enumeration and used when finalizing cut sets.
  llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut;
};

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_CUT_REWRITER_H
