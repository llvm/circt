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
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>

namespace circt {
namespace synth {
// Type for representing delays in the circuit. It's user's responsibility to
// use consistent units, i.e., all delays should be in the same unit (usually
// femtoseconds, but not limited to it).
using DelayType = int64_t;

/// Maximum number of inputs supported for truth table generation.
/// This limit prevents excessive memory usage as truth table size grows
/// exponentially with the number of inputs (2^n entries).
static constexpr unsigned maxTruthTableInputs = 16;

// This is a helper function to sort operations topologically in a logic
// network. This is necessary for cut rewriting to ensure that operations are
// processed in the correct order, respecting dependencies.
LogicalResult topologicallySortLogicNetwork(mlir::Operation *op);

// Get the truth table for a specific operation within a block.
// Block must be a SSACFG or topologically sorted.
FailureOr<BinaryTruthTable> getTruthTable(ValueRange values, Block *block);

//===----------------------------------------------------------------------===//
// Cut Data Structures
//===----------------------------------------------------------------------===//

// Forward declarations
class CutRewritePatternSet;
class CutRewriter;
class CutEnumerator;
struct CutRewritePattern;
struct CutRewriterOptions;
class LogicNetwork;

//===----------------------------------------------------------------------===//
// Logic Network Data Structures (Flat IR for efficient cut enumeration)
//===----------------------------------------------------------------------===//

/// Edge representation in the logic network.
/// Similar to mockturtle's signal, this encodes both a node index and inversion
/// in a single 32-bit value. The LSB indicates whether the signal is inverted.
struct Signal {
  uint32_t data = 0;

  Signal() = default;
  Signal(uint32_t index, bool inverted)
      : data((index << 1) | (inverted ? 1 : 0)) {}
  explicit Signal(uint32_t raw) : data(raw) {}

  /// Get the node index (without the inversion bit).
  uint32_t getIndex() const { return data >> 1; }

  /// Check if this edge is inverted.
  bool isInverted() const { return data & 1; }

  /// Get the raw data (index << 1 | inverted).
  uint32_t getRaw() const { return data; }

  Signal flipInversion() const { return Signal(getIndex(), !isInverted()); }

  /// Create an inverted version of this edge.
  Signal operator!() const { return Signal(data ^ 1); }

  bool operator==(const Signal &other) const { return data == other.data; }
  bool operator!=(const Signal &other) const { return data != other.data; }
  bool operator<(const Signal &other) const { return data < other.data; }
};

/// Represents a single gate/node in the flat logic network.
/// This structure is designed to be cache-friendly and supports up to 3 inputs
/// (sufficient for AND, XOR, MAJ gates). For nodes with fewer inputs, unused
/// edges have index 0 (constant 0 node).
///
/// Special indices:
///   - Index 0: Constant 0
///   - Index 1: Constant 1
///
/// It uses 8 bytes for operation pointer + enum, 12 bytes for edges = 20
/// bytes per gate.
struct LogicNetworkGate {
  /// Kind of logic gate.
  enum Kind : uint8_t {
    Constant = 0, ///< Constant 0/1 node (index 0 = const0, index 1 = const1)
    PrimaryInput = 1, ///< Primary input to the network
    And2 = 2,         ///< AND gate (2-input, aig::AndInverterOp)
    Xor2 = 3,         ///< XOR gate (2-input)
    Maj3 = 4,         ///< Majority gate (3-input, mig::MajOp)
    Identity = 5      ///< Identity gate (used for 1-input inverter)
  };

  /// Operation pointer and kind packed together.
  /// The kind is stored in the low bits of the pointer.
  llvm::PointerIntPair<Operation *, 3, Kind> opAndKind;

  /// Fanin edges (up to 3 inputs). For AND gates, only edges[0] and edges[1]
  /// are used. For MAJ gates, all three are used. For PrimaryInput/Constant,
  /// none are used. The inversion bit is encoded in each edge.
  Signal edges[3];

  LogicNetworkGate() : opAndKind(nullptr, Constant), edges{} {}
  LogicNetworkGate(Operation *op, Kind kind,
                   llvm::ArrayRef<Signal> operands = {})
      : opAndKind(op, kind), edges{} {
    assert(operands.size() <= 3 && "Too many operands for LogicNetworkGate");
    for (size_t i = 0; i < operands.size(); ++i)
      edges[i] = operands[i];
  }

  /// Get the kind of this gate.
  Kind getKind() const { return opAndKind.getInt(); }

  /// Get the operation pointer (nullptr for constants).
  Operation *getOperation() const { return opAndKind.getPointer(); }

  /// Get the number of fanin edges based on kind.
  unsigned getNumFanins() const {
    switch (getKind()) {
    case Constant:
    case PrimaryInput:
      return 0;
    case And2:
    case Xor2:
      return 2;
    case Maj3:
      return 3;
    case Identity:
      return 1;
    }
    llvm_unreachable("Unknown gate kind");
  }

  /// Check if this is a logic gate that can be part of a cut.
  bool isLogicGate() const {
    Kind k = getKind();
    return k == And2 || k == Xor2 || k == Maj3 || k == Identity;
  }

  /// Check if this should always be a cut input (PI or constant).
  bool isAlwaysCutInput() const {
    Kind k = getKind();
    return k == PrimaryInput || k == Constant;
  }
};

/// Flat logic network representation for efficient cut enumeration.
///
/// This class provides a mockturtle-style flat representation of the
/// combinational logic network. Each value in the MLIR IR is assigned a unique
/// index, and gates are stored in a contiguous vector for cache efficiency.
///
/// The network supports:
/// - O(1) lookup of gate information by index
/// - Compact representation with inversion encoded in edges
/// - Efficient simulation and truth table computation
///
/// Special reserved indices:
///   - Index 0: Constant 0
///   - Index 1: Constant 1
class LogicNetwork {
public:
  /// Special constant indices.
  static constexpr uint32_t kConstant0 = 0;
  static constexpr uint32_t kConstant1 = 1;

  ArrayRef<LogicNetworkGate> getGates() const { return gates; }

  LogicNetwork() {
    // Reserve index 0 for constant 0 and index 1 for constant 1
    gates.emplace_back(nullptr, LogicNetworkGate::Constant);
    gates.emplace_back(nullptr, LogicNetworkGate::Constant);
    // indexToValue needs placeholders for constants
    indexToValue.push_back(Value()); // const0
    indexToValue.push_back(Value()); // const1
  }

  /// Get a LogicEdge representing constant 0.
  static Signal getConstant0() { return Signal(kConstant0, false); }

  /// Get a LogicEdge representing constant 1 (constant 0 inverted).
  static Signal getConstant1() { return Signal(kConstant0, true); }

  /// Get or create an index for a value.
  /// If the value doesn't have an index yet, assigns one and returns the index.
  uint32_t getOrCreateIndex(Value value);

  /// Get the raw index for a value. Asserts if value is not found.
  /// Note: This returns only the index, not a Signal with inversion info.
  /// Use hasIndex() to check existence first, or use getOrCreateIndex().
  uint32_t getIndex(Value value) const;

  /// Check if a value has been indexed.
  bool hasIndex(Value value) const;

  /// Get the value for a given raw index. Asserts if index is out of bounds.
  /// Returns null Value for constant indices (0 and 1).
  Value getValue(uint32_t index) const;

  /// Fill values for the given raw indices.
  void getValues(ArrayRef<uint32_t> indices,
                 SmallVectorImpl<Value> &values) const;

  /// Get a Signal for a value.
  /// Asserts if value not found - use hasIndex() first if unsure.
  Signal getSignal(Value value, bool inverted) const {
    return Signal(getIndex(value), inverted);
  }

  /// Get or create a Signal for a value.
  Signal getOrCreateSignal(Value value, bool inverted) {
    return Signal(getOrCreateIndex(value), inverted);
  }

  /// Get the value for a given Signal (extracts index from Signal).
  Value getValue(Signal signal) const { return getValue(signal.getIndex()); }

  /// Get the gate at a given index.
  const LogicNetworkGate &getGate(uint32_t index) const { return gates[index]; }

  /// Get mutable reference to gate at index.
  LogicNetworkGate &getGate(uint32_t index) { return gates[index]; }

  /// Get the total number of nodes in the network.
  size_t size() const { return gates.size(); }

  /// Add a primary input to the network.
  uint32_t addPrimaryInput(Value value);

  /// Add a gate with explicit result value and operand signals.
  uint32_t addGate(Operation *op, LogicNetworkGate::Kind kind, Value result,
                   llvm::ArrayRef<Signal> operands = {});

  /// Add a gate using op->getResult(0) as the result value.
  uint32_t addGate(Operation *op, LogicNetworkGate::Kind kind,
                   llvm::ArrayRef<Signal> operands = {}) {
    return addGate(op, kind, op->getResult(0), operands);
  }

  /// Build the logic network from a region/block in topological order.
  /// Returns failure if the IR is not in a valid form.
  LogicalResult buildFromBlock(Block *block);

  /// Clear the network and reset to initial state.
  void clear();

private:
  /// Map from MLIR Value to network index.
  llvm::DenseMap<Value, uint32_t> valueToIndex;

  /// Map from network index to MLIR Value.
  llvm::SmallVector<Value> indexToValue;

  /// Vector of all gates in the network.
  llvm::SmallVector<LogicNetworkGate> gates;
};

/// Result of matching a cut against a pattern.
///
/// This structure contains the area and per-input delay information
/// computed during pattern matching.
///
/// The delays can be stored in two ways:
/// 1. As a reference to static/cached data (e.g., tech library delays)
///    - Use setDelayRef() for zero-cost reference (no allocation)
/// 2. As owned dynamic data (e.g., computed SOP delays)
///    - Use setOwnedDelays() to transfer ownership
///
struct MatchResult {
  /// Area cost of implementing this cut with the pattern.
  double area = 0.0;

  /// Default constructor.
  MatchResult() = default;

  /// Constructor with area and delays (by reference).
  MatchResult(double area, ArrayRef<DelayType> delays)
      : area(area), borrowedDelays(delays) {}

  /// Set delays by reference (zero-cost for static/cached delays).
  /// The caller must ensure the referenced data remains valid.
  void setDelayRef(ArrayRef<DelayType> delays) { borrowedDelays = delays; }

  /// Set delays by transferring ownership (for dynamically computed delays).
  /// This moves the data into internal storage.
  void setOwnedDelays(SmallVector<DelayType, 6> delays) {
    ownedDelays.emplace(std::move(delays));
    borrowedDelays = {};
  }

  /// Get all delays as an ArrayRef.
  ArrayRef<DelayType> getDelays() const {
    return ownedDelays.has_value() ? ArrayRef<DelayType>(*ownedDelays)
                                   : borrowedDelays;
  }

private:
  /// Borrowed delays (used when ownedDelays is empty).
  /// Points to external data provided via setDelayRef().
  ArrayRef<DelayType> borrowedDelays;

  /// Owned delays (used when present).
  /// Only allocated when setOwnedDelays() is called. When empty (std::nullopt),
  /// moving this MatchResult avoids constructing/moving the SmallVector,
  /// achieving zero-cost abstraction for the common case (borrowed delays).
  std::optional<SmallVector<DelayType, 6>> ownedDelays;
};

/// Represents a cut that has been successfully matched to a rewriting pattern.
///
/// This class encapsulates the result of matching a cut against a rewriting
/// pattern during optimization. It stores the matched pattern, the
/// cut that was matched, and timing information needed for optimization.
class MatchedPattern {
private:
  const CutRewritePattern *pattern = nullptr; ///< The matched library pattern
  SmallVector<DelayType, 1>
      arrivalTimes;  ///< Arrival times of outputs from this pattern
  double area = 0.0; ///< Area cost of this pattern

public:
  /// Default constructor creates an invalid matched pattern.
  MatchedPattern() = default;

  /// Constructor for a valid matched pattern.
  MatchedPattern(const CutRewritePattern *pattern,
                 SmallVector<DelayType, 1> arrivalTimes, double area)
      : pattern(pattern), arrivalTimes(std::move(arrivalTimes)), area(area) {}

  /// Get the arrival time of signals through this pattern.
  DelayType getArrivalTime(unsigned outputIndex) const;
  ArrayRef<DelayType> getArrivalTimes() const;
  DelayType getWorstOutputArrivalTime() const;

  /// Get the library pattern that was matched.
  const CutRewritePattern *getPattern() const;

  /// Get the area cost of using this pattern.
  double getArea() const;
};

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

  std::optional<MatchedPattern> matchedPattern;

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

  /// Get arrival times for each input of this cut.
  /// Returns failure if any input doesn't have a valid matched pattern.
  LogicalResult getInputArrivalTimes(CutEnumerator &enumerator,
                                     SmallVectorImpl<DelayType> &results) const;

  /// Matched pattern for this cut.
  void setMatchedPattern(MatchedPattern pattern) {
    matchedPattern = std::move(pattern);
  }

  /// Get the matched pattern for this cut.
  const std::optional<MatchedPattern> &getMatchedPattern() const {
    return matchedPattern;
  }
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
  Cut *bestCut = nullptr;
  bool isFrozen = false; ///< Whether cut set is finalized

public:
  /// Check if this cut set has a valid matched pattern.
  bool isMatched() const { return bestCut; }

  /// Get the cut associated with the best matched pattern.
  Cut *getBestMatchedCut() const;

  /// Finalize the cut set by removing duplicates and selecting the best
  /// pattern.
  ///
  /// This method:
  /// 1. Removes duplicate cuts based on inputs and root operation
  /// 2. Limits the number of cuts to prevent exponential growth
  /// 3. Matches each cut against available patterns
  /// 4. Selects the best pattern based on the optimization strategy
  void finalize(
      const CutRewriterOptions &options,
      llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut);

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

  /// Fail if there is a root operation that has no matching pattern.
  bool allowNoMatch = false;

  /// Put arrival times to rewritten operations.
  bool attachDebugTiming = false;

  /// Run priority cuts enumeration and dump the cut sets.
  bool testPriorityCuts = false;
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
      llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut =
          [](const Cut &) { return std::nullopt; });

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

  void dump() const;
  const llvm::MapVector<Value, std::unique_ptr<CutSet>> &getCutSets() const {
    return cutSets;
  }

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
  llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut;

  /// Flat logic network representation for efficient simulation.
  /// Built during cut enumeration and used for truth table computation.
  LogicNetwork logicNetwork;

public:
  /// Get the logic network (read-only).
  const LogicNetwork &getLogicNetwork() const { return logicNetwork; }

  /// Get the logic network (mutable).
  LogicNetwork &getLogicNetwork() { return logicNetwork; }
};

/// Base class for cut rewriting patterns used in combinational logic
/// optimization.
///
/// A CutRewritePattern represents a library component or optimization pattern
/// that can replace cuts in the combinational logic network. Each pattern
/// defines:
/// - How to recognize matching cuts and compute area/delay metrics
/// - How to transform/replace the matched cuts
///
/// Patterns can use truth table matching for efficient recognition or
/// implement custom matching logic for more complex cases.
struct CutRewritePattern {
  CutRewritePattern(mlir::MLIRContext *context) : context(context) {}
  /// Virtual destructor for base class.
  virtual ~CutRewritePattern() = default;

  /// Check if a cut matches this pattern and compute area/delay metrics.
  ///
  /// This method is called to determine if a cut can be replaced by this
  /// pattern. If the cut matches, it should return a MatchResult containing
  /// the area and per-input delays for this specific cut.
  ///
  /// If useTruthTableMatcher() returns true, this method is only
  /// called for cuts with matching truth tables.
  virtual std::optional<MatchResult> match(CutEnumerator &enumerator,
                                           const Cut &cut) const = 0;

  /// Specify truth tables that this pattern can match.
  ///
  /// If this method returns true, the pattern matcher will use truth table
  /// comparison for efficient pre-filtering. Only cuts with matching truth
  /// tables will be passed to the match() method. If it returns false, the
  /// pattern will be checked against all cuts regardless of their truth tables.
  /// This is useful for patterns that match regardless of their truth tables,
  /// such as LUT-based patterns.
  virtual bool
  useTruthTableMatcher(SmallVectorImpl<NPNClass> &matchingNPNClasses) const;

  /// Return a new operation that replaces the matched cut.
  ///
  /// Unlike MLIR's RewritePattern framework which allows arbitrary in-place
  /// modifications, this method creates a new operation to replace the matched
  /// cut rather than modifying existing operations. This constraint exists
  /// because the cut enumerator maintains references to operations throughout
  /// the circuit, making it safe to only replace the root operation of each
  /// cut while preserving all other operations unchanged.
  virtual FailureOr<Operation *> rewrite(mlir::OpBuilder &builder,
                                         CutEnumerator &enumerator,
                                         const Cut &cut) const = 0;

  /// Get the number of outputs this pattern produces.
  virtual unsigned getNumOutputs() const = 0;

  /// Get the name of this pattern. Used for debugging.
  virtual StringRef getPatternName() const { return "<unnamed>"; }

  /// Get location for this pattern(optional).
  virtual LocationAttr getLoc() const { return mlir::UnknownLoc::get(context); }

  mlir::MLIRContext *getContext() const { return context; }

private:
  mlir::MLIRContext *context;
};

/// Manages a collection of rewriting patterns for combinational logic
/// optimization.
///
/// This class organizes and provides efficient access to rewriting patterns
/// used during cut-based optimization. It maintains:
/// - A collection of all available patterns
/// - Fast lookup tables for truth table-based matching
/// - Separation of truth table vs. custom matching patterns
///
/// The pattern set is used by the CutRewriter to find suitable replacements
/// for cuts in the combinational logic network.
class CutRewritePatternSet {
public:
  /// Constructor that takes ownership of the provided patterns.
  ///
  /// During construction, patterns are analyzed and organized for efficient
  /// lookup. Truth table matchers are indexed by their NPN canonical forms.
  CutRewritePatternSet(
      llvm::SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns);

private:
  /// Owned collection of all rewriting patterns.
  llvm::SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;

  /// Fast lookup table mapping NPN canonical forms to matching patterns.
  /// Each entry maps a truth table and input size to patterns that can handle
  /// it.
  DenseMap<std::pair<APInt, unsigned>,
           SmallVector<std::pair<NPNClass, const CutRewritePattern *>>>
      npnToPatternMap;

  /// Patterns that use custom matching logic instead of NPN lookup.
  /// These patterns are checked against every cut.
  SmallVector<const CutRewritePattern *, 4> nonNPNPatterns;

  /// CutRewriter needs access to internal data structures for pattern matching.
  friend class CutRewriter;
};

/// Main cut-based rewriting algorithm for combinational logic optimization.
///
/// The CutRewriter implements a cut-based rewriting algorithm that:
/// 1. Enumerates cuts in the combinational logic network using a priority cuts
/// algorithm
/// 2. Matches cuts against available rewriting patterns
/// 3. Selects optimal patterns based on area or timing objectives
/// 4. Rewrites the circuit using the selected patterns
///
/// The algorithm processes the network in topological order, building up cut
/// sets for each node and selecting the best implementation based on the
/// specified optimization strategy.
///
/// Usage example:
/// ```cpp
/// CutRewriterOptions options;
/// options.strategy = OptimizationStrategy::Area;
/// options.maxCutInputSize = 4;
/// options.maxCutSizePerRoot = 8;
///
/// CutRewritePatternSet patterns(std::move(optimizationPatterns));
/// CutRewriter rewriter(module, options, patterns);
///
/// if (failed(rewriter.run())) {
///   // Handle rewriting failure
/// }
/// ```
class CutRewriter {
public:
  /// Constructor for the cut rewriter.
  CutRewriter(const CutRewriterOptions &options, CutRewritePatternSet &patterns)
      : options(options), patterns(patterns), cutEnumerator(options) {}

  /// Execute the complete cut-based rewriting algorithm.
  ///
  /// This method orchestrates the entire rewriting process:
  /// 1. Enumerate cuts for all nodes in the combinational logic
  /// 2. Match cuts against available patterns
  /// 3. Select optimal patterns based on strategy
  /// 4. Rewrite the circuit with selected patterns
  LogicalResult run(Operation *topOp);

private:
  /// Enumerate cuts for all nodes in the given module.
  /// Note: This preserves module boundaries and does not perform
  /// rewriting across the hierarchy.
  LogicalResult enumerateCuts(Operation *topOp);

  /// Find patterns that match a cut's truth table.
  ArrayRef<std::pair<NPNClass, const CutRewritePattern *>>
  getMatchingPatternsFromTruthTable(const Cut &cut) const;

  /// Match a cut against available patterns and compute arrival time.
  std::optional<MatchedPattern> patternMatchCut(const Cut &cut);

  /// Perform the actual circuit rewriting using selected patterns.
  LogicalResult runBottomUpRewrite(Operation *topOp);

  /// Configuration options
  const CutRewriterOptions &options;

  /// Available rewriting patterns
  const CutRewritePatternSet &patterns;

  CutEnumerator cutEnumerator;
};

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_CUT_REWRITER_H
