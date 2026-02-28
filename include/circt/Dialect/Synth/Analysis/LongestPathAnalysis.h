//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the longest path analysis for the Synth dialect.
// The analysis computes the maximum delay through combinational paths in a
// circuit where each primitive operation is considered to have a unit
// delay. It handles module hierarchies and provides detailed path information
// including sources, sinks, delays, and debug points.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_LONGESTPATHANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_LONGESTPATHANALYSIS_H

#include "circt/Conversion/CombToSynth.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <variant>

namespace mlir {
class AnalysisManager;
} // namespace mlir
namespace circt {
namespace igraph {
class InstanceGraph;
} // namespace igraph
namespace synth {

// A class represents an object in the dataflow graph.
// An Object identifies a specific bit of a value at a specific instance path.
struct Object {
  Object(circt::igraph::InstancePath path, Value value, size_t bitPos)
      : instancePath(path), value(value), bitPos(bitPos) {}
  Object() = default;

  bool operator==(const Object &other) const {
    return instancePath == other.instancePath && value == other.value &&
           bitPos == other.bitPos;
  }

  void print(llvm::raw_ostream &os) const;
  Object &prependPaths(circt::igraph::InstancePathCache &cache,
                       circt::igraph::InstancePath path);

  StringAttr getName() const;

  circt::igraph::InstancePath instancePath;
  Value value;
  size_t bitPos;
};

// A debug point represents a point in the dataflow graph which carries delay
// and optional comment. Debug points are used to track the history of a path
// and provide context for debugging.
struct DebugPoint {
  DebugPoint(circt::igraph::InstancePath path, Value value, size_t bitPos,
             int64_t delay = 0, StringRef comment = "")
      : object(path, value, bitPos), delay(delay), comment(comment) {}

  // Trait for ImmutableList.
  void Profile(llvm::FoldingSetNodeID &ID) const {
    for (auto &inst : object.instancePath) {
      ID.AddPointer(inst.getAsOpaquePointer());
    }
    ID.AddPointer(object.value.getAsOpaquePointer());
    ID.AddInteger(object.bitPos);
    ID.AddInteger(delay);
  }

  void print(llvm::raw_ostream &os) const;

  Object object;
  int64_t delay;
  StringRef comment;
};

// An OpenPath represents a path from a start point with an associated
// delay and history of debug points.
struct OpenPath {
  OpenPath(circt::igraph::InstancePath path, Value value, size_t bitPos,
           int64_t delay = 0, llvm::ImmutableList<DebugPoint> history = {})
      : OpenPath(Object(path, value, bitPos), delay, history) {}
  OpenPath(Object startPoint, int64_t delay = 0,
           llvm::ImmutableList<DebugPoint> history = {})
      : startPoint(startPoint), delay(delay), history(history) {}
  OpenPath() = default;

  const Object &getStartPoint() const { return startPoint; }
  int64_t getDelay() const { return delay; }
  const llvm::ImmutableList<DebugPoint> &getHistory() const { return history; }

  void print(llvm::raw_ostream &os) const;
  OpenPath &
  prependPaths(circt::igraph::InstancePathCache &cache,
               llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
               circt::igraph::InstancePath path);

  Object startPoint;
  int64_t delay;
  // History of debug points represented by linked lists.
  // The head of the list is the farthest point from the start point.
  llvm::ImmutableList<DebugPoint> history;
};

// A DataflowPath represents a complete timing path from a end point to a
// start point with associated delay information. This is the primary result
// type for longest path analysis, containing both end points and path history.
class DataflowPath {
public:
  // end point can be either an internal circuit object or a module output port
  // This flexibility allows representing both closed paths
  // (register-to-register) and open paths (register-to-output) in a unified way
  using OutputPort = std::tuple<hw::HWModuleOp, size_t, size_t>;
  using EndPointType = std::variant<Object, OutputPort>;

  // Constructor for paths with Object end point (internal circuit nodes)
  DataflowPath(Object endPoint, OpenPath startPoint, hw::HWModuleOp root)
      : endPoint(endPoint), path(startPoint), root(root) {}

  // Constructor for paths with port end point (module output ports)
  DataflowPath(OutputPort endPoint, OpenPath startPoint, hw::HWModuleOp root)
      : endPoint(endPoint), path(startPoint), root(root) {}

  DataflowPath() = default;

  int64_t getDelay() const { return path.delay; }
  const Object &getStartPoint() const { return path.startPoint; }
  const EndPointType &getEndPoint() const { return endPoint; }
  const Object &getEndPointAsObject() const {
    return std::get<Object>(endPoint);
  }
  const OutputPort &getEndPointAsPort() const {
    return std::get<OutputPort>(endPoint);
  }
  hw::HWModuleOp getRoot() const { return root; }
  const llvm::ImmutableList<DebugPoint> &getHistory() const {
    return path.history;
  }
  const OpenPath &getPath() const { return path; }

  // Get source location for the end point (for diagnostics)
  Location getEndPointLoc();

  void setDelay(int64_t delay) { path.delay = delay; }

  void print(llvm::raw_ostream &os);
  void printEndPoint(llvm::raw_ostream &os);

  // Path elaboration for hierarchical analysis
  // Prepends instance path information to create full hierarchical paths
  DataflowPath &
  prependPaths(circt::igraph::InstancePathCache &cache,
               llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
               circt::igraph::InstancePath path);

private:
  EndPointType endPoint; // Either Object or (port_index, bit_index)
  OpenPath path;         // The actual timing path with history
  hw::HWModuleOp root;   // Root module for this path
};

// JSON serialization for DataflowPath
llvm::json::Value toJSON(const circt::synth::DataflowPath &path);

/// Configuration options for the longest path analysis.
///
/// This struct controls various aspects of the analysis behavior, including
/// debugging features, computation modes, and optimization settings. Different
/// combinations of options are suitable for different use cases.
///
/// Example usage:
///   // For lazily computing paths with debug info
///   LongestPathAnalysisOption options(true, true, false);
///
///   // For fast critical path identification only
///   LongestPathAnalysisOption options(false, false, true);
struct LongestPathAnalysisOptions {
  /// Enable collection of debug points along timing paths.
  /// When enabled, records intermediate points with delay values and comments
  /// for debugging, visualization, and understanding delay contributions.
  /// Moderate performance impact.
  bool collectDebugInfo = false;

  /// Enable lazy computation mode for on-demand analysis.
  /// Performs delay computations lazily and caches results, tracking IR
  /// changes. Better for iterative workflows where only specific paths
  /// are queried. Disables parallel processing.
  bool lazyComputation = false;

  /// Keep only the maximum delay path per end point.
  /// Focuses on finding maximum delays, discarding non-critical paths.
  /// Significantly faster and uses less memory when only delay bounds
  /// are needed rather than complete path enumeration.
  bool keepOnlyMaxDelayPaths = false;

  /// Name of the top module for the analysis.
  /// If empty, the top module is inferred from the instance graph.
  StringAttr topModuleName = {};

  /// Construct analysis options with the specified settings.
  LongestPathAnalysisOptions(bool collectDebugInfo = false,
                             bool lazyComputation = false,
                             bool keepOnlyMaxDelayPaths = false,
                             StringAttr topModuleName = {})
      : collectDebugInfo(collectDebugInfo), lazyComputation(lazyComputation),
        keepOnlyMaxDelayPaths(keepOnlyMaxDelayPaths),
        topModuleName(topModuleName) {}
};

// This analysis finds the longest paths in the dataflow graph across modules.
// Also be aware of the lifetime of the analysis, the results would be
// invalid if the IR is modified. Currently there is no way to efficiently
// update the analysis results, so it's recommended to only use this analysis
// once on a design, and store the results in a separate data structure which
// users can manage the lifetime.
class LongestPathAnalysis {
public:
  // Entry points for analysis.
  LongestPathAnalysis(Operation *moduleOp, mlir::AnalysisManager &am,
                      const LongestPathAnalysisOptions &option = {});
  ~LongestPathAnalysis();

  // Return all longest paths to each start point for the given value and bit
  // position.
  LogicalResult computeGlobalPaths(Value value, size_t bitPos,
                                   SmallVectorImpl<DataflowPath> &results);

  // Compute local paths for specified value and bit. Local paths are paths
  // that are fully contained within a module.
  FailureOr<ArrayRef<OpenPath>> computeLocalPaths(Value value, size_t bitPos);

  // Return the maximum delay to the given value and bit position. If bitPos is
  // negative, then return the maximum delay across all bits.
  FailureOr<int64_t> getMaxDelay(Value value, int64_t bitPos = -1);

  // Return the average of the maximum delays across all bits of the given
  // value, which is useful approximation for the delay of the value. For each
  // bit position, finds all paths and takes the maximum delay. Then averages
  // these maximum delays across all bits of the value.
  FailureOr<int64_t> getAverageMaxDelay(Value value);

  // Return paths that are closed under the given module. Closed paths are
  // typically register-to-register paths. A closed path is a path that starts
  // and ends at sequential elements (registers/flip-flops), forming a complete
  // timing path through combinational logic. The path may cross module
  // boundaries but both end points are sequential elements, not ports.
  LogicalResult getInternalPaths(StringAttr moduleName,
                                 SmallVectorImpl<DataflowPath> &results,
                                 bool elaboratePaths = false) const;

  // Return input-to-internal timing paths for the given module.
  // These are open paths from module input ports to internal sequential
  // elements (registers/flip-flops).
  LogicalResult getOpenPathsFromInputPortsToInternal(
      StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const;

  // Return internal-to-output timing paths for the given module.
  // These are open paths from internal sequential elements to module output
  // ports.
  LogicalResult getOpenPathsFromInternalToOutputPorts(
      StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const;

  // Get all timing paths in the given module including both closed and open
  // paths. This is a convenience method that combines results from
  // getInternalPaths, getOpenPathsFromInputPortsToInternal, and
  // getOpenPathsFromInternalToOutputPorts. If elaboratedPaths is true, paths
  // include full hierarchical instance information.
  LogicalResult getAllPaths(StringAttr moduleName,
                            SmallVectorImpl<DataflowPath> &results,
                            bool elaboratePaths = false) const;

  // Return true if the analysis is available for the given module.
  bool isAnalysisAvailable(StringAttr moduleName) const;

  // Return the top nodes that were used for the analysis.
  llvm::ArrayRef<hw::HWModuleOp> getTopModules() const;

  MLIRContext *getContext() const { return ctx; }

protected:
  friend class IncrementalLongestPathAnalysis;
  struct Impl;
  Impl *impl;

private:
  mlir::MLIRContext *ctx;
  bool isAnalysisValid = true;
};

// Incremental version of longest path analysis that supports on-demand
// computation and tracks IR modifications to maintain validity.
class IncrementalLongestPathAnalysis : private LongestPathAnalysis,
                                       public mlir::PatternRewriter::Listener {
public:
  IncrementalLongestPathAnalysis(Operation *moduleOp, mlir::AnalysisManager &am)
      : LongestPathAnalysis(
            moduleOp, am,
            LongestPathAnalysisOptions(/*collectDebugInfo=*/false,
                                       /*lazyComputation=*/true,
                                       /*keepOnlyMaxDelayPaths=*/true)) {}

  IncrementalLongestPathAnalysis(Operation *moduleOp, mlir::AnalysisManager &am,
                                 const LongestPathAnalysisOptions &option)
      : LongestPathAnalysis(moduleOp, am, option) {
    assert(option.lazyComputation && "Lazy computation must be enabled");
  }

  using LongestPathAnalysis::getAverageMaxDelay;
  using LongestPathAnalysis::getMaxDelay;

  // Check if operation can be safely modified without invalidating analysis.
  bool isOperationValidToMutate(Operation *op) const;

  // PatternRewriter::Listener interface - called automatically.
  void notifyOperationModified(Operation *op) override;
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;
  void notifyOperationErased(Operation *op) override;
};

// A collection of longest paths. The data structure owns the paths, and used
// for computing statistics and CAPI.
class LongestPathCollection {
public:
  LongestPathCollection(MLIRContext *ctx) : ctx(ctx){};
  const DataflowPath &getPath(unsigned index) const { return paths[index]; }
  MLIRContext *getContext() const { return ctx; }
  llvm::SmallVector<DataflowPath, 64> paths;

  // Sort the paths by delay in descending order.
  void sortInDescendingOrder();

  // Drop all paths except the longest path per end point or start point.
  void dropNonCriticalPaths(bool perEndPoint = true);

  // Merge another collection into this one.
  void merge(const LongestPathCollection &other);

private:
  MLIRContext *ctx;
};

// Register prerequisite passes for Synthth longest path analysis.
// This includes the Comb to Synth conversion pass and necessary
// canonicalization passes to clean up the IR.
inline void registerSynthAnalysisPrerequisitePasses() {
  hw::registerHWAggregateToCombPass();
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertCombToSynth();
  });
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
}

} // namespace synth
} // namespace circt

namespace llvm {
// Provide DenseMapInfo for Object.
template <>
struct DenseMapInfo<circt::synth::Object> {
  using Info = llvm::DenseMapInfo<
      std::tuple<circt::igraph::InstancePath, mlir::Value, size_t>>;
  using Object = circt::synth::Object;
  static Object getEmptyKey() {
    auto [path, value, bitPos] = Info::getEmptyKey();
    return Object(path, value, bitPos);
  }

  static Object getTombstoneKey() {
    auto [path, value, bitPos] = Info::getTombstoneKey();
    return Object(path, value, bitPos);
  }
  static llvm::hash_code getHashValue(Object object) {
    return Info::getHashValue(
        {object.instancePath, object.value, object.bitPos});
  }
  static bool isEqual(const Object &a, const Object &b) {
    return Info::isEqual({a.instancePath, a.value, a.bitPos},
                         {b.instancePath, b.value, b.bitPos});
  }
};

} // namespace llvm

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_LONGESTPATHANALYSIS_H
