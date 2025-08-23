//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the longest path analysis for the AIG dialect.
// The analysis computes the maximum delay through combinational paths in a
// circuit where each AIG and-inverter operation is considered to have a unit
// delay. It handles module hierarchies and provides detailed path information
// including sources, sinks, delays, and debug points.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_AIG_ANALYSIS_H
#define CIRCT_ANALYSIS_AIG_ANALYSIS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <variant>

namespace mlir {
class AnalysisManager;
} // namespace mlir
namespace circt {
namespace igraph {
class InstanceGraph;
} // namespace igraph
namespace aig {

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

// An OpenPath represents a path from a fan-in with an associated
// delay and history of debug points.
struct OpenPath {
  OpenPath(circt::igraph::InstancePath path, Value value, size_t bitPos,
           int64_t delay = 0, llvm::ImmutableList<DebugPoint> history = {})
      : fanIn(path, value, bitPos), delay(delay), history(history) {}
  OpenPath() = default;

  const Object &getFanIn() const { return fanIn; }
  int64_t getDelay() const { return delay; }
  const llvm::ImmutableList<DebugPoint> &getHistory() const { return history; }

  void print(llvm::raw_ostream &os) const;
  OpenPath &
  prependPaths(circt::igraph::InstancePathCache &cache,
               llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
               circt::igraph::InstancePath path);

  Object fanIn;
  int64_t delay;
  // History of debug points represented by linked lists.
  // The head of the list is the farthest point from the fanIn.
  llvm::ImmutableList<DebugPoint> history;
};

// A DataflowPath represents a complete timing path from a fanout to a fanin
// with associated delay information. This is the primary result type for
// longest path analysis, containing both endpoints and path history.
class DataflowPath {
public:
  // FanOut can be either an internal circuit object or a module output port
  // This flexibility allows representing both closed paths
  // (register-to-register) and open paths (register-to-output) in a unified way
  using OutputPort = std::tuple<hw::HWModuleOp, size_t, size_t>;
  using FanOutType = std::variant<Object, OutputPort>;

  // Constructor for paths with Object fanout (internal circuit nodes)
  DataflowPath(Object fanOut, OpenPath fanIn, hw::HWModuleOp root)
      : fanOut(fanOut), path(fanIn), root(root) {}

  // Constructor for paths with port fanout (module output ports)
  DataflowPath(OutputPort fanOut, OpenPath fanIn, hw::HWModuleOp root)
      : fanOut(fanOut), path(fanIn), root(root) {}

  DataflowPath() = default;

  int64_t getDelay() const { return path.delay; }
  const Object &getFanIn() const { return path.fanIn; }
  const FanOutType &getFanOut() const { return fanOut; }
  const Object &getFanOutAsObject() const { return std::get<Object>(fanOut); }
  const OutputPort &getFanOutAsPort() const {
    return std::get<OutputPort>(fanOut);
  }
  hw::HWModuleOp getRoot() const { return root; }
  const llvm::ImmutableList<DebugPoint> &getHistory() const {
    return path.history;
  }
  const OpenPath &getPath() const { return path; }

  // Get source location for the fanout point (for diagnostics)
  Location getFanOutLoc();

  void setDelay(int64_t delay) { path.delay = delay; }

  void print(llvm::raw_ostream &os);
  void printFanOut(llvm::raw_ostream &os);

  // Path elaboration for hierarchical analysis
  // Prepends instance path information to create full hierarchical paths
  DataflowPath &
  prependPaths(circt::igraph::InstancePathCache &cache,
               llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
               circt::igraph::InstancePath path);

private:
  FanOutType fanOut;   // Either Object or (port_index, bit_index)
  OpenPath path;       // The actual timing path with history
  hw::HWModuleOp root; // Root module for this path
};

// JSON serialization for DataflowPath
llvm::json::Value toJSON(const circt::aig::DataflowPath &path);

// Options for the longest path analysis.
struct LongestPathAnalysisOption {
  bool traceDebugPoints = false;
  bool incremental = false;

  LongestPathAnalysisOption(bool traceDebugPoints, bool incremental)
      : traceDebugPoints(traceDebugPoints), incremental(incremental) {}
  LongestPathAnalysisOption() = default;
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
                      const LongestPathAnalysisOption &option = {});
  ~LongestPathAnalysis();

  // Return all longest paths to each Fanin for the given value and bit
  // position.
  LogicalResult getResults(Value value, size_t bitPos,
                           SmallVectorImpl<DataflowPath> &results) const;

  // Return the maximum delay to the given value for all bit positions.
  int64_t getMaxDelay(Value value) const;

  // Return the average of the maximum delays across all bits of the given
  // value, which is useful approximation for the delay of the value. For each
  // bit position, finds all paths and takes the maximum delay. Then averages
  // these maximum delays across all bits of the value.
  int64_t getAverageMaxDelay(Value value) const;

  // Return paths that are closed under the given module. Closed paths are
  // typically register-to-register paths. A closed path is a path that starts
  // and ends at sequential elements (registers/flip-flops), forming a complete
  // timing path through combinational logic. The path may cross module
  // boundaries but both endpoints are sequential elements, not ports.
  LogicalResult getClosedPaths(StringAttr moduleName,
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
  // getClosedPaths, getOpenPathsFromInputPortsToInternal, and
  // getOpenPathsFromInternalToOutputPorts. If elaboratedPaths is true, paths
  // include full hierarchical instance information.
  LogicalResult getAllPaths(StringAttr moduleName,
                            SmallVectorImpl<DataflowPath> &results,
                            bool elaboratePaths = false) const;

  // Return true if the analysis is available for the given module.
  bool isAnalysisAvailable(StringAttr moduleName) const;

  // Return the top nodes that were used for the analysis.
  llvm::ArrayRef<hw::HWModuleOp> getTopModules() const;

  // This is the name of the attribute that can be attached to the module
  // to specify the top module for the analysis. This is optional, if not
  // specified, the analysis will infer the top module from the instance graph.
  // However it's recommended to specify it, as the entire module tends to
  // contain testbench or verification modules, which may have expensive paths
  // that are not of interest.
  static StringRef getTopModuleNameAttrName() {
    return "aig.longest-path-analysis-top";
  }

  MLIRContext *getContext() const { return ctx; }

protected:
  friend class IncrementalLongestPathAnalysis;
  struct Impl;
  Impl *impl;

private:
  mlir::MLIRContext *ctx;
};

// For incremental updates, this class exposes subset of the analysis results
// and allows incremental updates.
class IncrementalLongestPathAnalysis : private LongestPathAnalysis,
                                       public mlir::PatternRewriter::Listener {
public:
  IncrementalLongestPathAnalysis(Operation *moduleOp, mlir::AnalysisManager &am)
      : LongestPathAnalysis(moduleOp, am,
                            LongestPathAnalysisOption(false, true)) {}

  FailureOr<int64_t> getOrComputeDelay(Value value, size_t bitPos);
  FailureOr<ArrayRef<OpenPath>> getOrComputePaths(Value value, size_t bitPos);

  // Returns true if the analysis is still valid after the value is erased.
  // If the client erased a value, then the analyais cannot be used anymore.
  // A client should check this before erasing a value, and if this
  // returns false, the client should raise an error or should not rely on the
  // analysis.
  bool isValueValidToErase(Value value) const;

  // MLIR PatternRewriter::Listener interface.
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;
  void notifyOperationErased(Operation *op) override;

  bool isAnalysisValid = true;
};

// A wrapper class for the longest path analysis that also traces debug points.
// This is necessary for analysis manager to cache the analysis results.
class LongestPathAnalysisWithTrace : public LongestPathAnalysis {
public:
  LongestPathAnalysisWithTrace(Operation *moduleOp, mlir::AnalysisManager &am)
      : LongestPathAnalysis(moduleOp, am, {true, false}) {}
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

  // Sort and drop all paths except the longest path per fanout point.
  void sortAndDropNonCriticalPathsPerFanOut();

private:
  MLIRContext *ctx;
};

} // namespace aig
} // namespace circt

namespace llvm {
// Provide DenseMapInfo for Object.
template <>
struct DenseMapInfo<circt::aig::Object> {
  using Info = llvm::DenseMapInfo<
      std::tuple<circt::igraph::InstancePath, mlir::Value, size_t>>;
  using Object = circt::aig::Object;
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

#endif // CIRCT_ANALYSIS_AIG_ANALYSIS_H
