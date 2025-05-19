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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ImmutableList.h"

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
  circt::igraph::InstancePath instancePath;
  Value value;
  size_t bitPos;
  Object(circt::igraph::InstancePath path, Value value, size_t bitPos)
      : instancePath(path), value(value), bitPos(bitPos) {}
  Object() = default;
  void print(llvm::raw_ostream &os) const;

  bool operator==(const Object &other) const {
    return instancePath == other.instancePath && value == other.value &&
           bitPos == other.bitPos;
  }
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
  Object fanIn;
  int64_t delay;
  // History of debug points represented by linked lists.
  // The head of the list is the farthest point from the fanIn.
  llvm::ImmutableList<DebugPoint> history;
  OpenPath(circt::igraph::InstancePath path, Value value, size_t bitPos,
           int64_t delay = 0, llvm::ImmutableList<DebugPoint> history = {})
      : fanIn(path, value, bitPos), delay(delay), history(history) {}

  OpenPath() = default;
  void print(llvm::raw_ostream &os) const;
};

// A DataflowPath is a complete path from a fanout to a fanin with associated
// delay information.
class DataflowPath {
public:
  DataflowPath(Object fanOut, OpenPath fanIn, hw::HWModuleOp root)
      : fanOut(fanOut), path(fanIn), root(root) {}
  DataflowPath() = default;

  int64_t getDelay() const { return path.delay; }
  const Object &getFanIn() const { return path.fanIn; }
  const Object &getFanOut() const { return fanOut; }
  const hw::HWModuleOp &getRoot() const { return root; }

  void print(llvm::raw_ostream &os);
  void setDelay(int64_t delay) { path.delay = delay; }

private:
  Object fanOut;
  OpenPath path;
  hw::HWModuleOp root;
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
  LongestPathAnalysis(Operation *moduleOp, mlir::AnalysisManager &am);
  ~LongestPathAnalysis();

  // Return all longest paths to each Fanin for the given value and bit
  // position.
  LogicalResult getResults(Value value, size_t bitPos,
                           SmallVectorImpl<DataflowPath> &results) const;

  // Return the average of the maximum delays across all bits of the given
  // value, which is useful approximation for the delay of the value. For each
  // bit position, finds all paths and takes the maximum delay. Then averages
  // these maximum delays across all bits of the value.
  int64_t getAverageMaxDelay(Value value) const;

  // Paths to paths that are closed under the give module. Results are
  // sorted by delay from longest to shortest. Closed paths are typically
  // register-to-register paths.
  LogicalResult getClosedPaths(StringAttr moduleName,
                               SmallVectorImpl<DataflowPath> &results) const;

  // Return open paths for the given module. Results are sorted by delay from
  // longest to shortest. Open paths are typically input-to-register or
  // register-to-output paths.
  LogicalResult
  getOpenPaths(StringAttr moduleName,
               SmallVectorImpl<std::pair<Object, OpenPath>> &openPathsToFF,
               SmallVectorImpl<std::tuple<size_t, size_t, OpenPath>>
                   &openPathsFromOutputPorts) const;

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

private:
  struct Impl;
  Impl *impl;
};

} // namespace aig
} // namespace circt

#endif // CIRCT_ANALYSIS_AIG_ANALYSIS_H
