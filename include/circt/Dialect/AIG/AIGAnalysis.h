//===- OpCountAnalysis.h - operation count analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving the frequency of different kinds of operations found in a
// builtin.module.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_AIG_ANALYSIS_H
#define CIRCT_ANALYSIS_AIG_ANALYSIS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableList.h"
#include <memory>

namespace mlir {
class AnalysisManager;
} // namespace mlir
namespace circt {
namespace igraph {
class InstanceGraph;
}
namespace aig {

struct DebugPoint {
  DebugPoint(circt::igraph::InstancePath path, Value value, size_t bitPos,
             int64_t delay = 0, StringRef comment = "")
      : path(path), value(value), bitPos(bitPos), delay(delay),
        comment(comment) {}

  // Trait for list of debug points.
  void Profile(llvm::FoldingSetNodeID &ID) const {
    for (auto &inst : path) {
      ID.AddPointer(inst.getAsOpaquePointer());
    }
    ID.AddPointer(value.getAsOpaquePointer());
    ID.AddInteger(bitPos);
    ID.AddInteger(delay);
  }

  void print(llvm::raw_ostream &os) const;

  circt::igraph::InstancePath path;
  Value value;
  size_t bitPos;
  int64_t delay;
  StringRef comment;
};

// A class represents a path in the dataflow graph.
// The destination is: `instancePath.value[bitPos]` at time `delay` going
// through `history`.
struct DataflowPath {
  Value value;
  size_t bitPos;
  circt::igraph::InstancePath instancePath;
  int64_t delay = -1;
  llvm::ImmutableList<DebugPoint> history;
  DataflowPath(circt::igraph::InstancePath path, Value value, size_t bitPos,
               int64_t delay = 0, llvm::ImmutableList<DebugPoint> history = {})
      : value(value), bitPos(bitPos), instancePath(path), delay(delay),
        history(history) {
    assert(value);
  }

  DataflowPath() = default;
  bool operator>(const DataflowPath &other) const {
    return delay > other.delay;
  }
  bool operator<(const DataflowPath &other) const {
    return delay < other.delay;
  }

  void print(llvm::raw_ostream &os) const;
};

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

struct PathResult {
  Object fanout;
  DataflowPath fanin;
  hw::HWModuleOp root;
  PathResult(Object fanout, DataflowPath fanin, hw::HWModuleOp root)
      : fanout(fanout), fanin(fanin), root(root) {}
  PathResult(){};

  void print(llvm::raw_ostream &os) {
    os << "PathResult(";
    os << "root=" << root.getModuleNameAttr() << ", ";
    os << "fanout=";
    fanout.print(os);
    os << ", ";
    os << "fanin=";
    fanin.print(os);
    if (!fanin.history.isEmpty()) {
      os << ", history=[";
      llvm::interleaveComma(fanin.history, os,
                            [&](DebugPoint p) { p.print(os); });
      os << "]";
    }
    os << ")";
  }

  bool operator<(const PathResult &other) const {
    return fanin.delay < other.fanin.delay;
  }
};

class LongestPathAnalysis {
public:
  // Entry points for analysis.
  LongestPathAnalysis(Operation *moduleOp, mlir::AnalysisManager &am);

  // Return all paths for the given values.
  LogicalResult getResults(Value value, size_t bitPos,
                           SmallVectorImpl<PathResult> &results);

  // Paths to FFs are precomputed efficiently, return results.
  ArrayRef<PathResult> getResultsForFF();

  struct Impl;
  std::unique_ptr<Impl> impl;
};

class ResourceUsageAnalysis {
public:
  ResourceUsageAnalysis(Operation *moduleOp, mlir::AnalysisManager &am);

  struct ResourceUsage {
    ResourceUsage(uint64_t numAndInverterGates, uint64_t numDFFBits)
        : numAndInverterGates(numAndInverterGates), numDFFBits(numDFFBits) {}
    ResourceUsage() = default;
    ResourceUsage &operator+=(const ResourceUsage &other) {
      numAndInverterGates += other.numAndInverterGates;
      numDFFBits += other.numDFFBits;
      return *this;
    }

    uint64_t getNumAndInverterGates() const { return numAndInverterGates; }
    uint64_t getNumDFFBits() const { return numDFFBits; }

  private:
    uint64_t numAndInverterGates = 0;
    uint64_t numDFFBits = 0;
  };

  struct ModuleResourceUsage {
    ModuleResourceUsage(StringAttr moduleName, ResourceUsage local,
                        ResourceUsage total)
        : moduleName(moduleName), local(local), total(total) {}
    StringAttr moduleName;
    ResourceUsage local, total;
    struct InstanceResource {
      StringAttr moduleName, instanceName;
      ModuleResourceUsage *usage;
      InstanceResource(StringAttr moduleName, StringAttr instanceName,
                       ModuleResourceUsage *usage)
          : moduleName(moduleName), instanceName(instanceName), usage(usage) {}
    };
    SmallVector<InstanceResource> instances;
    ResourceUsage getTotal() const { return total; }
    void emitJSON(raw_ostream &os) const;
  };

  ModuleResourceUsage *getResourceUsage(hw::HWModuleOp top);

  // A map from the top-level module to the resource usage of the design.
  DenseMap<StringAttr, std::unique_ptr<ModuleResourceUsage>> designUsageCache;
  igraph::InstanceGraph *instanceGraph;
};

} // namespace aig
} // namespace circt

#endif // CIRCT_ANALYSIS_AIG_ANALYSIS_H
