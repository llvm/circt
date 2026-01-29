//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis computes the longest combinational paths through a circuit
// represented in the AIG (And-Inverter Graph) dialect. The key aspects are:
//
// - Each AIG and-inverter operation is considered to have a unit delay of 1
// - The analysis traverses the circuit graph from inputs/registers to outputs
// - It handles hierarchical designs by analyzing modules bottom-up
// - Results include full path information with delays and debug points
// - Caching is used extensively to improve performance on large designs
//
// The core algorithm works as follows:
// 1. Build an instance graph of the full design hierarchy
// 2. Analyze modules in post-order (children before parents)
// 3. For each module:
//    - Trace paths from inputs and registers
//    - Propagate delays through logic and across module boundaries
//    - Record maximum delay and path info for each node
// 4. Combine results across hierarchy to get full chip critical paths
//
// The analysis handles both closed paths (register-to-register) and open
// paths (input-to-register, register-to-output) across the full hierarchy.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT//MapVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#define DEBUG_TYPE "aig-longest-path-analysis"
using namespace circt;
using namespace synth;

static size_t getBitWidth(Value value) {
  if (auto vecType = dyn_cast<seq::ClockType>(value.getType()))
    return 1;
  if (auto memory = dyn_cast<seq::FirMemType>(value.getType()))
    return memory.getWidth();
  return hw::getBitWidth(value.getType());
}

template <typename T, typename Key>
static void
deduplicatePathsImpl(SmallVectorImpl<T> &results, size_t startIndex,
                     llvm::function_ref<Key(const T &)> keyFn,
                     llvm::function_ref<int64_t(const T &)> delayFn) {
  // Take only maximum for each path destination.
  DenseMap<Key, size_t> keyToIndex;
  for (size_t i = startIndex; i < results.size(); ++i) {
    auto &path = results[i];
    auto key = keyFn(path);
    auto delay = delayFn(path);
    auto it = keyToIndex.find(key);
    if (it == keyToIndex.end()) {
      // Insert a new entry.
      size_t newIndex = keyToIndex.size() + startIndex;
      keyToIndex[key] = newIndex;
      results[newIndex] = std::move(results[i]);
      continue;
    }
    if (delay > delayFn(results[it->second]))
      results[it->second] = std::move(results[i]);
  }

  results.resize(keyToIndex.size() + startIndex);
}

// Filter and optimize OpenPaths based on analysis configuration.
static void filterPaths(SmallVectorImpl<OpenPath> &results,
                        bool keepOnlyMaxDelay, bool isLocalScope) {
  if (results.empty())
    return;

  // Fast path for local scope with max-delay filtering:
  // Simply find and keep only the single longest delay path
  if (keepOnlyMaxDelay && isLocalScope) {
    OpenPath maxDelay;
    maxDelay.delay = -1; // Initialize to invalid delay

    // Find the path with maximum delay
    for (auto &path : results) {
      if (path.delay > maxDelay.delay)
        maxDelay = path;
    }

    // Replace all paths with just the maximum delay path
    results.clear();
    if (maxDelay.delay >= 0) // Only add if we found a valid path
      results.push_back(maxDelay);
    return;
  }

  // Remove paths with identical start point, keeping only the longest delay
  // path for each unique points.
  deduplicatePathsImpl<OpenPath, Object>(
      results, 0, [](const auto &path) { return path.startPoint; },
      [](const auto &path) { return path.delay; });

  // Global scope max-delay filtering:
  // Keep all module input ports (BlockArguments) but only the single
  // longest internal path to preserve boundary information
  if (keepOnlyMaxDelay) {
    assert(!isLocalScope);
    size_t writeIndex = 0;
    OpenPath maxDelay;
    maxDelay.delay = -1; // Initialize to invalid delay

    for (size_t i = 0; i < results.size(); ++i) {
      // Preserve all module input port paths (BlockArguments) since the input
      // port might be the critical path.
      if (isa<BlockArgument>(results[i].getStartPoint().value)) {
        // Keep all module input port paths, as inputs themselves may be
        // critical
        results[writeIndex++] = results[i];
      } else {
        // For internal paths, track only the maximum delay path
        if (results[i].delay > maxDelay.delay)
          maxDelay = results[i];
      }
    }

    // Resize to remove processed internal paths, then add the max delay path
    results.resize(writeIndex);
    if (maxDelay.delay >= 0) // Only add if we found a valid internal path
      results.push_back(maxDelay);
  }
}

static void filterPaths(SmallVectorImpl<DataflowPath> &results,
                        size_t startIndex = 0) {
  deduplicatePathsImpl<DataflowPath,
                       std::pair<DataflowPath::EndPointType, Object>>(
      results, startIndex,
      [](const DataflowPath &path) {
        return std::pair(path.getEndPoint(), path.getStartPoint());
      },
      [](const DataflowPath &path) { return path.getDelay(); });
}

static llvm::ImmutableList<DebugPoint>
mapList(llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
        llvm::ImmutableList<DebugPoint> list,
        llvm::function_ref<DebugPoint(DebugPoint)> fn) {
  if (list.isEmpty())
    return list;
  auto &head = list.getHead();
  return debugPointFactory->add(fn(head),
                                mapList(debugPointFactory, list.getTail(), fn));
}

static llvm::ImmutableList<DebugPoint>
concatList(llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
           llvm::ImmutableList<DebugPoint> lhs,
           llvm::ImmutableList<DebugPoint> rhs) {
  if (lhs.isEmpty())
    return rhs;
  return debugPointFactory->add(
      lhs.getHead(), concatList(debugPointFactory, lhs.getTail(), rhs));
}

static StringAttr getNameImpl(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto op = dyn_cast<hw::HWModuleOp>(arg.getParentBlock()->getParentOp());
    if (!op) {
      // TODO: Handle other operations.
      return StringAttr::get(value.getContext(), "<unknown-argument>");
    }
    return op.getArgName(arg.getArgNumber());
  }
  return TypeSwitch<Operation *, StringAttr>(value.getDefiningOp())
      .Case<seq::CompRegOp, seq::FirRegOp>(
          [](auto op) { return op.getNameAttr(); })
      .Case<hw::InstanceOp>([&](hw::InstanceOp op) {
        SmallString<16> str;
        str += op.getInstanceName();
        str += ".";
        str += cast<StringAttr>(
            op.getResultNamesAttr()[cast<OpResult>(value).getResultNumber()]);
        return StringAttr::get(op.getContext(), str);
      })
      .Case<seq::FirMemReadOp>([&](seq::FirMemReadOp op) {
        llvm::SmallString<16> str;
        str += op.getMemory().getDefiningOp<seq::FirMemOp>().getNameAttr();
        str += ".read_port";
        return StringAttr::get(value.getContext(), str);
      })
      .Case<seq::FirMemReadWriteOp>([&](seq::FirMemReadWriteOp op) {
        llvm::SmallString<16> str;
        str += op.getMemory().getDefiningOp<seq::FirMemOp>().getNameAttr();
        str += ".rw_port";
        return StringAttr::get(value.getContext(), str);
      })
      .Case<seq::FirMemOp>([&](seq::FirMemOp op) {
        llvm::SmallString<16> str;
        str += op.getMemory().getDefiningOp<seq::FirMemOp>().getNameAttr();
        str += ".write_port";
        return StringAttr::get(value.getContext(), str);
      })
      .Default([&](auto op) {
        if (auto name = op->template getAttrOfType<StringAttr>("sv.namehint"))
          return name;
        llvm::errs() << "Unknown op: " << *op << "\n";
        return StringAttr::get(value.getContext(), "");
      });
}

static void printObjectImpl(llvm::raw_ostream &os, const Object &object,
                            int64_t delay = -1,
                            llvm::ImmutableList<DebugPoint> history = {},
                            StringRef comment = "") {
  std::string pathString;
  llvm::raw_string_ostream osPath(pathString);
  object.instancePath.print(osPath);
  os << "Object(" << pathString << "." << object.getName().getValue() << "["
     << object.bitPos << "]";
  if (delay != -1)
    os << ", delay=" << delay;
  if (!history.isEmpty()) {
    os << ", history=[";
    llvm::interleaveComma(history, os, [&](DebugPoint p) { p.print(os); });
    os << "]";
  }
  if (!comment.empty())
    os << ", comment=\"" << comment << "\"";
  os << ")";
}

template <typename T>
static int64_t getMaxDelayInPaths(ArrayRef<T> paths) {
  int64_t maxDelay = 0;
  for (auto &path : paths)
    maxDelay = std::max(maxDelay, path.getDelay());
  return maxDelay;
}

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void OpenPath::print(llvm::raw_ostream &os) const {
  printObjectImpl(os, startPoint, delay, history);
}

void DebugPoint::print(llvm::raw_ostream &os) const {
  printObjectImpl(os, object, delay, {}, comment);
}

void Object::print(llvm::raw_ostream &os) const { printObjectImpl(os, *this); }

StringAttr Object::getName() const { return getNameImpl(value); }

void DataflowPath::printEndPoint(llvm::raw_ostream &os) {
  if (auto *object = std::get_if<Object>(&endPoint)) {
    object->print(os);
  } else {
    auto &[module, resultNumber, bitPos] =
        *std::get_if<DataflowPath::OutputPort>(&endPoint);
    auto outputPortName = root.getOutputName(resultNumber);
    os << "Object($root." << outputPortName << "[" << bitPos << "])";
  }
}

void DataflowPath::print(llvm::raw_ostream &os) {
  os << "root=" << root.getModuleName() << ", ";
  os << "endPoint=";
  printEndPoint(os);
  os << ", ";
  os << "startPoint=";
  path.print(os);
}

//===----------------------------------------------------------------------===//
// OpenPath
//===----------------------------------------------------------------------===//

Object &Object::prependPaths(circt::igraph::InstancePathCache &cache,
                             circt::igraph::InstancePath path) {
  instancePath = cache.concatPath(path, instancePath);
  return *this;
}

OpenPath &OpenPath::prependPaths(
    circt::igraph::InstancePathCache &cache,
    llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
    circt::igraph::InstancePath path) {
  startPoint.prependPaths(cache, path);
  if (debugPointFactory)
    this->history = mapList(debugPointFactory, this->history,
                            [&](DebugPoint p) -> DebugPoint {
                              p.object.prependPaths(cache, path);
                              return p;
                            });
  return *this;
}

//===----------------------------------------------------------------------===//
// DataflowPath
//===----------------------------------------------------------------------===//

DataflowPath &DataflowPath::prependPaths(
    circt::igraph::InstancePathCache &cache,
    llvm::ImmutableListFactory<DebugPoint> *debugPointFactory,
    circt::igraph::InstancePath path) {
  this->path.prependPaths(cache, debugPointFactory, path);
  if (!path.empty()) {
    auto root = path.top()->getParentOfType<hw::HWModuleOp>();
    assert(root && "root is not a hw::HWModuleOp");
    this->root = root;
  }

  // If the end point is an object, prepend the path.
  if (auto *object = std::get_if<Object>(&endPoint))
    object->prependPaths(cache, path);

  return *this;
}

Location DataflowPath::getEndPointLoc() {
  // If the end point is an object, return the location of the object.
  if (auto *object = std::get_if<Object>(&endPoint))
    return object->value.getLoc();

  // Return output port location.
  auto &[module, resultNumber, bitPos] =
      *std::get_if<DataflowPath::OutputPort>(&endPoint);
  return module.getOutputLoc(resultNumber);
}

//===----------------------------------------------------------------------===//
// JSON serialization
//===----------------------------------------------------------------------===//

static llvm::json::Value toJSON(const circt::igraph::InstancePath &path) {
  llvm::json::Array result;
  for (auto op : path) {
    llvm::json::Object obj;
    obj["instance_name"] = op.getInstanceName();
    obj["module_name"] = op.getReferencedModuleNames()[0];
    result.push_back(std::move(obj));
  }
  return result;
}

static llvm::json::Value toJSON(const circt::synth::Object &object) {
  return llvm::json::Object{
      {"instance_path", toJSON(object.instancePath)},
      {"name", object.getName().getValue()},
      {"bit_pos", object.bitPos},
  };
}

static llvm::json::Value toJSON(const DataflowPath::EndPointType &path,
                                hw::HWModuleOp root) {
  using namespace llvm::json;
  if (auto *object = std::get_if<circt::synth::Object>(&path))
    return toJSON(*object);

  auto &[module, resultNumber, bitPos] =
      *std::get_if<DataflowPath::OutputPort>(&path);
  return llvm::json::Object{
      {"instance_path", {}}, // Instance path is empty for output ports.
      {"name", root.getOutputName(resultNumber)},
      {"bit_pos", bitPos},
  };
}

static llvm::json::Value toJSON(const DebugPoint &point) {
  return llvm::json::Object{
      {"object", toJSON(point.object)},
      {"delay", point.delay},
      {"comment", point.comment},
  };
}

static llvm::json::Value toJSON(const OpenPath &path) {
  llvm::json::Array history;
  for (auto &point : path.history)
    history.push_back(toJSON(point));
  return llvm::json::Object{{"start_point", toJSON(path.startPoint)},
                            {"delay", path.delay},
                            {"history", std::move(history)}};
}

llvm::json::Value circt::synth::toJSON(const DataflowPath &path) {
  return llvm::json::Object{
      {"end_point", ::toJSON(path.getEndPoint(), path.getRoot())},
      {"path", ::toJSON(path.getPath())},
      {"root", path.getRoot().getModuleName()},
  };
}

class LocalVisitor;

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//
/// This class provides a thread-safe interface to access the analysis results.

class Context {
public:
  Context(igraph::InstanceGraph *instanceGraph,
          const LongestPathAnalysisOptions &option)
      : instanceGraph(instanceGraph), option(option) {}
  void notifyStart(StringAttr name) {
    std::lock_guard<llvm::sys::SmartMutex<true>> lock(mutex);
    running.insert(name);
    llvm::dbgs() << "[Timing] " << name << " started. running=[";
    for (auto &name : running)
      llvm::dbgs() << name << " ";
    llvm::dbgs() << "]\n";
  }

  void notifyEnd(StringAttr name) {
    std::lock_guard<llvm::sys::SmartMutex<true>> lock(mutex);
    running.remove(name);

    llvm::dbgs() << "[Timing] " << name << " finished. running=[";
    for (auto &name : running)
      llvm::dbgs() << name << " ";
    llvm::dbgs() << "]\n";
  }

  // Lookup a local visitor for `name`.
  const LocalVisitor *getLocalVisitor(StringAttr name) const;

  // Lookup a mutable local visitor for `name`.
  LocalVisitor *getLocalVisitorMutable(StringAttr name) const;

  // A map from the module name to the local visitor.
  llvm::MapVector<StringAttr, std::unique_ptr<LocalVisitor>> localVisitors;

  // This is non-null only if `module` is a ModuleOp.
  circt::igraph::InstanceGraph *instanceGraph = nullptr;

  bool doTraceDebugPoints() const { return option.collectDebugInfo; }
  bool doLazyComputation() const { return option.lazyComputation; }
  bool doKeepOnlyMaxDelayPaths() const { return option.keepOnlyMaxDelayPaths; }
  bool isLocalScope() const { return instanceGraph == nullptr; }
  StringAttr getTopModuleName() const { return option.topModuleName; }

private:
  bool isRunningParallel() const { return !doLazyComputation(); }
  llvm::sys::SmartMutex<true> mutex;
  llvm::SetVector<StringAttr> running;
  LongestPathAnalysisOptions option;
};

//===----------------------------------------------------------------------===//
// OperationAnalyzer
//===----------------------------------------------------------------------===//

// OperationAnalyzer handles timing analysis for individual operations that
// haven't been converted to AIG yet. It creates isolated modules for each
// unique operation type/signature combination, runs the conversion pipeline
// (HW -> Comb -> AIG), and analyzes the resulting AIG representation.
//
// This is used as a fallback when the main analysis encounters operations
// that don't have direct AIG equivalents. The analyzer:
// 1. Creates a wrapper HW module for the operation
// 2. Runs the standard conversion pipeline to lower it to AIG
// 3. Analyzes the resulting AIG to compute timing paths
// 4. Caches results based on operation name and function signature
class OperationAnalyzer {
public:
  // Constructor creates a new analyzer with an empty module for building
  // temporary operation wrappers.
  // The analyzer needs its own Context separate from the main analysis
  // to avoid using onlyMaxDelay mode, as we require precise input
  // dependency information for accurate delay propagation
  OperationAnalyzer(Location loc)
      : ctx(nullptr, LongestPathAnalysisOptions(false, true, false)), loc(loc) {
    mlir::OpBuilder builder(loc->getContext());
    moduleOp = mlir::ModuleOp::create(builder, loc);
    emptyName = StringAttr::get(loc->getContext(), "");
  }

  // Initialize the pass pipeline used to lower operations to AIG.
  // This sets up the standard conversion sequence: HW -> Comb -> AIG -> cleanup
  LogicalResult initializePipeline();

  // Analyze timing paths for a specific operation result and bit position.
  // Creates a wrapper module if needed, runs conversion, and returns timing
  // information as input dependencies.
  // Results are tuples of (input operand index, bit position in operand, delay)
  LogicalResult analyzeOperation(
      OpResult value, size_t bitPos,
      SmallVectorImpl<std::tuple<size_t, size_t, int64_t>> &results);

private:
  // Get or create a LocalVisitor for analyzing the given operation.
  // Uses caching based on operation name and function signature to avoid
  // recomputing analysis for identical operations.
  FailureOr<LocalVisitor *> getOrComputeLocalVisitor(Operation *op);

  // Extract the function signature (input/output types) from an operation.
  // This is used as part of the cache key to distinguish operations with
  // different type signatures.
  static mlir::FunctionType getFunctionTypeForOp(Operation *op);

  // Cache mapping (operation_name, function_type) -> analyzed LocalVisitor
  // This avoids recomputing analysis for operations with identical signatures
  llvm::DenseMap<std::pair<mlir::OperationName, mlir::FunctionType>,
                 std::unique_ptr<LocalVisitor>>
      cache;

  // Standard conversion pipeline: HW -> Comb -> AIG -> cleanup passes
  // This pipeline is applied to wrapper modules to prepare them for analysis
  constexpr static StringRef pipelineStr =
      "hw.module(hw-aggregate-to-comb,convert-comb-to-synth,cse,canonicalize)";
  std::unique_ptr<mlir::PassManager> passManager;

  // Temporary module used to hold wrapper modules during analysis
  // Each operation gets its own wrapper module created inside this parent
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp;

  Context ctx;  // Analysis context and configuration
  Location loc; // Source location for error reporting
  StringAttr emptyName;
};

mlir::FunctionType OperationAnalyzer::getFunctionTypeForOp(Operation *op) {
  return mlir::FunctionType::get(op->getContext(), op->getOperandTypes(),
                                 op->getResultTypes());
}

// -----------------------------------------------------------------------------
// LocalVisitor
// -----------------------------------------------------------------------------

class LocalVisitor {
public:
  LocalVisitor(hw::HWModuleOp module, Context *ctx);
  LogicalResult initializeAndRun();
  // Wait until the thread is done.
  void waitUntilDone() const;

  // Get the longest paths for the given value and bit position.
  // If the result is not cached, compute it and cache it.
  FailureOr<ArrayRef<OpenPath>> getOrComputePaths(Value value, size_t bitPos);

  // Get the longest paths for the given value and bit position. This returns
  // an empty array if not pre-computed.
  ArrayRef<OpenPath> getCachedPaths(Value value, size_t bitPos) const;

  // A map from the object to the maximum distance and history.
  using ObjectToMaxDistance =
      llvm::MapVector<Object,
                      std::pair<int64_t, llvm::ImmutableList<DebugPoint>>>;

  void getInternalPaths(SmallVectorImpl<DataflowPath> &results) const;
  void setTopLevel() { this->topLevel = true; }
  bool isTopLevel() const { return topLevel; }
  hw::HWModuleOp getHWModuleOp() const { return module; }

  const auto &getFromInputPortToEndPoint() const {
    return fromInputPortToEndPoint;
  }
  const auto &getFromOutputPortToStartPoint() const {
    return fromOutputPortToStartPoint;
  }
  const auto &getEndPointResults() const { return endPointResults; }

  circt::igraph::InstancePathCache *getInstancePathCache() const {
    return instancePathCache.get();
  }

  llvm::ImmutableListFactory<DebugPoint> *getDebugPointFactory() const {
    return debugPointFactory.get();
  }

private:
  void putUnclosedResult(const Object &object, int64_t delay,
                         llvm::ImmutableList<DebugPoint> history,
                         ObjectToMaxDistance &objectToMaxDistance);

  // A map from the input port to the farthest end point.
  llvm::MapVector<std::pair<BlockArgument, size_t>, ObjectToMaxDistance>
      fromInputPortToEndPoint;

  // A map from the output port to the farthest start point.
  llvm::MapVector<std::tuple<size_t, size_t>, ObjectToMaxDistance>
      fromOutputPortToStartPoint;

  LogicalResult initializeAndRun(hw::InstanceOp instance);
  LogicalResult initializeAndRun(hw::OutputOp output);

  // --------------------------------------------------------------------------
  // Visitors
  // --------------------------------------------------------------------------
  LogicalResult visitValue(Value value, size_t bitPos,
                           SmallVectorImpl<OpenPath> &results);
  // Module boundary.
  LogicalResult visit(mlir::BlockArgument argument, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(hw::InstanceOp op, size_t bitPos, size_t resultNum,
                      SmallVectorImpl<OpenPath> &results);

  // Data-movement ops.
  LogicalResult visit(hw::WireOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::ConcatOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::ExtractOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::ReplicateOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  // Track equivalence classes for data movement ops
  // Treat output as equivalent to input for pure data movement operations
  llvm::EquivalenceClasses<std::pair<Value, size_t>> ec;
  DenseMap<std::pair<Value, size_t>, std::pair<Value, size_t>> ecMap;
  std::pair<Value, size_t> findLeader(Value value, size_t bitpos) const {
    return ec.getLeaderValue({value, bitpos});
  }
  LogicalResult markEquivalent(Value from, size_t fromBitPos, Value to,
                               size_t toBitPos,
                               SmallVectorImpl<OpenPath> &results);

  // Bit-logical ops.
  LogicalResult visit(aig::AndInverterOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(mig::MajorityInverterOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::AndOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::XorOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::OrOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::MuxOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);
  LogicalResult addLogicOp(Operation *op, size_t bitPos,
                           SmallVectorImpl<OpenPath> &results);
  LogicalResult visit(comb::TruthTableOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results);

  // Constants.
  LogicalResult visit(hw::ConstantOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return success();
  }

  // Registers are start point.
  LogicalResult visit(seq::FirRegOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markStartPoint(op, bitPos, results);
  }

  LogicalResult visit(seq::CompRegOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markStartPoint(op, bitPos, results);
  }

  LogicalResult visit(seq::FirMemReadOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markStartPoint(op, bitPos, results);
  }

  LogicalResult visit(seq::FirMemReadWriteOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markStartPoint(op, bitPos, results);
  }

  LogicalResult visitDefault(OpResult result, size_t bitPos,
                             SmallVectorImpl<OpenPath> &results);

  // Helper functions.
  LogicalResult addEdge(Value to, size_t toBitPos, int64_t delay,
                        SmallVectorImpl<OpenPath> &results);
  LogicalResult markStartPoint(Value value, size_t bitPos,
                               SmallVectorImpl<OpenPath> &results);
  LogicalResult markRegEndPoint(Value endPoint, Value start, Value reset = {},
                                Value resetValue = {}, Value enable = {});

  // The module we are visiting.
  hw::HWModuleOp module;
  Context *ctx;

  // Thread-local data structures.
  std::unique_ptr<circt::igraph::InstancePathCache> instancePathCache;
  // TODO: Make debug points optional.
  std::unique_ptr<llvm::ImmutableListFactory<DebugPoint>> debugPointFactory;

  // A map from the value point to the longest paths.
  DenseMap<std::pair<Value, size_t>, std::unique_ptr<SmallVector<OpenPath>>>
      cachedResults;

  // A map from the object to the longest paths.
  DenseMap<Object, SmallVector<OpenPath>> endPointResults;

  // The operation analyzer for comb/hw operations remaining in the circuit.
  std::unique_ptr<OperationAnalyzer> operationAnalyzer;

  // This is set to true when the thread is done.
  std::atomic_bool done;
  mutable std::condition_variable cv;
  mutable std::mutex mutex;

  // A flag to indicate the module is top-level.
  bool topLevel = false;
};

LocalVisitor::LocalVisitor(hw::HWModuleOp module, Context *ctx)
    : module(module), ctx(ctx) {
  debugPointFactory =
      std::make_unique<llvm::ImmutableListFactory<DebugPoint>>();
  instancePathCache = ctx->instanceGraph
                          ? std::make_unique<circt::igraph::InstancePathCache>(
                                *ctx->instanceGraph)
                          : nullptr;
  operationAnalyzer = std::make_unique<OperationAnalyzer>(module->getLoc());

  done = false;
}

ArrayRef<OpenPath> LocalVisitor::getCachedPaths(Value value,
                                                size_t bitPos) const {
  std::pair<Value, size_t> valueAndBitPos(value, bitPos);
  auto leader = ec.findLeader(valueAndBitPos);
  if (leader != ec.member_end()) {
    if (*leader != valueAndBitPos) {
      // If this is not the leader, then use the leader.
      return getCachedPaths(leader->first, leader->second);
    }
  }

  auto it = cachedResults.find(valueAndBitPos);
  // If not found, then consider it to be a constant.
  if (it == cachedResults.end())
    return {};
  return *it->second;
}

void LocalVisitor::putUnclosedResult(const Object &object, int64_t delay,
                                     llvm::ImmutableList<DebugPoint> history,
                                     ObjectToMaxDistance &objectToMaxDistance) {
  auto &slot = objectToMaxDistance[object];
  if (slot.first >= delay && delay != 0)
    return;
  slot = {delay, history};
}

void LocalVisitor::waitUntilDone() const {
  // Wait for the thread to finish.
  std::unique_lock<std::mutex> lock(mutex);
  cv.wait(lock, [this] { return done.load(); });
}

LogicalResult LocalVisitor::markRegEndPoint(Value endPoint, Value start,
                                            Value reset, Value resetValue,
                                            Value enable) {
  auto bitWidth = getBitWidth(endPoint);
  auto record = [&](size_t endPointBitPos, Value value, size_t bitPos) {
    auto result = getOrComputePaths(value, bitPos);
    if (failed(result))
      return failure();
    for (auto &path : *result) {
      if (auto blockArg = dyn_cast<BlockArgument>(path.startPoint.value)) {
        // Not closed.
        putUnclosedResult(
            {{}, endPoint, endPointBitPos}, path.delay, path.history,
            fromInputPortToEndPoint[{blockArg, path.startPoint.bitPos}]);
      } else {
        // If the start point is not a port, record to the results.
        endPointResults[{{}, endPoint, endPointBitPos}].push_back(path);
      }
    }
    return success();
  };

  // Get paths for each bit, and record them.
  for (size_t i = 0, e = bitWidth; i < e; ++i) {
    // Call getOrComputePaths to make sure the paths are computed for endPoint.
    // This avoids a race condition.
    if (failed(getOrComputePaths(endPoint, i)))
      return failure();
    if (failed(record(i, start, i)))
      return failure();
  }

  // Edge from a reg to reset/resetValue.
  if (reset)
    for (size_t i = 0, e = bitWidth; i < e; ++i) {
      if (failed(record(i, reset, 0)) || failed(record(i, resetValue, i)))
        return failure();
    }

  // Edge from a reg to enable signal.
  if (enable)
    for (size_t i = 0, e = bitWidth; i < e; ++i) {
      if (failed(record(i, enable, 0)))
        return failure();
    }

  return success();
}

LogicalResult LocalVisitor::markEquivalent(Value from, size_t fromBitPos,
                                           Value to, size_t toBitPos,
                                           SmallVectorImpl<OpenPath> &results) {
  [[maybe_unused]] auto leader = ec.getOrInsertLeaderValue({to, toBitPos});
  // Merge classes, and visit the leader.
  [[maybe_unused]] auto newLeader =
      ec.unionSets({to, toBitPos}, {from, fromBitPos});
  assert(leader == *newLeader);
  return visitValue(to, toBitPos, results);
}

LogicalResult LocalVisitor::addEdge(Value to, size_t bitPos, int64_t delay,
                                    SmallVectorImpl<OpenPath> &results) {
  auto result = getOrComputePaths(to, bitPos);
  if (failed(result))
    return failure();
  for (auto &path : *result) {
    auto newPath = path;
    newPath.delay += delay;
    results.push_back(newPath);
  }
  return success();
}

LogicalResult LocalVisitor::visit(aig::AndInverterOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {

  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(mig::MajorityInverterOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  // Use a 3-input majority inverter as the basic unit.
  // An n-input majority inverter requires n/2 stages of 3-input gates.
  size_t depth = op.getInputs().size() / 2;
  for (auto input : op.getInputs()) {
    if (failed(addEdge(input, bitPos, depth, results)))
      return failure();
  }
  return success();
}

LogicalResult LocalVisitor::visit(comb::AndOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(comb::OrOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(comb::XorOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(comb::MuxOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  // Add a cost of 1 for the mux.
  if (failed(addEdge(op.getCond(), 0, 1, results)) ||
      failed(addEdge(op.getTrueValue(), bitPos, 1, results)) ||
      failed(addEdge(op.getFalseValue(), bitPos, 1, results)))
    return failure();
  filterPaths(results, ctx->doKeepOnlyMaxDelayPaths(), ctx->isLocalScope());
  return success();
}

LogicalResult LocalVisitor::visit(comb::TruthTableOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  for (auto input : op.getInputs()) {
    if (failed(addEdge(input, 0, 1, results)))
      return failure();
  }
  return success();
}

LogicalResult LocalVisitor::visit(comb::ExtractOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  assert(getBitWidth(op.getInput()) > bitPos + op.getLowBit());
  auto result = markEquivalent(op, bitPos, op.getInput(),
                               bitPos + op.getLowBit(), results);
  return result;
}

LogicalResult LocalVisitor::visit(comb::ReplicateOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  return markEquivalent(op, bitPos, op.getInput(),
                        bitPos % getBitWidth(op.getInput()), results);
}

LogicalResult LocalVisitor::markStartPoint(Value value, size_t bitPos,
                                           SmallVectorImpl<OpenPath> &results) {
  results.emplace_back(circt::igraph::InstancePath(), value, bitPos);
  return success();
}

LogicalResult LocalVisitor::visit(hw::WireOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  return markEquivalent(op, bitPos, op.getInput(), bitPos, results);
}

LogicalResult LocalVisitor::visit(hw::InstanceOp op, size_t bitPos,
                                  size_t resultNum,
                                  SmallVectorImpl<OpenPath> &results) {
  auto moduleName = op.getReferencedModuleNameAttr();
  auto value = op->getResult(resultNum);

  // If an instance graph is not available, we treat instance results as a
  // start point.
  if (!ctx->instanceGraph)
    return markStartPoint(value, bitPos, results);

  // If an instance graph is available, then we can look up the module.
  auto *node = ctx->instanceGraph->lookup(moduleName);
  assert(node && "module not found");

  // Otherwise, if the module is not a HWModuleOp, then we should treat it as a
  // start point.
  if (!isa<hw::HWModuleOp>(node->getModule()))
    return markStartPoint(value, bitPos, results);

  auto *localVisitor = ctx->getLocalVisitorMutable(moduleName);

  auto module = localVisitor->getHWModuleOp();
  auto operand = module.getBodyBlock()->getTerminator()->getOperand(resultNum);
  auto result = localVisitor->getOrComputePaths(operand, bitPos);
  if (failed(result))
    return failure();

  for (auto &path : *result) {
    auto delay = path.delay;
    auto history = path.history;
    auto newPath =
        instancePathCache->prependInstance(op, path.startPoint.instancePath);
    auto startPointPoint = path.startPoint;
    // If the start point is not a block argument, record it directly.
    auto arg = dyn_cast<BlockArgument>(startPointPoint.value);
    if (!arg) {
      // Update the history to have correct instance path.
      auto newHistory = debugPointFactory->getEmptyList();
      if (ctx->doTraceDebugPoints()) {
        newHistory =
            mapList(debugPointFactory.get(), history, [&](DebugPoint p) {
              p.object.instancePath =
                  instancePathCache->prependInstance(op, p.object.instancePath);
              return p;
            });
        newHistory = debugPointFactory->add(
            DebugPoint({}, value, bitPos, delay, "output port"), newHistory);
      }

      results.emplace_back(newPath, startPointPoint.value,
                           startPointPoint.bitPos, delay, newHistory);
      continue;
    }

    // Otherwise, we need to look up the instance operand.
    auto result = getOrComputePaths(op->getOperand(arg.getArgNumber()),
                                    startPointPoint.bitPos);
    if (failed(result))
      return failure();
    for (auto path : *result) {
      auto newHistory = debugPointFactory->getEmptyList();
      if (ctx->doTraceDebugPoints()) {
        // Update the history to have correct instance path.
        newHistory =
            mapList(debugPointFactory.get(), history, [&](DebugPoint p) {
              p.object.instancePath =
                  instancePathCache->prependInstance(op, p.object.instancePath);
              p.delay += path.delay;
              return p;
            });
        DebugPoint debugPoint({}, value, bitPos, delay + path.delay,
                              "output port");
        newHistory = debugPointFactory->add(debugPoint, newHistory);
      }

      path.delay += delay;
      path.history =
          concatList(debugPointFactory.get(), newHistory, path.history);
      results.push_back(path);
    }
  }
  return success();
}

LogicalResult LocalVisitor::visit(comb::ConcatOp op, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  // Find the exact bit pos in the concat.
  size_t newBitPos = bitPos;
  for (auto operand : llvm::reverse(op.getInputs())) {
    auto size = getBitWidth(operand);
    if (newBitPos >= size) {
      newBitPos -= size;
      continue;
    }
    return markEquivalent(op, bitPos, operand, newBitPos, results);
  }

  llvm::report_fatal_error("Should not reach here");
  return failure();
}

LogicalResult LocalVisitor::addLogicOp(Operation *op, size_t bitPos,
                                       SmallVectorImpl<OpenPath> &results) {
  auto size = op->getNumOperands();
  auto cost = llvm::Log2_64_Ceil(size);
  // Create edges each operand with cost ceil(log(size)).
  for (auto operand : op->getOperands())
    if (failed(addEdge(operand, bitPos, cost, results)))
      return failure();
  filterPaths(results, ctx->doKeepOnlyMaxDelayPaths(), ctx->isLocalScope());
  return success();
}

LogicalResult LocalVisitor::visitDefault(OpResult value, size_t bitPos,
                                         SmallVectorImpl<OpenPath> &results) {
  if (!isa_and_nonnull<hw::HWDialect, comb::CombDialect>(
          value.getDefiningOp()->getDialect()))
    return success();
  // Query it to an operation analyzer.
  LLVM_DEBUG({
    llvm::dbgs() << "Visiting default: ";
    llvm::dbgs() << " " << value << "[" << bitPos << "]\n";
  });
  SmallVector<std::tuple<size_t, size_t, int64_t>> oracleResults;
  auto paths =
      operationAnalyzer->analyzeOperation(value, bitPos, oracleResults);
  if (failed(paths)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Failed to get results for: " << value << "[" << bitPos
                   << "]\n";
    });
    return success();
  }
  auto *op = value.getDefiningOp();
  for (auto [inputPortIndex, startPointBitPos, delay] : oracleResults) {
    LLVM_DEBUG({
      llvm::dbgs() << "Adding edge: " << value << "[" << bitPos << "] -> "
                   << op->getOperand(inputPortIndex) << "[" << startPointBitPos
                   << "] with delay " << delay << "\n";
    });
    if (failed(addEdge(op->getOperand(inputPortIndex), startPointBitPos, delay,
                       results)))
      return failure();
  }
  return success();
}

LogicalResult LocalVisitor::visit(mlir::BlockArgument arg, size_t bitPos,
                                  SmallVectorImpl<OpenPath> &results) {
  assert(arg.getOwner() == module.getBodyBlock());

  // Record a debug point.
  auto newHistory = ctx->doTraceDebugPoints()
                        ? debugPointFactory->add(
                              DebugPoint({}, arg, bitPos, 0, "input port"), {})
                        : debugPointFactory->getEmptyList();
  OpenPath newPoint({}, arg, bitPos, 0, newHistory);
  results.push_back(newPoint);
  return success();
}

FailureOr<ArrayRef<OpenPath>> LocalVisitor::getOrComputePaths(Value value,
                                                              size_t bitPos) {
  if (auto *op = value.getDefiningOp())
    if (op->hasTrait<OpTrait::ConstantLike>())
      return ArrayRef<OpenPath>{};

  if (ec.contains({value, bitPos})) {
    auto leader = ec.findLeader({value, bitPos});
    // If this is not the leader, then use the leader.
    if (*leader != std::pair(value, bitPos)) {
      return getOrComputePaths(leader->first, leader->second);
    }
  }

  auto it = cachedResults.find({value, bitPos});
  if (it != cachedResults.end())
    return ArrayRef<OpenPath>(*it->second);

  auto results = std::make_unique<SmallVector<OpenPath>>();
  if (failed(visitValue(value, bitPos, *results)))
    return {};

  // Unique the results.
  filterPaths(*results, ctx->doKeepOnlyMaxDelayPaths(), ctx->isLocalScope());
  LLVM_DEBUG({
    llvm::dbgs() << value << "[" << bitPos << "] "
                 << "Found " << results->size() << " paths\n";
    llvm::dbgs() << "====Paths:\n";
    for (auto &path : *results) {
      path.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "====\n";
  });

  auto insertedResult =
      cachedResults.try_emplace({value, bitPos}, std::move(results));
  assert(insertedResult.second);
  return ArrayRef<OpenPath>(*insertedResult.first->second);
}

LogicalResult LocalVisitor::visitValue(Value value, size_t bitPos,
                                       SmallVectorImpl<OpenPath> &results) {
  LLVM_DEBUG({
    llvm::dbgs() << "Visiting: ";
    llvm::dbgs() << " " << value << "[" << bitPos << "]\n";
  });

  if (auto blockArg = dyn_cast<mlir::BlockArgument>(value))
    return visit(blockArg, bitPos, results);

  auto *op = value.getDefiningOp();

  if (op->hasTrait<OpTrait::ConstantLike>())
    return success();

  auto result =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp,
                aig::AndInverterOp, mig::MajorityInverterOp, comb::AndOp,
                comb::OrOp, comb::MuxOp, comb::XorOp, comb::TruthTableOp,
                seq::FirRegOp, seq::CompRegOp, seq::FirMemReadOp,
                seq::FirMemReadWriteOp, hw::WireOp>([&](auto op) {
            size_t idx = results.size();
            auto result = visit(op, bitPos, results);
            if (ctx->doTraceDebugPoints())
              if (auto name =
                      op->template getAttrOfType<StringAttr>("sv.namehint")) {

                for (auto i = idx, e = results.size(); i < e; ++i) {
                  DebugPoint debugPoint({}, value, bitPos, results[i].delay,
                                        "namehint");
                  auto newHistory =
                      debugPointFactory->add(debugPoint, results[i].history);
                  results[i].history = newHistory;
                }
              }
            return result;
          })
          .Case<hw::InstanceOp>([&](hw::InstanceOp op) {
            return visit(op, bitPos, cast<OpResult>(value).getResultNumber(),
                         results);
          })
          .Default([&](auto op) {
            return visitDefault(cast<OpResult>(value), bitPos, results);
          });
  return result;
}

LogicalResult LocalVisitor::initializeAndRun(hw::InstanceOp instance) {
  const auto *childVisitor =
      ctx->getLocalVisitorMutable(instance.getReferencedModuleNameAttr());
  // If not found, the module is blackbox so skip it.
  if (!childVisitor)
    return success();

  // Connect dataflow from instance input ports to end point in the child.
  for (const auto &[object, openPaths] :
       childVisitor->getFromInputPortToEndPoint()) {
    auto [arg, argBitPos] = object;
    for (auto [point, delayAndHistory] : openPaths) {
      auto [instancePath, endPoint, endPointBitPos] = point;
      auto [delay, history] = delayAndHistory;
      // Prepend the instance path.
      assert(instancePathCache);
      auto newPath = instancePathCache->prependInstance(instance, instancePath);
      auto computedResults =
          getOrComputePaths(instance.getOperand(arg.getArgNumber()), argBitPos);
      if (failed(computedResults))
        return failure();

      for (auto &result : *computedResults) {
        auto newHistory = ctx->doTraceDebugPoints()
                              ? mapList(debugPointFactory.get(), history,
                                        [&](DebugPoint p) {
                                          // Update the instance path to
                                          // prepend the current instance.
                                          p.object.instancePath = newPath;
                                          p.delay += result.delay;
                                          return p;
                                        })
                              : debugPointFactory->getEmptyList();
        if (auto newPort = dyn_cast<BlockArgument>(result.startPoint.value)) {
          putUnclosedResult(
              {newPath, endPoint, endPointBitPos}, result.delay + delay,
              newHistory,
              fromInputPortToEndPoint[{newPort, result.startPoint.bitPos}]);
        } else {
          endPointResults[{newPath, endPoint, endPointBitPos}].emplace_back(
              result.startPoint.instancePath, result.startPoint.value,
              result.startPoint.bitPos, result.delay + delay,
              ctx->doTraceDebugPoints() ? concatList(debugPointFactory.get(),
                                                     newHistory, result.history)
                                        : debugPointFactory->getEmptyList());
        }
      }
    }
  }

  // Pre-compute the results for the instance output ports.
  for (auto instance : instance->getResults()) {
    for (size_t i = 0, e = getBitWidth(instance); i < e; ++i) {
      auto computedResults = getOrComputePaths(instance, i);
      if (failed(computedResults))
        return failure();
    }
  }
  return success();
}

LogicalResult LocalVisitor::initializeAndRun(hw::OutputOp output) {
  for (OpOperand &operand : output->getOpOperands()) {
    for (size_t i = 0, e = getBitWidth(operand.get()); i < e; ++i) {
      auto &recordOutput =
          fromOutputPortToStartPoint[{operand.getOperandNumber(), i}];
      auto computedResults = getOrComputePaths(operand.get(), i);
      if (failed(computedResults))
        return failure();
      for (const auto &result : *computedResults) {
        putUnclosedResult(result.startPoint, result.delay, result.history,
                          recordOutput);
      }
    }
  }
  return success();
}

LogicalResult LocalVisitor::initializeAndRun() {
  LLVM_DEBUG({ ctx->notifyStart(module.getModuleNameAttr()); });
  if (ctx->doLazyComputation())
    return success();

  // Initialize the results for the block arguments.
  for (auto blockArgument : module.getBodyBlock()->getArguments())
    for (size_t i = 0, e = getBitWidth(blockArgument); i < e; ++i)
      (void)getOrComputePaths(blockArgument, i);

  auto walkResult = module->walk([&](Operation *op) {
    auto result =
        mlir::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<seq::FirRegOp>([&](seq::FirRegOp op) {
              return markRegEndPoint(op, op.getNext(), op.getReset(),
                                     op.getResetValue());
            })
            .Case<seq::CompRegOp>([&](auto op) {
              return markRegEndPoint(op, op.getInput(), op.getReset(),
                                     op.getResetValue());
            })
            .Case<seq::FirMemWriteOp>([&](auto op) {
              // TODO: Add address.
              return markRegEndPoint(op.getMemory(), op.getData(), {}, {},
                                     op.getEnable());
            })
            .Case<seq::FirMemReadWriteOp>([&](seq::FirMemReadWriteOp op) {
              // TODO: Add address.
              return markRegEndPoint(op.getMemory(), op.getWriteData(), {}, {},
                                     op.getEnable());
            })
            .Case<aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp,
                  comb::MuxOp, seq::FirMemReadOp>([&](auto op) {
              // NOTE: Visiting and-inverter is not necessary but
              // useful to reduce recursion depth.
              for (size_t i = 0, e = getBitWidth(op); i < e; ++i)
                if (failed(getOrComputePaths(op, i)))
                  return failure();
              return success();
            })
            .Case<hw::InstanceOp, hw::OutputOp>(
                [&](auto op) { return initializeAndRun(op); })
            .Default([](auto op) { return success(); });
    if (failed(result))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  {
    std::lock_guard<std::mutex> lock(mutex);
    done.store(true);
    cv.notify_all();
  }
  LLVM_DEBUG({ ctx->notifyEnd(module.getModuleNameAttr()); });
  return failure(walkResult.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

const LocalVisitor *Context::getLocalVisitor(StringAttr name) const {
  return getLocalVisitorMutable(name);
}

LocalVisitor *Context::getLocalVisitorMutable(StringAttr name) const {
  auto *it = localVisitors.find(name);
  if (it == localVisitors.end())
    return nullptr;

  if (isRunningParallel())
    it->second->waitUntilDone();

  // NOTE: Don't call waitUntilDone here.
  return it->second.get();
}

// ===----------------------------------------------------------------------===//
// OperationAnalyzer
// ===----------------------------------------------------------------------===//

FailureOr<LocalVisitor *>
OperationAnalyzer::getOrComputeLocalVisitor(Operation *op) {
  // Check cache first.
  auto opName = op->getName();
  auto functionType = getFunctionTypeForOp(op);
  auto key = std::make_pair(opName, functionType);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second.get();

  SmallVector<hw::PortInfo> ports;
  // Helper to convert types into integer types.
  auto getType = [&](Type type) -> Type {
    if (type.isInteger())
      return type;
    auto bitWidth = hw::getBitWidth(type);
    if (bitWidth < 0)
      return Type(); // Unsupported type
    return IntegerType::get(op->getContext(), bitWidth);
  };

  // Helper to add a port to the module definition
  auto addPort = [&](Type type, hw::ModulePort::Direction dir) {
    hw::PortInfo portInfo;
    portInfo.dir = dir;
    portInfo.name = emptyName;
    portInfo.type = type;
    ports.push_back(portInfo);
  };

  // Create input ports for each operand
  for (auto input : op->getOperands()) {
    auto type = getType(input.getType());
    if (!type)
      return failure(); // Unsupported operand type
    addPort(type, hw::ModulePort::Direction::Input);
  }

  // Create output ports for each result.
  SmallVector<Type> resultsTypes;
  for (Value result : op->getResults()) {
    auto type = getType(result.getType());
    if (!type)
      return failure(); // Unsupported result type
    addPort(type, hw::ModulePort::Direction::Output);
    resultsTypes.push_back(type);
  }

  // Build the wrapper HW module
  OpBuilder builder(op->getContext());
  builder.setInsertionPointToEnd(moduleOp->getBody());

  // Generate unique module name.
  auto moduleName = builder.getStringAttr("module_" + Twine(cache.size()));
  hw::HWModuleOp hwModule =
      hw::HWModuleOp::create(builder, op->getLoc(), moduleName, ports);

  // Clone the operation inside the wrapper module
  builder.setInsertionPointToStart(hwModule.getBodyBlock());
  auto *cloned = builder.clone(*op);

  // Connect module inputs to cloned operation operands
  // Handle type mismatches with bitcast operations
  // Type mismatches can occur when the original operation uses non-integer
  // types (e.g., structs, arrays) that get converted to integer types for the
  // wrapper module ports. Since we use the same bit width via
  // hw::getBitWidth(), bitcast is safe for bit-compatible types.
  for (auto arg : hwModule.getBodyBlock()->getArguments()) {
    Value input = arg;
    auto idx = arg.getArgNumber();

    // Insert bitcast if input port type differs from operand type
    if (input.getType() != cloned->getOperand(idx).getType())
      input = hw::BitcastOp::create(builder, op->getLoc(),
                                    cloned->getOperand(idx).getType(), input);

    cloned->setOperand(idx, input);
  }

  // Connect cloned operation results to module outputs
  // Handle type mismatches with bitcast operations
  SmallVector<Value> outputs;
  for (auto result : cloned->getResults()) {
    auto idx = result.getResultNumber();

    // Insert bitcast if result type differs from output port type
    if (result.getType() != resultsTypes[idx])
      result = hw::BitcastOp::create(builder, op->getLoc(), resultsTypes[idx],
                                     result)
                   ->getResult(0);

    outputs.push_back(result);
  }

  hwModule.getBodyBlock()->getTerminator()->setOperands(outputs);

  // Run conversion pipeline (HW -> Comb -> AIG -> cleanup)
  if (!passManager && failed(initializePipeline()))
    return mlir::emitError(loc)
           << "Failed to initialize pipeline, possibly passes used in the "
              "analysis are not registered";

  if (failed(passManager->run(moduleOp->getOperation())))
    return mlir::emitError(loc) << "Failed to run lowering pipeline";

  // Create LocalVisitor to analyze the converted AIG module
  auto localVisitor = std::make_unique<LocalVisitor>(hwModule, &ctx);
  if (failed(localVisitor->initializeAndRun()))
    return failure();

  // Cache the result and return
  auto [iterator, inserted] = cache.insert({key, std::move(localVisitor)});
  assert(inserted && "Cache insertion must succeed for new key");
  return iterator->second.get();
}

LogicalResult OperationAnalyzer::analyzeOperation(
    OpResult value, size_t bitPos,
    SmallVectorImpl<std::tuple<size_t, size_t, int64_t>> &results) {
  auto *op = value.getDefiningOp();
  auto localVisitorResult = getOrComputeLocalVisitor(op);
  if (failed(localVisitorResult))
    return failure();

  auto *localVisitor = *localVisitorResult;

  // Get output.
  Value operand =
      localVisitor->getHWModuleOp().getBodyBlock()->getTerminator()->getOperand(
          value.getResultNumber());
  auto openPaths = localVisitor->getOrComputePaths(operand, bitPos);
  if (failed(openPaths))
    return failure();

  results.reserve(openPaths->size() + results.size());
  for (auto &path : *openPaths) {
    // end point is always a block argument since there is no other value that
    // could be a start point.
    BlockArgument blockArg = cast<BlockArgument>(path.startPoint.value);
    auto inputPortIndex = blockArg.getArgNumber();
    results.push_back(
        std::make_tuple(inputPortIndex, path.startPoint.bitPos, path.delay));
  }

  return success();
}

LogicalResult OperationAnalyzer::initializePipeline() {
  passManager = std::make_unique<mlir::PassManager>(loc->getContext());
  return parsePassPipeline(pipelineStr, *passManager);
}

//===----------------------------------------------------------------------===//
// LongestPathAnalysis::Impl
//===----------------------------------------------------------------------===//

/// Internal implementation for LongestPathAnalysis.
///
/// This class owns per-module LocalVisitors, orchestrates initialization over
/// the instance graph, and provides the concrete implementations for the
/// public LongestPathAnalysis API.
struct LongestPathAnalysis::Impl {
  Impl(Operation *module, mlir::AnalysisManager &am,
       const LongestPathAnalysisOptions &option);

  /// Initialize and run analysis for a full MLIR module (hierarchical).
  LogicalResult initializeAndRun(mlir::ModuleOp module);

  /// Initialize and run analysis for a single HW module (local scope).
  LogicalResult initializeAndRun(hw::HWModuleOp module);

  /// Return true if we have a LocalVisitor for the given HW module.
  bool isAnalysisAvailable(StringAttr moduleName) const;

  /// Compute hierarchical timing paths to (value, bitPos) and append to
  /// results.
  LogicalResult computeGlobalPaths(Value value, size_t bitPos,
                                   SmallVectorImpl<DataflowPath> &results);

  /// Collect register-to-register (closed) paths within the module. When
  /// 'elaborate' is true, paths are elaborated with instance paths from
  /// `moduleName`.
  template <bool elaborate>
  LogicalResult
  collectClosedPaths(StringAttr moduleName,
                     SmallVectorImpl<DataflowPath> &results) const;

  /// Collect open paths from module input ports to internal sequential sinks.
  LogicalResult
  collectInputToInternalPaths(StringAttr moduleName,
                              SmallVectorImpl<DataflowPath> &results) const;

  /// Collect open paths from internal sequential sources to module output
  /// ports.
  LogicalResult
  collectInternalToOutputPaths(StringAttr moduleName,
                               SmallVectorImpl<DataflowPath> &results) const;

  /// Top modules inferred or specified for this analysis run.
  llvm::ArrayRef<hw::HWModuleOp> getTopModules() const { return topModules; }

  /// Return average of per-bit max delays for a value.
  FailureOr<int64_t> getAverageMaxDelay(Value value);

  /// Return the max delay for a value.
  FailureOr<int64_t> getMaxDelay(Value value, int64_t bitPos);

  /// Compute local open paths to (value, bitPos).
  FailureOr<ArrayRef<OpenPath>> computeLocalPaths(Value value, size_t bitPos);

protected:
  friend class IncrementalLongestPathAnalysis;

private:
  /// Recursive helper for computeGlobalPaths.
  LogicalResult computeGlobalPaths(const Object &originalObject, Value value,
                                   size_t bitPos,
                                   SmallVectorImpl<DataflowPath> &results);

  /// Analysis context.
  Context ctx;
  /// Top-level HW modules that seed hierarchical analysis.
  SmallVector<hw::HWModuleOp> topModules;
};

LogicalResult LongestPathAnalysis::Impl::computeGlobalPaths(
    Value value, size_t bitPos, SmallVectorImpl<DataflowPath> &results) {
  return computeGlobalPaths(Object({}, value, bitPos), value, bitPos, results);
}

LogicalResult LongestPathAnalysis::Impl::computeGlobalPaths(
    const Object &originalObject, Value value, size_t bitPos,
    SmallVectorImpl<DataflowPath> &results) {
  auto parentHWModule =
      value.getParentRegion()->getParentOfType<hw::HWModuleOp>();
  if (!parentHWModule)
    return mlir::emitError(value.getLoc())
           << "query value is not in a HWModuleOp";
  auto *localVisitor =
      ctx.getLocalVisitorMutable(parentHWModule.getModuleNameAttr());
  if (!localVisitor)
    return success();

  auto *instancePathCache = localVisitor->getInstancePathCache();
  size_t oldIndex = results.size();
  auto *node =
      ctx.instanceGraph
          ? ctx.instanceGraph->lookup(parentHWModule.getModuleNameAttr())
          : nullptr;
  LLVM_DEBUG({
    llvm::dbgs() << "Running " << parentHWModule.getModuleNameAttr() << " "
                 << value << " " << bitPos << "\n";
  });
  auto paths = localVisitor->getOrComputePaths(value, bitPos);
  if (failed(paths))
    return failure();

  for (auto &path : *paths) {
    auto arg = dyn_cast<BlockArgument>(path.startPoint.value);
    if (!arg || localVisitor->isTopLevel()) {
      // If the value is not a block argument, then we are done.
      results.push_back({originalObject, path, parentHWModule});
      continue;
    }

    auto newObject = originalObject;
    assert(node && "If an instance graph is not available, localVisitor must "
                   "be a toplevel");
    for (auto *inst : node->uses()) {
      auto startIndex = results.size();
      if (instancePathCache)
        newObject.instancePath = instancePathCache->appendInstance(
            originalObject.instancePath, inst->getInstance());

      auto result = computeGlobalPaths(
          newObject, inst->getInstance()->getOperand(arg.getArgNumber()),
          path.startPoint.bitPos, results);
      if (failed(result))
        return result;
      for (auto i = startIndex, e = results.size(); i < e; ++i)
        results[i].setDelay(results[i].getDelay() + path.delay);
    }
  }

  filterPaths(results, oldIndex);
  return success();
}

template <bool elaborate>
LogicalResult LongestPathAnalysis::Impl::collectClosedPaths(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  auto collectClosedPaths = [&](StringAttr name,
                                SmallVectorImpl<DataflowPath> &localResults,
                                igraph::InstanceGraphNode *top = nullptr) {
    if (!isAnalysisAvailable(name))
      return;
    auto *visitor = ctx.getLocalVisitorMutable(name);
    for (auto &[point, state] : visitor->getEndPointResults()) {
      for (const auto &dataFlow : state) {
        if constexpr (elaborate) {
          // If elaborate, we need to prepend the path to the root.
          auto *instancePathCache = visitor->getInstancePathCache();
          auto topToRoot = instancePathCache->getRelativePaths(
              visitor->getHWModuleOp(), top);
          for (auto &instancePath : topToRoot) {
            localResults.emplace_back(point, dataFlow,
                                      top->getModule<hw::HWModuleOp>());
            localResults.back().prependPaths(*visitor->getInstancePathCache(),
                                             visitor->getDebugPointFactory(),
                                             instancePath);
          }
        } else {
          localResults.emplace_back(point, dataFlow, visitor->getHWModuleOp());
        }
      }
    }
  };

  if (ctx.instanceGraph) {
    // Accumulate all closed results under the given module.
    auto *node = ctx.instanceGraph->lookup(moduleName);
    llvm::MapVector<StringAttr, SmallVector<DataflowPath>> resultsMap;
    // Prepare the results map for parallel processing.
    for (auto *child : llvm::post_order(node))
      resultsMap[child->getModule().getModuleNameAttr()] = {};

    mlir::parallelForEach(
        node->getModule().getContext(), resultsMap,
        [&](auto &it) { collectClosedPaths(it.first, it.second, node); });

    for (auto &[name, localResults] : resultsMap)
      results.append(localResults.begin(), localResults.end());
  } else {
    collectClosedPaths(moduleName, results);
  }

  return success();
}

LogicalResult LongestPathAnalysis::Impl::collectInputToInternalPaths(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  auto *visitor = ctx.getLocalVisitor(moduleName);
  if (!visitor)
    return failure();

  for (auto &[key, value] : visitor->getFromInputPortToEndPoint()) {
    auto [arg, argBitPos] = key;
    for (auto [point, delayAndHistory] : value) {
      auto [path, start, startBitPos] = point;
      auto [delay, history] = delayAndHistory;
      results.emplace_back(Object(path, start, startBitPos),
                           OpenPath({}, arg, argBitPos, delay, history),
                           visitor->getHWModuleOp());
    }
  }

  return success();
}

LogicalResult LongestPathAnalysis::Impl::collectInternalToOutputPaths(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  auto *visitor = ctx.getLocalVisitor(moduleName);
  if (!visitor)
    return failure();

  for (auto &[key, value] : visitor->getFromOutputPortToStartPoint()) {
    auto [resultNum, bitPos] = key;
    for (auto [point, delayAndHistory] : value) {
      auto [path, start, startBitPos] = point;
      auto [delay, history] = delayAndHistory;
      results.emplace_back(
          std::make_tuple(visitor->getHWModuleOp(), resultNum, bitPos),
          OpenPath(path, start, startBitPos, delay, history),
          visitor->getHWModuleOp());
    }
  }

  return success();
}

LongestPathAnalysis::Impl::Impl(Operation *moduleOp, mlir::AnalysisManager &am,
                                const LongestPathAnalysisOptions &option)
    : ctx(isa<mlir::ModuleOp>(moduleOp)
              ? &am.getAnalysis<igraph::InstanceGraph>()
              : nullptr,
          option) {
  if (auto module = dyn_cast<mlir::ModuleOp>(moduleOp)) {
    if (failed(initializeAndRun(module)))
      llvm::report_fatal_error("Failed to run longest path analysis");
  } else if (auto hwMod = dyn_cast<hw::HWModuleOp>(moduleOp)) {
    if (failed(initializeAndRun(hwMod)))
      llvm::report_fatal_error("Failed to run longest path analysis");
  } else {
    llvm::report_fatal_error("Analysis scheduled on invalid operation");
  }
}

LogicalResult
LongestPathAnalysis::Impl::initializeAndRun(hw::HWModuleOp module) {
  auto it =
      ctx.localVisitors.insert({module.getModuleNameAttr(),
                                std::make_unique<LocalVisitor>(module, &ctx)});
  assert(it.second);
  it.first->second->setTopLevel();
  return it.first->second->initializeAndRun();
}

LogicalResult
LongestPathAnalysis::Impl::initializeAndRun(mlir::ModuleOp module) {
  auto topNameAttr = ctx.getTopModuleName();
  topModules.clear();
  llvm::SetVector<Operation *> visited;
  auto *instanceGraph = ctx.instanceGraph;
  if (topNameAttr && topNameAttr.getValue() != "") {
    auto *topNode = instanceGraph->lookupOrNull(topNameAttr);
    if (!topNode || !topNode->getModule() ||
        !isa<hw::HWModuleOp>(topNode->getModule())) {
      module.emitError() << "top module not found in instance graph "
                         << topNameAttr;
      return failure();
    }
    topModules.push_back(topNode->getModule<hw::HWModuleOp>());
  } else {
    auto inferredResults = instanceGraph->getInferredTopLevelNodes();
    if (failed(inferredResults))
      return inferredResults;

    for (auto *node : *inferredResults) {
      if (auto top = dyn_cast<hw::HWModuleOp>(*node->getModule()))
        topModules.push_back(top);
    }
  }

  SmallVector<igraph::InstanceGraphNode *> worklist;
  for (auto topNode : topModules)
    worklist.push_back(instanceGraph->lookup(topNode.getModuleNameAttr()));
  // Get a set of design modules that are reachable from the top nodes,
  // excluding bound instances.
  while (!worklist.empty()) {
    auto *node = worklist.pop_back_val();
    assert(node && "node should not be null");
    auto op = node->getModule();
    if (!isa_and_nonnull<hw::HWModuleOp>(op) || !visited.insert(op))
      continue;

    for (auto *child : *node) {
      auto childOp = child->getInstance();
      if (!childOp || childOp->hasAttr("doNotPrint"))
        continue;

      worklist.push_back(child->getTarget());
    }
  }

  // Initialize the local visitors if the module was visited, in the
  // post-order of the instance graph.
  for (auto module : topModules) {
    auto *topNode = instanceGraph->lookup(module.getModuleNameAttr());
    for (auto *node : llvm::post_order(topNode))
      if (node && node->getModule())
        if (auto hwMod = dyn_cast<hw::HWModuleOp>(*node->getModule())) {
          if (visited.contains(hwMod))
            ctx.localVisitors.insert(
                {hwMod.getModuleNameAttr(),
                 std::make_unique<LocalVisitor>(hwMod, &ctx)});
        }

    ctx.localVisitors[topNode->getModule().getModuleNameAttr()]->setTopLevel();
  }

  return mlir::failableParallelForEach(
      module.getContext(), ctx.localVisitors,
      [&](auto &it) { return it.second->initializeAndRun(); });
}

bool LongestPathAnalysis::Impl::isAnalysisAvailable(
    StringAttr moduleName) const {
  return ctx.localVisitors.find(moduleName) != ctx.localVisitors.end();
}
// Return the average of the maximum delays across all bits of the given
// value, which is useful approximation for the delay of the value. For each
// bit position, finds all paths and takes the maximum delay. Then averages
// these maximum delays across all bits of the value.
FailureOr<int64_t> LongestPathAnalysis::Impl::getAverageMaxDelay(Value value) {
  SmallVector<DataflowPath> results;
  size_t bitWidth = getBitWidth(value);
  if (bitWidth == 0)
    return 0;
  int64_t totalDelay = 0;
  for (size_t i = 0; i < bitWidth; ++i) {
    // Clear results from previous iteration.
    results.clear();
    auto result = computeGlobalPaths(value, i, results);
    if (failed(result))
      return failure();

    int64_t maxDelay = getMaxDelayInPaths(ArrayRef<DataflowPath>(results));
    totalDelay += maxDelay;
  }
  return llvm::divideCeil(totalDelay, bitWidth);
}

FailureOr<int64_t> LongestPathAnalysis::Impl::getMaxDelay(Value value,
                                                          int64_t bitPos) {
  SmallVector<DataflowPath> results;
  auto collectAndFindMax = ([&](int64_t bitPos) -> FailureOr<int64_t> {
    results.clear();
    auto result = computeGlobalPaths(value, bitPos, results);
    if (failed(result))
      return failure();
    return getMaxDelayInPaths(ArrayRef<DataflowPath>(results));
  });
  if (bitPos >= 0)
    return collectAndFindMax(bitPos);

  size_t bitWidth = getBitWidth(value);
  if (bitWidth == 0)
    return 0;

  int64_t maxDelay = 0;
  for (size_t i = 0; i < bitWidth; ++i) {
    auto result = collectAndFindMax(i);
    if (failed(result))
      return failure();
    maxDelay = std::max(maxDelay, *result);
  }
  return maxDelay;
}

FailureOr<ArrayRef<OpenPath>>
LongestPathAnalysis::Impl::computeLocalPaths(Value value, size_t bitPos) {
  auto parentHWModule =
      value.getParentRegion()->getParentOfType<hw::HWModuleOp>();
  if (!parentHWModule)
    return mlir::emitError(value.getLoc())
           << "query value is not in a HWModuleOp";
  assert(ctx.localVisitors.size() == 1 &&
         "In incremental mode, there should be only one local visitor");

  auto *localVisitor =
      ctx.getLocalVisitorMutable(parentHWModule.getModuleNameAttr());
  if (!localVisitor)
    return mlir::emitError(value.getLoc())
           << "the local visitor for the given value does not exist";
  return localVisitor->getOrComputePaths(value, bitPos);
}

//===----------------------------------------------------------------------===//
// LongestPathAnalysis
//===----------------------------------------------------------------------===//

LongestPathAnalysis::~LongestPathAnalysis() { delete impl; }

LongestPathAnalysis::LongestPathAnalysis(
    Operation *moduleOp, mlir::AnalysisManager &am,
    const LongestPathAnalysisOptions &option)
    : impl(new Impl(moduleOp, am, option)), ctx(moduleOp->getContext()) {
  LLVM_DEBUG({
    llvm::dbgs() << "LongestPathAnalysis created\n";
    if (option.collectDebugInfo)
      llvm::dbgs() << " - Collecting debug info\n";
    if (option.lazyComputation)
      llvm::dbgs() << " - Lazy computation enabled\n";
    if (option.keepOnlyMaxDelayPaths)
      llvm::dbgs() << " - Keeping only max delay paths\n";
  });
}

bool LongestPathAnalysis::isAnalysisAvailable(StringAttr moduleName) const {
  return impl->isAnalysisAvailable(moduleName);
}

FailureOr<int64_t> LongestPathAnalysis::getAverageMaxDelay(Value value) {
  return impl->getAverageMaxDelay(value);
}

FailureOr<int64_t> LongestPathAnalysis::getMaxDelay(Value value,
                                                    int64_t bitPos) {
  return impl->getMaxDelay(value, bitPos);
}

LogicalResult
LongestPathAnalysis::getInternalPaths(StringAttr moduleName,
                                      SmallVectorImpl<DataflowPath> &results,
                                      bool elaboratePaths) const {
  if (!isAnalysisValid)
    return failure();
  if (elaboratePaths)
    return impl->collectClosedPaths<true>(moduleName, results);
  return impl->collectClosedPaths<false>(moduleName, results);
}

LogicalResult LongestPathAnalysis::getOpenPathsFromInputPortsToInternal(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  if (!isAnalysisValid)
    return failure();

  return impl->collectInputToInternalPaths(moduleName, results);
}

LogicalResult LongestPathAnalysis::getOpenPathsFromInternalToOutputPorts(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  if (!isAnalysisValid)
    return failure();

  return impl->collectInternalToOutputPaths(moduleName, results);
}

LogicalResult
LongestPathAnalysis::getAllPaths(StringAttr moduleName,
                                 SmallVectorImpl<DataflowPath> &results,
                                 bool elaboratePaths) const {
  if (failed(getInternalPaths(moduleName, results, elaboratePaths)))
    return failure();
  if (failed(getOpenPathsFromInputPortsToInternal(moduleName, results)))
    return failure();
  if (failed(getOpenPathsFromInternalToOutputPorts(moduleName, results)))
    return failure();
  return success();
}

ArrayRef<hw::HWModuleOp> LongestPathAnalysis::getTopModules() const {
  return impl->getTopModules();
}

FailureOr<ArrayRef<OpenPath>>
LongestPathAnalysis::computeLocalPaths(Value value, size_t bitPos) {
  if (!isAnalysisValid)
    return failure();

  return impl->computeLocalPaths(value, bitPos);
}

LogicalResult LongestPathAnalysis::computeGlobalPaths(
    Value value, size_t bitPos, SmallVectorImpl<DataflowPath> &results) {
  if (!isAnalysisValid)
    return mlir::emitError(value.getLoc()) << "analysis has been invalidated";

  return impl->computeGlobalPaths(value, bitPos, results);
}

//===----------------------------------------------------------------------===//
// IncrementalLongestPathAnalysis
//===----------------------------------------------------------------------===//

bool IncrementalLongestPathAnalysis::isOperationValidToMutate(
    Operation *op) const {
  if (!isAnalysisValid)
    return false;

  auto parentHWModule =
      op->getParentRegion()->getParentOfType<hw::HWModuleOp>();
  if (!parentHWModule)
    return true;
  auto *localVisitor =
      impl->ctx.getLocalVisitor(parentHWModule.getModuleNameAttr());
  if (!localVisitor)
    return true;

  // If all results of the operation have no paths, then it is safe to mutate
  // the operation.
  return llvm::all_of(op->getResults(), [localVisitor](Value value) {
    for (int64_t i = 0, e = getBitWidth(value); i < e; ++i) {
      auto path = localVisitor->getCachedPaths(value, i);
      if (!path.empty())
        return false;
    }
    return true;
  });
}

void IncrementalLongestPathAnalysis::notifyOperationModified(Operation *op) {
  if (!isOperationValidToMutate(op))
    isAnalysisValid = false;
}
void IncrementalLongestPathAnalysis::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  notifyOperationModified(op);
}

void IncrementalLongestPathAnalysis::notifyOperationErased(Operation *op) {
  notifyOperationModified(op);
}

// ===----------------------------------------------------------------------===//
// LongestPathCollection
// ===----------------------------------------------------------------------===//

void LongestPathCollection::sortInDescendingOrder() {
  llvm::stable_sort(paths, [](const DataflowPath &a, const DataflowPath &b) {
    return a.getDelay() > b.getDelay();
  });
}

void LongestPathCollection::dropNonCriticalPaths(bool perEndPoint) {
  // Deduplicate paths by start/end-point, keeping only the worst-case delay.
  if (perEndPoint) {
    llvm::DenseSet<DataflowPath::EndPointType> seen;
    for (size_t i = 0; i < paths.size(); ++i) {
      if (seen.insert(paths[i].getEndPoint()).second)
        paths[seen.size() - 1] = std::move(paths[i]);
    }

    paths.resize(seen.size());
  } else {
    llvm::DenseSet<Object> seen;
    for (size_t i = 0; i < paths.size(); ++i) {
      if (seen.insert(paths[i].getStartPoint()).second)
        paths[seen.size() - 1] = std::move(paths[i]);
    }
    paths.resize(seen.size());
  }
}

void LongestPathCollection::merge(const LongestPathCollection &other) {
  paths.append(other.paths.begin(), other.paths.end());
  sortInDescendingOrder();
}
