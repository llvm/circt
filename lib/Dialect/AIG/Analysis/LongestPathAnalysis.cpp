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

#include "circt/Dialect/AIG/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
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
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT//MapVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

#define DEBUG_TYPE "aig-longest-path-analysis"
using namespace circt;
using namespace aig;

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
  DenseMap<Key, size_t> saved;
  for (auto [i, path] :
       llvm::enumerate(ArrayRef(results).drop_front(startIndex))) {
    auto &slot = saved[keyFn(path)];
    if (slot == 0) {
      slot = startIndex + i + 1;
      continue;
    }
    if (delayFn(results[slot - 1]) < delayFn(path))
      results[slot - 1] = path;
  }

  results.resize(saved.size() + startIndex);
}

static void deduplicatePaths(SmallVectorImpl<OpenPath> &results,
                             size_t startIndex = 0) {
  deduplicatePathsImpl<OpenPath, Object>(
      results, startIndex, [](const auto &path) { return path.fanIn; },
      [](const auto &path) { return path.delay; });
}

static void deduplicatePaths(SmallVectorImpl<DataflowPath> &results,
                             size_t startIndex = 0) {
  deduplicatePathsImpl<DataflowPath, std::pair<Object, Object>>(
      results, startIndex,
      [](const DataflowPath &path) {
        return std::pair(path.getFanOut(), path.getFanIn());
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
  os << "Object(" << pathString << "." << getNameImpl(object.value).getValue()
     << "[" << object.bitPos << "]";
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

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTLONGESTPATHANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void OpenPath::print(llvm::raw_ostream &os) const {
  printObjectImpl(os, fanIn, delay, history);
}

void DebugPoint::print(llvm::raw_ostream &os) const {
  printObjectImpl(os, object, delay, {}, comment);
}

void Object::print(llvm::raw_ostream &os) const { printObjectImpl(os, *this); }

void DataflowPath::print(llvm::raw_ostream &os) {
  os << "root=" << root.getModuleName() << ", ";
  os << "fanOut=";
  fanOut.print(os);
  os << ", ";
  os << "fanIn=";
  path.print(os);
}

template <>
struct llvm::DenseMapInfo<Object> {
  using Info = llvm::DenseMapInfo<
      std::tuple<circt::igraph::InstancePath, Value, size_t>>;
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

class LocalVisitor;

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//
/// This class provides a thread-safe interface to access the analysis results.

class Context {
public:
  Context(igraph::InstanceGraph *instanceGraph,
          const LongestPathAnalysisOption &option)
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

  // Lookup a local visitor for `name`, and wait until it's done.
  const LocalVisitor *getAndWaitLocalVisitor(StringAttr name) const;

  // A map from the module name to the local visitor.
  llvm::MapVector<StringAttr, std::unique_ptr<LocalVisitor>> localVisitors;
  // This is non-null only if `module` is a ModuleOp.
  circt::igraph::InstanceGraph *instanceGraph = nullptr;

  bool doTraceDebugPoints() const { return option.traceDebugPoints; }

private:
  llvm::sys::SmartMutex<true> mutex;
  llvm::SetVector<StringAttr> running;
  const LongestPathAnalysisOption &option;
};

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
  FailureOr<ArrayRef<OpenPath>> getOrComputeResults(Value value, size_t bitPos);

  // Get the longest paths for the given value and bit position. This returns
  // an empty array if not pre-computed.
  ArrayRef<OpenPath> getResults(Value value, size_t bitPos) const;

  // A map from the object to the maximum distance and history.
  using ObjectToMaxDistance =
      llvm::MapVector<Object,
                      std::pair<int64_t, llvm::ImmutableList<DebugPoint>>>;

  void getClosedPaths(SmallVectorImpl<DataflowPath> &results) const;
  void setTopLevel() { this->topLevel = true; }
  bool isTopLevel() const { return topLevel; }
  hw::HWModuleOp getHWModuleOp() const { return module; }

  const auto &getFromInputPortToFanOut() const { return fromInputPortToFanOut; }
  const auto &getFromOutputPortToFanIn() const { return fromOutputPortToFanIn; }
  const auto &getFanOutResults() const { return fanOutResults; }

private:
  void putUnclosedResult(const Object &object, int64_t delay,
                         llvm::ImmutableList<DebugPoint> history,
                         ObjectToMaxDistance &objectToMaxDistance);

  // A map from the input port to the farthest fanOut.
  llvm::MapVector<std::pair<BlockArgument, size_t>, ObjectToMaxDistance>
      fromInputPortToFanOut;

  // A map from the output port to the farthest fanIn.
  llvm::MapVector<std::tuple<size_t, size_t>, ObjectToMaxDistance>
      fromOutputPortToFanIn;

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

  // Constants.
  LogicalResult visit(hw::ConstantOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return success();
  }

  // Registers are fanIn.
  LogicalResult visit(seq::FirRegOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markFanIn(op, bitPos, results);
  }

  LogicalResult visit(seq::CompRegOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markFanIn(op, bitPos, results);
  }

  LogicalResult visit(seq::FirMemReadOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markFanIn(op, bitPos, results);
  }

  LogicalResult visit(seq::FirMemReadWriteOp op, size_t bitPos,
                      SmallVectorImpl<OpenPath> &results) {
    return markFanIn(op, bitPos, results);
  }

  LogicalResult visitDefault(Operation *op, size_t bitPos,
                             SmallVectorImpl<OpenPath> &results);

  // Helper functions.
  LogicalResult addEdge(Value to, size_t toBitPos, int64_t delay,
                        SmallVectorImpl<OpenPath> &results);
  LogicalResult markFanIn(Value value, size_t bitPos,
                          SmallVectorImpl<OpenPath> &results);
  LogicalResult markRegFanOut(Value fanOut, Value start, Value reset = {},
                              Value resetValue = {}, Value enable = {});

  // The module we are visiting.
  hw::HWModuleOp module;
  Context *ctx;

  // Thread-local data structures.
  std::unique_ptr<circt::igraph::InstancePathCache> instancePathCache;
  // TODO: Make debug points optional.
  std::unique_ptr<llvm::ImmutableListFactory<DebugPoint>> debugPointFactory;

  // A map from the value point to the longest paths.
  DenseMap<std::pair<Value, size_t>, SmallVector<OpenPath>> cachedResults;

  // A map from the object to the longest paths.
  DenseMap<Object, SmallVector<OpenPath>> fanOutResults;

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
  done = false;
}

ArrayRef<OpenPath> LocalVisitor::getResults(Value value, size_t bitPos) const {
  std::pair<Value, size_t> valueAndBitPos(value, bitPos);
  auto leader = ec.findLeader(valueAndBitPos);
  if (leader != ec.member_end()) {
    if (*leader != valueAndBitPos) {
      // If this is not the leader, then use the leader.
      return getResults(leader->first, leader->second);
    }
  }

  auto it = cachedResults.find(valueAndBitPos);
  // If not found, then consider it to be a constant.
  if (it == cachedResults.end())
    return {};
  return it->second;
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

LogicalResult LocalVisitor::markRegFanOut(Value fanOut, Value start,
                                          Value reset, Value resetValue,
                                          Value enable) {
  auto bitWidth = getBitWidth(fanOut);
  auto record = [&](size_t fanOutBitPos, Value value, size_t bitPos) {
    auto result = getOrComputeResults(value, bitPos);
    if (failed(result))
      return failure();
    for (auto &path : *result) {
      if (auto blockArg = dyn_cast<BlockArgument>(path.fanIn.value)) {
        // Not closed.
        putUnclosedResult({{}, fanOut, fanOutBitPos}, path.delay, path.history,
                          fromInputPortToFanOut[{blockArg, path.fanIn.bitPos}]);
      } else {
        // If the fanIn is not a port, record to the results.
        fanOutResults[{{}, fanOut, fanOutBitPos}].push_back(path);
      }
    }
    return success();
  };

  // Get paths for each bit, and record them.
  for (size_t i = 0, e = bitWidth; i < e; ++i) {
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
  auto leader = ec.getOrInsertLeaderValue({to, toBitPos});
  (void)leader;
  // Merge classes, and visit the leader.
  auto newLeader = ec.unionSets({to, toBitPos}, {from, fromBitPos});
  assert(leader == *newLeader);
  return visitValue(to, toBitPos, results);
}

LogicalResult LocalVisitor::addEdge(Value to, size_t bitPos, int64_t delay,
                                    SmallVectorImpl<OpenPath> &results) {
  auto result = getOrComputeResults(to, bitPos);
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
  deduplicatePaths(results);
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

LogicalResult LocalVisitor::markFanIn(Value value, size_t bitPos,
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
  // fanIn.
  if (!ctx->instanceGraph)
    return markFanIn(value, bitPos, results);

  // If an instance graph is available, then we can look up the module.
  auto *node = ctx->instanceGraph->lookup(moduleName);
  assert(node && "module not found");

  // Otherwise, if the module is not a HWModuleOp, then we should treat it as a
  // fanIn.
  if (!isa<hw::HWModuleOp>(node->getModule()))
    return markFanIn(value, bitPos, results);

  auto *localVisitor = ctx->getAndWaitLocalVisitor(moduleName);
  auto *fanInIt = localVisitor->fromOutputPortToFanIn.find({resultNum, bitPos});

  // It means output is constant, so it's ok.
  if (fanInIt == localVisitor->fromOutputPortToFanIn.end())
    return success();

  const auto &fanIns = fanInIt->second;

  for (auto &[fanInPoint, delayAndHistory] : fanIns) {
    auto [delay, history] = delayAndHistory;
    auto newPath =
        instancePathCache->prependInstance(op, fanInPoint.instancePath);

    // If the fanIn is not a block argument, record it directly.
    auto arg = dyn_cast<BlockArgument>(fanInPoint.value);
    if (!arg) {
      // Update the history to have correct instance path.
      auto newHistory = debugPointFactory->getEmptyList();
      if (ctx->doTraceDebugPoints()) {
        newHistory =
            mapList(debugPointFactory.get(), history, [&](DebugPoint p) {
              p.object.instancePath = newPath;
              return p;
            });
        newHistory = debugPointFactory->add(
            DebugPoint({}, value, bitPos, delay, "output port"), newHistory);
      }

      results.emplace_back(newPath, fanInPoint.value, fanInPoint.bitPos, delay,
                           newHistory);
      continue;
    }

    // Otherwise, we need to look up the instance operand.
    auto result = getOrComputeResults(op->getOperand(arg.getArgNumber()),
                                      fanInPoint.bitPos);
    if (failed(result))
      return failure();
    for (auto path : *result) {
      auto newHistory = debugPointFactory->getEmptyList();
      if (ctx->doTraceDebugPoints()) {
        // Update the history to have correct instance path.
        newHistory =
            mapList(debugPointFactory.get(), history, [&](DebugPoint p) {
              p.object.instancePath = newPath;
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
  auto cost = std::max(1u, llvm::Log2_64_Ceil(size));
  // Create edges each operand with cost ceil(log(size)).
  for (auto operand : op->getOperands())
    if (failed(addEdge(operand, bitPos, cost, results)))
      return failure();

  deduplicatePaths(results);
  return success();
}

LogicalResult LocalVisitor::visitDefault(Operation *op, size_t bitPos,
                                         SmallVectorImpl<OpenPath> &results) {
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

FailureOr<ArrayRef<OpenPath>> LocalVisitor::getOrComputeResults(Value value,
                                                                size_t bitPos) {
  if (ec.contains({value, bitPos})) {
    auto leader = ec.findLeader({value, bitPos});
    // If this is not the leader, then use the leader.
    if (*leader != std::pair(value, bitPos)) {
      return getOrComputeResults(leader->first, leader->second);
    }
  }

  auto it = cachedResults.find({value, bitPos});
  if (it != cachedResults.end())
    return ArrayRef<OpenPath>(it->second);

  SmallVector<OpenPath> results;
  if (failed(visitValue(value, bitPos, results)))
    return {};

  // Unique the results.
  deduplicatePaths(results);
  LLVM_DEBUG({
    llvm::dbgs() << value << "[" << bitPos << "] "
                 << "Found " << results.size() << " paths\n";
    llvm::dbgs() << "====Paths:\n";
    for (auto &path : results) {
      path.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "====\n";
  });

  auto insertedResult =
      cachedResults.try_emplace({value, bitPos}, std::move(results));
  assert(insertedResult.second);
  return ArrayRef<OpenPath>(insertedResult.first->second);
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
  auto result =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp,
                aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::MuxOp,
                comb::XorOp, seq::FirRegOp, seq::CompRegOp, hw::ConstantOp,
                seq::FirMemReadOp, seq::FirMemReadWriteOp, hw::WireOp>(
              [&](auto op) {
                size_t idx = results.size();
                auto result = visit(op, bitPos, results);
                if (ctx->doTraceDebugPoints())
                  if (auto name = op->template getAttrOfType<StringAttr>(
                          "sv.namehint")) {

                    for (auto i = idx, e = results.size(); i < e; ++i) {
                      DebugPoint debugPoint({}, value, bitPos, results[i].delay,
                                            "namehint");
                      auto newHistory = debugPointFactory->add(
                          debugPoint, results[i].history);
                      results[i].history = newHistory;
                    }
                  }
                return result;
              })
          .Case<hw::InstanceOp>([&](hw::InstanceOp op) {
            return visit(op, bitPos, cast<OpResult>(value).getResultNumber(),
                         results);
          })
          .Default([&](auto op) { return visitDefault(op, bitPos, results); });
  return result;
}

LogicalResult LocalVisitor::initializeAndRun(hw::InstanceOp instance) {
  const auto *childVisitor =
      ctx->getAndWaitLocalVisitor(instance.getReferencedModuleNameAttr());
  // If not found, the module is blackbox so skip it.
  if (!childVisitor)
    return success();

  // Connect dataflow from instance input ports to fanout in the child.
  for (const auto &[object, openPaths] :
       childVisitor->getFromInputPortToFanOut()) {
    auto [arg, argBitPos] = object;
    for (auto [point, delayAndHistory] : openPaths) {
      auto [instancePath, fanOut, fanOutBitPos] = point;
      auto [delay, history] = delayAndHistory;
      // Prepend the instance path.
      assert(instancePathCache);
      auto newPath = instancePathCache->prependInstance(instance, instancePath);
      auto computedResults = getOrComputeResults(
          instance.getOperand(arg.getArgNumber()), argBitPos);
      if (failed(computedResults))
        return failure();

      for (auto &result : *computedResults) {
        auto newHistory = ctx->doTraceDebugPoints()
                              ? mapList(debugPointFactory.get(), history,
                                        [&](DebugPoint p) {
                                          // Update the instance path to prepend
                                          // the current instance.
                                          p.object.instancePath = newPath;
                                          p.delay += result.delay;
                                          return p;
                                        })
                              : debugPointFactory->getEmptyList();
        if (auto newPort = dyn_cast<BlockArgument>(result.fanIn.value)) {
          putUnclosedResult(
              {newPath, fanOut, fanOutBitPos}, result.delay + delay, newHistory,
              fromInputPortToFanOut[{newPort, result.fanIn.bitPos}]);
        } else {
          fanOutResults[{newPath, fanOut, fanOutBitPos}].emplace_back(
              newPath, result.fanIn.value, result.fanIn.bitPos,
              result.delay + delay,
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
      auto computedResults = getOrComputeResults(instance, i);
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
          fromOutputPortToFanIn[{operand.getOperandNumber(), i}];
      auto computedResults = getOrComputeResults(operand.get(), i);
      if (failed(computedResults))
        return failure();
      for (const auto &result : *computedResults) {
        putUnclosedResult(result.fanIn, result.delay, result.history,
                          recordOutput);
      }
    }
  }
  return success();
}

LogicalResult LocalVisitor::initializeAndRun() {
  LLVM_DEBUG({ ctx->notifyStart(module.getModuleNameAttr()); });
  // Initialize the results for the block arguments.
  for (auto blockArgument : module.getBodyBlock()->getArguments())
    for (size_t i = 0, e = getBitWidth(blockArgument); i < e; ++i)
      (void)getOrComputeResults(blockArgument, i);

  auto walkResult = module->walk([&](Operation *op) {
    auto result =
        mlir::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<seq::FirRegOp>([&](seq::FirRegOp op) {
              return markRegFanOut(op, op.getNext(), op.getReset(),
                                   op.getResetValue());
            })
            .Case<seq::CompRegOp>([&](auto op) {
              return markRegFanOut(op, op.getInput(), op.getReset(),
                                   op.getResetValue());
            })
            .Case<seq::FirMemWriteOp>([&](auto op) {
              // TODO: Add address.
              return markRegFanOut(op.getMemory(), op.getData(), {}, {},
                                   op.getEnable());
            })
            .Case<seq::FirMemReadWriteOp>([&](seq::FirMemReadWriteOp op) {
              // TODO: Add address.
              return markRegFanOut(op.getMemory(), op.getWriteData(), {}, {},
                                   op.getEnable());
            })
            .Case<aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp,
                  comb::MuxOp>([&](auto op) {
              // NOTE: Visiting and-inverter is not necessary but
              // useful to reduce recursion depth.
              for (size_t i = 0, e = getBitWidth(op); i < e; ++i)
                if (failed(getOrComputeResults(op, i)))
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
  auto *it = localVisitors.find(name);
  if (it == localVisitors.end())
    return nullptr;
  return it->second.get();
}

const LocalVisitor *Context::getAndWaitLocalVisitor(StringAttr name) const {
  auto *visitor = getLocalVisitor(name);
  if (!visitor)
    return nullptr;
  visitor->waitUntilDone();
  return visitor;
}

//===----------------------------------------------------------------------===//
// LongestPathAnalysis::Impl
//===----------------------------------------------------------------------===//

struct LongestPathAnalysis::Impl {
  Impl(Operation *module, mlir::AnalysisManager &am,
       const LongestPathAnalysisOption &option);
  LogicalResult initializeAndRun(mlir::ModuleOp module);
  LogicalResult initializeAndRun(hw::HWModuleOp module);

  // See LongestPathAnalysis.
  bool isAnalysisAvailable(StringAttr moduleName) const;
  int64_t getAverageMaxDelay(Value value) const;
  int64_t getMaxDelay(Value value) const;
  LogicalResult
  getResults(Value value, size_t bitPos, SmallVectorImpl<DataflowPath> &results,
             circt::igraph::InstancePathCache *instancePathCache = nullptr,
             llvm::ImmutableListFactory<DebugPoint> *debugPointFactory =
                 nullptr) const;

  LogicalResult getClosedPaths(StringAttr moduleName,
                               SmallVectorImpl<DataflowPath> &results) const;
  LogicalResult
  getOpenPaths(StringAttr moduleName,
               SmallVectorImpl<std::pair<Object, OpenPath>> &openPathsFromFF,
               SmallVectorImpl<std::tuple<size_t, size_t, OpenPath>>
                   &openPathsFromOutputPorts) const;

  llvm::ArrayRef<hw::HWModuleOp> getTopModules() const { return topModules; }

private:
  LogicalResult getResultsImpl(
      const Object &originalObject, Value value, size_t bitPos,
      SmallVectorImpl<DataflowPath> &results,
      circt::igraph::InstancePathCache *instancePathCache,
      llvm::ImmutableListFactory<DebugPoint> *debugPointFactory) const;

  Context ctx;
  SmallVector<hw::HWModuleOp> topModules;
};

LogicalResult LongestPathAnalysis::Impl::getResults(
    Value value, size_t bitPos, SmallVectorImpl<DataflowPath> &results,
    circt::igraph::InstancePathCache *instancePathCache,
    llvm::ImmutableListFactory<DebugPoint> *debugPointFactory) const {
  return getResultsImpl(Object({}, value, bitPos), value, bitPos, results,
                        instancePathCache, debugPointFactory);
}

LogicalResult LongestPathAnalysis::Impl::getResultsImpl(
    const Object &originalObject, Value value, size_t bitPos,
    SmallVectorImpl<DataflowPath> &results,
    circt::igraph::InstancePathCache *instancePathCache,
    llvm::ImmutableListFactory<DebugPoint> *debugPointFactory) const {
  auto parentHWModule =
      value.getParentRegion()->getParentOfType<hw::HWModuleOp>();
  if (!parentHWModule)
    return mlir::emitError(value.getLoc())
           << "query value is not in a HWModuleOp";
  auto *localVisitor = ctx.getLocalVisitor(parentHWModule.getModuleNameAttr());
  if (!localVisitor)
    return success();

  size_t oldIndex = results.size();
  auto *node =
      ctx.instanceGraph
          ? ctx.instanceGraph->lookup(parentHWModule.getModuleNameAttr())
          : nullptr;
  LLVM_DEBUG({
    llvm::dbgs() << "Running " << parentHWModule.getModuleNameAttr() << " "
                 << value << " " << bitPos << "\n";
  });

  for (auto &path : localVisitor->getResults(value, bitPos)) {
    auto arg = dyn_cast<BlockArgument>(path.fanIn.value);
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

      auto result = getResultsImpl(
          newObject, inst->getInstance()->getOperand(arg.getArgNumber()),
          path.fanIn.bitPos, results, instancePathCache, debugPointFactory);
      if (failed(result))
        return result;
      for (auto i = startIndex, e = results.size(); i < e; ++i)
        results[i].setDelay(results[i].getDelay() + path.delay);
    }
  }

  deduplicatePaths(results, oldIndex);
  return success();
}

LogicalResult LongestPathAnalysis::Impl::getClosedPaths(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  auto collectClosedPaths = [&](StringAttr name) {
    if (!isAnalysisAvailable(name))
      return;
    auto *visitor = ctx.getLocalVisitor(name);
    for (auto &[point, state] : visitor->getFanOutResults())
      for (const auto &dataFlow : state)
        results.emplace_back(point, dataFlow, visitor->getHWModuleOp());
  };

  if (ctx.instanceGraph) {
    // Accumulate all closed results under the given module.
    auto *node = ctx.instanceGraph->lookup(moduleName);
    for (auto *child : llvm::post_order(node))
      collectClosedPaths(child->getModule().getModuleNameAttr());
  } else {
    collectClosedPaths(moduleName);
  }

  // Sort by delay.
  // TODO: Consider using names and bitpos when tie-breaking to make it more
  // stable.
  std::sort(results.begin(), results.end(),
            [](DataflowPath &a, DataflowPath &b) {
              return a.getDelay() > b.getDelay();
            });
  return success();
}

LogicalResult LongestPathAnalysis::Impl::getOpenPaths(
    StringAttr moduleName,
    SmallVectorImpl<std::pair<Object, OpenPath>> &openPathsFromFF,
    SmallVectorImpl<std::tuple<size_t, size_t, OpenPath>>
        &openPathsFromOutputPorts) const {
  auto *visitor = ctx.getLocalVisitor(moduleName);
  if (!visitor)
    return failure();

  for (auto &[key, value] : visitor->getFromInputPortToFanOut()) {
    auto [arg, argBitPos] = key;
    for (auto [point, delayAndHistory] : value) {
      auto [path, start, startBitPos] = point;
      auto [delay, history] = delayAndHistory;
      openPathsFromFF.emplace_back(
          Object(path, start, startBitPos),
          OpenPath({}, arg, argBitPos, delay, history));
    }
  }

  for (auto &[key, value] : visitor->getFromOutputPortToFanIn()) {
    auto [resultNum, bitPos] = key;
    for (auto [point, delayAndHistory] : value) {
      auto [path, start, startBitPos] = point;
      auto [delay, history] = delayAndHistory;
      openPathsFromOutputPorts.emplace_back(
          resultNum, bitPos,
          OpenPath(path, start, startBitPos, delay, history));
    }
  }

  std::sort(openPathsFromFF.begin(), openPathsFromFF.end(),
            [](auto &a, auto &b) { return a.second.delay > b.second.delay; });
  std::sort(openPathsFromOutputPorts.begin(), openPathsFromOutputPorts.end(),
            [](auto &a, auto &b) {
              return std::get<2>(a).delay > std::get<2>(b).delay;
            });
  return success();
}

LongestPathAnalysis::Impl::Impl(Operation *moduleOp, mlir::AnalysisManager &am,
                                const LongestPathAnalysisOption &option)
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
  auto topNameAttr =
      module->getAttrOfType<FlatSymbolRefAttr>(getTopModuleNameAttrName());

  topModules.clear();
  llvm::SetVector<Operation *> visited;
  auto *instanceGraph = ctx.instanceGraph;
  if (topNameAttr) {
    auto *topNode = instanceGraph->lookup(topNameAttr.getAttr());
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
int64_t LongestPathAnalysis::Impl::getAverageMaxDelay(Value value) const {
  SmallVector<DataflowPath> results;
  size_t bitWidth = getBitWidth(value);
  if (bitWidth == 0)
    return 0;
  int64_t totalDelay = 0;
  for (size_t i = 0; i < bitWidth; ++i) {
    // Clear results from previous iteration.
    results.clear();
    auto result = getResults(value, i, results);
    if (failed(result))
      return 0;

    int64_t maxDelay = 0;
    for (auto &path : results)
      maxDelay = std::max(maxDelay, path.getDelay());
    totalDelay += maxDelay;
  }
  return llvm::divideCeil(totalDelay, bitWidth);
}

int64_t LongestPathAnalysis::Impl::getMaxDelay(Value value) const {
  SmallVector<DataflowPath> results;
  size_t bitWidth = getBitWidth(value);
  if (bitWidth == 0)
    return 0;
  int64_t maxDelay = 0;
  for (size_t i = 0; i < bitWidth; ++i) {
    // Clear results from previous iteration.
    results.clear();
    auto result = getResults(value, i, results);
    if (failed(result))
      return 0;

    for (auto &path : results)
      maxDelay = std::max(maxDelay, path.getDelay());
  }
  return maxDelay;
}

//===----------------------------------------------------------------------===//
// LongestPathAnalysis
//===----------------------------------------------------------------------===//

LongestPathAnalysis::~LongestPathAnalysis() { delete impl; }

LongestPathAnalysis::LongestPathAnalysis(
    Operation *moduleOp, mlir::AnalysisManager &am,
    const LongestPathAnalysisOption &option)
    : impl(new Impl(moduleOp, am, option)) {}

bool LongestPathAnalysis::isAnalysisAvailable(StringAttr moduleName) const {
  return impl->isAnalysisAvailable(moduleName);
}

int64_t LongestPathAnalysis::getAverageMaxDelay(Value value) const {
  return impl->getAverageMaxDelay(value);
}

int64_t LongestPathAnalysis::getMaxDelay(Value value) const {
  return impl->getMaxDelay(value);
}

LogicalResult LongestPathAnalysis::getClosedPaths(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  return impl->getClosedPaths(moduleName, results);
}

LogicalResult LongestPathAnalysis::getOpenPaths(
    StringAttr moduleName,
    SmallVectorImpl<std::pair<Object, OpenPath>> &openPathsToFF,
    SmallVectorImpl<std::tuple<size_t, size_t, OpenPath>>
        &openPathsFromOutputPorts) const {
  return impl->getOpenPaths(moduleName, openPathsToFF,
                            openPathsFromOutputPorts);
}

ArrayRef<hw::HWModuleOp> LongestPathAnalysis::getTopModules() const {
  return impl->getTopModules();
}

namespace {
struct PrintLongestPathAnalysisPass
    : public impl::PrintLongestPathAnalysisBase<PrintLongestPathAnalysisPass> {
  using PrintLongestPathAnalysisBase::outputFile;
  using PrintLongestPathAnalysisBase::PrintLongestPathAnalysisBase;
  using PrintLongestPathAnalysisBase::showTopKPercent;
  void runOnOperation() override;
  LogicalResult printAnalysisResult(const LongestPathAnalysis &analysis,
                                    igraph::InstancePathCache &pathCache,
                                    hw::HWModuleOp top, llvm::raw_ostream &os);
};

} // namespace

LogicalResult PrintLongestPathAnalysisPass::printAnalysisResult(
    const LongestPathAnalysis &analysis, igraph::InstancePathCache &pathCache,
    hw::HWModuleOp top, llvm::raw_ostream &os) {
  SmallVector<DataflowPath> closedPaths;
  SmallVector<std::pair<Object, OpenPath>> openPathsToFF;
  SmallVector<std::tuple<size_t, size_t, OpenPath>> openPathsFromOutputPorts;
  auto moduleName = top.getModuleNameAttr();
  if (failed(analysis.getClosedPaths(moduleName, closedPaths)) ||
      failed(analysis.getOpenPaths(moduleName, openPathsToFF,
                                   openPathsFromOutputPorts)))
    return failure();

  // Emit diagnostics if testing is enabled.
  if (test) {
    for (auto &result : closedPaths) {
      auto fanOutLoc = result.getFanOut().value.getLoc();
      auto diag = mlir::emitRemark(fanOutLoc);
      SmallString<128> buf;
      llvm::raw_svector_ostream os(buf);
      result.print(os);
      diag << buf;
    }
    for (auto &[object, path] : openPathsToFF) {
      auto loc = object.value.getLoc();
      auto diag = mlir::emitRemark(loc);
      DataflowPath closedPath(object, path, top);
      SmallString<128> buf;
      llvm::raw_svector_ostream os(buf);
      closedPath.print(os);
      diag << buf;
    }
    for (auto &[resultNum, bitPos, path] : openPathsFromOutputPorts) {
      auto outputPortLoc = top.getOutputLoc(resultNum);
      auto outputPortName = top.getOutputName(resultNum);
      auto diag = mlir::emitRemark(outputPortLoc);
      SmallString<128> buf;
      llvm::raw_svector_ostream os(buf);
      path.print(os);
      diag << "fanOut=Object($root." << outputPortName << "[" << bitPos
           << "]), fanIn=" << buf;
    }
  }

  os << "# Analysis result for " << top.getModuleNameAttr() << "\n"
     << "Found " << closedPaths.size() << " closed paths\n";
  if (!closedPaths.empty())
    os << "Maximum path delay: " << closedPaths.front().getDelay() << "\n";

  // TODO: Print open paths.
  return success();
}

void PrintLongestPathAnalysisPass::runOnOperation() {
  auto am = getAnalysisManager();
  aig::LongestPathAnalysis analysis(getOperation(), am,
                                    {
                                        true,
                                    });
  igraph::InstancePathCache pathCache(
      getAnalysis<circt::igraph::InstanceGraph>());
  auto outputFileVal = outputFile.getValue();

  std::string error;
  auto file = mlir::openOutputFile(outputFile.getValue(), &error);
  if (!file) {
    llvm::errs() << error;
    return signalPassFailure();
  }

  for (auto top : analysis.getTopModules())
    if (failed(printAnalysisResult(analysis, pathCache, top, file->os())))
      return signalPassFailure();
  file->keep();
  return markAllAnalysesPreserved();
}
