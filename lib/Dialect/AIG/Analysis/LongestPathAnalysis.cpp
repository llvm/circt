//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the critical path analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGAnalysis.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/CSE.h"
#include "llvm//Support/ToolOutputFile.h"
#include "llvm/ADT//MapVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Mutex.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mutex>

#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <queue>
#include <set>

#define DEBUG_TYPE "aig-longest-path-analysis"
using namespace circt;
using namespace aig;

static size_t getBitWidth(Value value) {
  if (auto vecType = dyn_cast<seq::ClockType>(value.getType()))
    return 1;
  return hw::getBitWidth(value.getType());
}

static bool isRootValue(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return true;
  return isa<seq::CompRegOp, seq::FirRegOp, hw::InstanceOp, seq::FirMemReadOp>(
      value.getDefiningOp());
}

static void printImpl(llvm::raw_ostream &os, StringAttr name,
                      circt::igraph::InstancePath path, size_t bitPos,
                      int64_t delay, StringRef comment) {
  std::string pathString;
  llvm::raw_string_ostream osPath(pathString);
  path.print(osPath);
  os << "Object(" << pathString << "." << name.getValue() << "[" << bitPos
     << "]";
  os << ", delay=" << delay;
  if (!comment.empty())
    os << ", comment=\"" << comment << "\"";
  os << ")";
}

static void deduplicatePaths(SmallVectorImpl<DataflowPath> &results,
                             size_t startIndex = 0) {
  // Take only maximum for each path destination.
  DenseMap<std::tuple<circt::igraph::InstancePath, Value, size_t>, size_t>
      saved;
  for (auto [i, path] :
       llvm::enumerate(ArrayRef(results).drop_front(startIndex))) {
    auto &slot = saved[{path.instancePath, path.value, path.bitPos}];
    if (slot == 0) {
      slot = startIndex + i + 1;
      continue;
    }
    if (results[slot - 1].delay < path.delay)
      results[slot - 1] = path;
  }

  results.resize(saved.size() + startIndex);
}

static void deduplicatePaths(SmallVectorImpl<PathResult> &results,
                             size_t startIndex = 0) {
  // Take only maximum for each path destination.
  DenseMap<std::tuple<circt::igraph::InstancePath, Value, size_t,
                      circt::igraph::InstancePath, Value, size_t>,
           size_t>
      saved;
  for (auto [i, path] :
       llvm::enumerate(ArrayRef(results).drop_front(startIndex))) {
    auto &slot =
        saved[{path.fanout.instancePath, path.fanout.value, path.fanout.bitPos,
               path.fanin.instancePath, path.fanin.value, path.fanin.bitPos}];
    if (slot == 0) {
      slot = startIndex + i + 1;
      continue;
    }
    if (results[slot - 1].fanin.delay < path.fanin.delay)
      results[slot - 1] = path;
  }

  results.resize(saved.size() + startIndex);
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

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTLONGESTPATHANALYSIS
#define GEN_PASS_DEF_PRINTGLOBALDATAPATHANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

namespace {
struct PrintLongestPathAnalysisPass
    : public impl::PrintLongestPathAnalysisBase<PrintLongestPathAnalysisPass> {
  using PrintLongestPathAnalysisBase::outputFile;
  using PrintLongestPathAnalysisBase::PrintLongestPathAnalysisBase;
  using PrintLongestPathAnalysisBase::showTopKPercent;
  void runOnOperation() override;
};

} // namespace

static StringAttr getNameImpl(Value value, size_t bitPos) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto op = cast<hw::HWModuleOp>(arg.getParentBlock()->getParentOp());
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
        str += "_read_port";
        return StringAttr::get(value.getContext(), str);
      })
      .Default([&](auto op) {
        llvm::errs() << "Unknown op: " << *op << "\n";
        return StringAttr::get(value.getContext(), "");
      });
}

void DataflowPath::print(llvm::raw_ostream &os) const {
  os << "Object(";
  instancePath.print(os);
  os << "." << getNameImpl(value, bitPos).getValue() << "[" << bitPos
     << "], delay=" << delay << ")";
}

void Object::print(llvm::raw_ostream &os) const {
  os << "Object(";
  instancePath.print(os);
  os << "." << getNameImpl(value, bitPos).getValue() << "[" << bitPos << "])";
}

template <>
struct llvm::DenseMapInfo<Object> {
  static Object getEmptyKey() {
    auto [path, value, bitPos] = llvm::DenseMapInfo<
        std::tuple<circt::igraph::InstancePath, Value, size_t>>::getEmptyKey();
    return Object(path, value, bitPos);
  }
  static Object getTombstoneKey() {
    auto [path, value, bitPos] =
        llvm::DenseMapInfo<std::tuple<circt::igraph::InstancePath, Value,
                                      size_t>>::getTombstoneKey();
    return Object(path, value, bitPos);
  }
  static llvm::hash_code getHashValue(Object object) {
    return llvm::hash_combine(object.instancePath.getHash(),
                              object.value.getAsOpaquePointer(), object.bitPos);
  }
  static bool isEqual(const Object &a, const Object &b) { return a == b; }
};

class LocalVisitor;
struct Context {
  llvm::sys::SmartMutex<true> mutex;
  llvm::SetVector<StringAttr> running;
  // This is non-null only if `module` is a ModuleOp.
  circt::igraph::InstanceGraph *instanceGraph = nullptr;

  // If true, store longest paths for flip flops.
  bool storeFlipFlopLongestPath = false;

  Context(igraph::InstanceGraph *instanceGraph, bool storeFlipFlopLongestPath)
      : instanceGraph(instanceGraph),
        storeFlipFlopLongestPath(storeFlipFlopLongestPath) {}

  void notifyStart(StringAttr name) {
    std::lock_guard<llvm::sys::SmartMutex<true>> lock(mutex);
    running.insert(name);
    llvm::errs() << "[Timing] " << name << " started. running=[";
    for (auto &name : running)
      llvm::errs() << name << " ";
    llvm::errs() << "]\n";
  }
  void notifyEnd(StringAttr name) {
    std::lock_guard<llvm::sys::SmartMutex<true>> lock(mutex);
    running.remove(name);

    llvm::errs() << "[Timing] " << name << " finished. running=[";
    for (auto &name : running)
      llvm::errs() << name << " ";
    llvm::errs() << "]\n";
  }

  const LocalVisitor *getAndWaitLocalVisitor(StringAttr name) const;
  const LocalVisitor *getLocalVisitor(StringAttr name) const;

  llvm::MapVector<StringAttr, std::unique_ptr<LocalVisitor>> localvisitors;
};

class LocalVisitor {
public:
  LocalVisitor(hw::HWModuleOp module, Context *ctx) : module(module), ctx(ctx) {
    debugPointFactory =
        std::make_unique<llvm::ImmutableListFactory<DebugPoint>>();
    instancePathCache =
        ctx->instanceGraph ? std::make_unique<circt::igraph::InstancePathCache>(
                                 *ctx->instanceGraph)
                           : nullptr;
    done = false;
  }

  LogicalResult initializeAndRun();
  std::pair<Value, size_t> findLeader(Value value, size_t bitpos) const {
    return ec.getLeaderValue({value, bitpos});
  }

  hw::HWModuleOp getHWModule() const { return module; }

  bool isDone() const { return done.load(); }

  const auto &getFanoutResults() const { return fanoutResults; }

  // Wait until the thread is done.
  void waitUntilDone() const {
    while (!done.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  FailureOr<ArrayRef<DataflowPath>> getOrComputeResults(Value value,
                                                        size_t bitPos);
  ArrayRef<DataflowPath> getResults(Value value, size_t bitPos) const {
    auto it = cachedResults.find({value, bitPos});
    if (it == cachedResults.end()) {
      std::lock_guard<llvm::sys::SmartMutex<true>> lock(ctx->mutex);
      llvm::errs() << "Not found: " << value << "[" << bitPos << "]\n";
      return {};
    }
    return it->second;
  }

  using SavedPoint =
      llvm::MapVector<Object,
                      std::pair<int64_t, llvm::ImmutableList<DebugPoint>>>;

  void getResultsForFF(SmallVectorImpl<PathResult> &results);

  void setTopLevel() { this->topLevel = true; }

  bool isTopLevel() const { return topLevel; }
  const auto &getFromInputPortToFanOut() const { return fromInputPortToFanOut; }

private:
  void saveInputPort(circt::igraph::InstancePath instancePath, Value value,
                     size_t bitPos, int64_t delay,
                     llvm::ImmutableList<DebugPoint> history,
                     SavedPoint &saved) {
    auto &slot = saved[{instancePath, value, bitPos}];
    if (slot.first >= delay)
      return;
    slot = {delay, history};
  }

  llvm::MapVector<std::pair<BlockArgument, size_t>, SavedPoint>
      fromInputPortToFanOut;
  llvm::MapVector<std::tuple<size_t, size_t>, SavedPoint> fromOutputPortToFanIn;

  LogicalResult visitValue(Value value, size_t bitPos,
                           SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(mlir::BlockArgument argument, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(hw::InstanceOp op, size_t bitPos, size_t resultNum,
                      SmallVectorImpl<DataflowPath> &results);

  // Data-movement ops.
  LogicalResult visit(hw::WireOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(comb::ConcatOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(comb::ExtractOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(comb::ReplicateOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);

  // Bit-logical ops.
  LogicalResult visit(aig::AndInverterOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(comb::AndOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(comb::XorOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(comb::OrOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult visit(comb::MuxOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
  LogicalResult addLogicOp(Operation *op, size_t bitPos,
                           SmallVectorImpl<DataflowPath> &results);

  // Constants.
  LogicalResult visit(hw::ConstantOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results) {
    return success();
  }

  // FanIn.
  LogicalResult visit(seq::FirRegOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results) {
    markFanIn(op, bitPos, results);
    return success();
  }

  LogicalResult visit(seq::CompRegOp op, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results) {
    markFanIn(op, bitPos, results);
    return success();
  }

  LogicalResult visitDefault(Operation *op, size_t bitPos,
                             SmallVectorImpl<DataflowPath> &results);

  LogicalResult addEdge(Value to, size_t toBitPos, int64_t delay,
                        SmallVectorImpl<DataflowPath> &results);

  // This tracks equivalence classes of values. If an operation is purely a
  // data movement, then we can treat the output as equivalent to the input.
  llvm::EquivalenceClasses<std::pair<Value, size_t>> ec;
  DenseMap<std::pair<Value, size_t>, std::pair<Value, size_t>> ecMap;

  LogicalResult markEquivalent(Value from, size_t fromBitPos, Value to,
                               size_t toBitPos,
                               SmallVectorImpl<DataflowPath> &results);

  void markFanIn(Value value, size_t bitPos,
                 SmallVectorImpl<DataflowPath> &results);

  LogicalResult markFanOut(Value fanout, Value start);

  hw::HWModuleOp module;
  Context *ctx;

  std::unique_ptr<circt::igraph::InstancePathCache> instancePathCache;
  // Thread-local debug point factory.
  std::unique_ptr<llvm::ImmutableListFactory<DebugPoint>> debugPointFactory;

  // A map from the value point to the longest paths.
  DenseMap<std::pair<Value, size_t>, SmallVector<DataflowPath>> cachedResults;

  SmallVector<std::pair<Object, DataflowPath>> results;

  DenseMap<Object, SmallVector<DataflowPath>> fanoutResults;

  std::atomic_bool done;
  bool topLevel = false;
};

const LocalVisitor *Context::getAndWaitLocalVisitor(StringAttr name) const {
  const auto *it = localvisitors.find(name);
  if (it == localvisitors.end())
    return nullptr;
  it->second->waitUntilDone();
  return it->second.get();
}

LogicalResult LocalVisitor::markFanOut(Value fanout, Value start) {
  for (size_t i = 0, e = getBitWidth(fanout); i < e; ++i) {
    auto result = getOrComputeResults(start, i);
    if (failed(result))
      return failure();
    for (auto &path : *result) {
      if (auto blockArg = dyn_cast<BlockArgument>(path.value)) {
        // Not closed.
        saveInputPort({}, fanout, i, path.delay, path.history,
                      fromInputPortToFanOut[{blockArg, path.bitPos}]);
      } else {
        // Closed.
        fanoutResults[{{}, fanout, i}].push_back(path);
      }
    }
  }
  return success();
}

LogicalResult
LocalVisitor::markEquivalent(Value from, size_t fromBitPos, Value to,
                             size_t toBitPos,
                             SmallVectorImpl<DataflowPath> &results) {
  auto leader = ec.getOrInsertLeaderValue({to, toBitPos});
  auto &slot = ecMap[leader];
  if (!slot.first)
    slot = {to, toBitPos};

  auto newLeader = ec.unionSets({to, toBitPos}, {from, fromBitPos});
  assert(leader == *newLeader);
  return visitValue(to, toBitPos, results);
}

void DebugPoint::print(llvm::raw_ostream &os) const {
  printImpl(os, getNameImpl(value, bitPos), path, bitPos, delay, comment);
}

struct LongestPathAnalysis::Impl {
  Impl(Operation *module, mlir::AnalysisManager &am)
      : module(module), ctx(isa<mlir::ModuleOp>(module)
                                ? &am.getAnalysis<igraph::InstanceGraph>()
                                : nullptr,
                            false) {}

  Impl(Operation *module, circt::igraph::InstanceGraph *instanceGraph,
       bool storeFlipFlopLongestPath)
      : module(module), ctx(instanceGraph, storeFlipFlopLongestPath) {}

  LogicalResult initializeAndRun(mlir::ModuleOp module);
  LogicalResult initializeAndRun(hw::HWModuleOp module);
  bool isAnalysisAvailable(hw::HWModuleOp module) const;

  void getResultsForFF(SmallVectorImpl<PathResult> &results);

  int64_t getAverageMaxDelay(Value value) const;

  // Return all paths for the given value across module boundaries.
  LogicalResult
  getResults(Value value, size_t bitPos, SmallVectorImpl<PathResult> &results,
             circt::igraph::InstancePathCache *instancePathCache = nullptr,
             llvm::ImmutableListFactory<DebugPoint> *debugPointFactory =
                 nullptr) const;

  LogicalResult getResultsForFF(SmallVectorImpl<PathResult> &results) const;

private:
  LogicalResult getResultsImpl(
      const Object &originalObject, Value value, size_t bitPos,
      SmallVectorImpl<PathResult> &results,
      circt::igraph::InstancePathCache *instancePathCache,
      llvm::ImmutableListFactory<DebugPoint> *debugPointFactory) const;

  Operation *module;
  Context ctx;
  SmallVector<igraph::InstanceGraphNode *> topNodes;
};

LongestPathAnalysis::LongestPathAnalysis(Operation *moduleOp,
                                         mlir::AnalysisManager &am)
    : impl(std::make_unique<Impl>(moduleOp, am)) {
  if (auto module = dyn_cast<mlir::ModuleOp>(moduleOp)) {
    if (failed(impl->initializeAndRun(module)))
      llvm::report_fatal_error("Failed to run longest path analysis");
  } else if (auto hwMod = dyn_cast<hw::HWModuleOp>(moduleOp)) {
    if (failed(impl->initializeAndRun(hwMod)))
      llvm::report_fatal_error("Failed to run longest path analysis");
  } else {
    llvm::report_fatal_error("Analysis scheduled on invalid operation");
  }
}

LogicalResult LocalVisitor::addEdge(Value to, size_t bitPos, int64_t delay,
                                    SmallVectorImpl<DataflowPath> &results) {
  auto result = getOrComputeResults(to, bitPos);
  if (failed(result))
    return failure();
  for (auto &point : *result) {
    auto newPoint = point;
    newPoint.delay += delay;
    results.push_back(newPoint);
  }
  return success();
}

LogicalResult LocalVisitor::visit(aig::AndInverterOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(comb::AndOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(comb::OrOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(comb::XorOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(op, bitPos, results);
}

LogicalResult LocalVisitor::visit(comb::MuxOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  if (failed(addEdge(op.getCond(), 0, 1, results)) ||
      failed(addEdge(op.getTrueValue(), bitPos, 1, results)) ||
      failed(addEdge(op.getFalseValue(), bitPos, 1, results)))
    return failure();
  deduplicatePaths(results);
  return success();
}

LogicalResult LocalVisitor::visit(comb::ExtractOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  assert(getBitWidth(op.getInput()) > bitPos + op.getLowBit());
  return markEquivalent(op, bitPos, op.getInput(), bitPos + op.getLowBit(),
                        results);
}

LogicalResult LocalVisitor::visit(comb::ReplicateOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  return markEquivalent(op, bitPos, op.getInput(),
                        bitPos % getBitWidth(op.getInput()), results);
}

void LocalVisitor::markFanIn(Value value, size_t bitPos,
                             SmallVectorImpl<DataflowPath> &results) {
  results.emplace_back(circt::igraph::InstancePath(), value, bitPos);
}

LogicalResult LocalVisitor::visit(hw::WireOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
  return markEquivalent(op, bitPos, op.getInput(), bitPos, results);
}

LogicalResult LocalVisitor::visit(hw::InstanceOp op, size_t bitPos,
                                  size_t resultNum,
                                  SmallVectorImpl<DataflowPath> &results) {
  auto moduleName = op.getReferencedModuleNameAttr();
  auto value = op->getResult(resultNum);

  // If this is local, then we are done.
  if (!ctx->instanceGraph) {
    markFanIn(value, bitPos, results);
    return success();
  }

  // If an instance graph is available, then we can look up the module. It's an
  // error when the module is not found.
  auto *node = ctx->instanceGraph->lookup(moduleName);
  if (!node || !node->getModule()) {
    op.emitError() << "module " << moduleName << " not found";
    return failure();
  }

  // Otherwise, if the module is not a HWModuleOp, then we should treat it as a
  // black box.
  if (!isa<hw::HWModuleOp>(node->getModule())) {
    markFanIn(value, bitPos, results);
    return success();
  }

  DebugPoint debugPoint({}, value, bitPos, 0, "output port");

  auto *localvisitor = ctx->getAndWaitLocalVisitor(moduleName);
  auto *fanInIt = localvisitor->fromOutputPortToFanIn.find({resultNum, bitPos});

  // It means output is constant, so it's ok.
  if (fanInIt == localvisitor->fromOutputPortToFanIn.end())
    return success();

  const auto &fanins = fanInIt->second;
  // Concat History.
  for (auto &[faninPoint, delayAndHistory] : fanins) {
    auto [path, start, startBitPos] = faninPoint;
    auto [delay, history] = delayAndHistory;
    auto newPath = instancePathCache->prependInstance(op, path);

    // Update the history.
    auto newHistory =
        mapList(debugPointFactory.get(), history, [&](DebugPoint p) {
          p.path = newPath;
          return p;
        });
    newHistory = debugPointFactory->add(debugPoint, newHistory);
    if (auto arg = dyn_cast<BlockArgument>(start)) {
      auto result =
          getOrComputeResults(op->getOperand(arg.getArgNumber()), startBitPos);
      if (failed(result))
        return failure();
      for (auto path : *result) {
        path.delay += delay;
        path.history =
            concatList(debugPointFactory.get(), newHistory, path.history);
        results.push_back(path);
      }
    } else {
      assert((isa<seq::FirRegOp, hw::InstanceOp>(start.getDefiningOp())));

      results.emplace_back(newPath, start, startBitPos, delay, newHistory);
    }
  }

  return success();
}

LogicalResult LocalVisitor::visit(comb::ConcatOp op, size_t bitPos,
                                  SmallVectorImpl<DataflowPath> &results) {
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
                                       SmallVectorImpl<DataflowPath> &results) {
  auto size = op->getNumOperands();
  auto cost = std::max(1u, llvm::Log2_64_Ceil(size));
  // Create edges each operand with cost ceil(log(size)).
  for (auto operand : op->getOperands())
    if (failed(addEdge(operand, bitPos, cost, results)))
      return failure();

  deduplicatePaths(results);
  return success();
}

LogicalResult
LocalVisitor::visitDefault(Operation *op, size_t bitPos,
                           SmallVectorImpl<DataflowPath> &results) {
  // llvm::errs() << "Unknown op: " << *op << "\n";
  return success();
}

LogicalResult LocalVisitor::visit(mlir::BlockArgument arg,
                                  size_t bitPos, // Start.
                                  SmallVectorImpl<DataflowPath> &results) {
  assert(arg.getOwner() == module.getBodyBlock());

  // Record a debug point.
  DebugPoint debug({}, arg, bitPos, 0, "input port");
  auto newHistory = debugPointFactory->add(debug, {});
  assert(arg.getParentBlock() == module.getBodyBlock());

  DataflowPath newPoint({}, arg, bitPos, 0, newHistory);
  results.push_back(newPoint);
  return success();
}

FailureOr<ArrayRef<DataflowPath>>
LocalVisitor::getOrComputeResults(Value value, size_t bitPos) {
  if (ec.contains({value, bitPos})) {
    auto leader = ec.findLeader({value, bitPos});
    // If this is not the leader, then use the leader.
    if (*leader != std::pair(value, bitPos)) {
      auto root = ecMap.at(*leader);
      return getOrComputeResults(root.first, root.second);
    }
  }

  auto it = cachedResults.find({value, bitPos});
  if (it != cachedResults.end())
    return ArrayRef<DataflowPath>(it->second);

  SmallVector<DataflowPath> results;
  if (failed(visitValue(value, bitPos, results)))
    return {};

  // Unique the results.
  deduplicatePaths(results);
  LLVM_DEBUG({
    llvm::errs() << value << "[" << bitPos << "] " << "Found " << results.size()
                 << " paths\n";
    llvm::errs() << "====Paths:\n";
    for (auto &path : results) {
      path.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "====\n";
  });

  auto insertedResult =
      cachedResults.try_emplace({value, bitPos}, std::move(results));
  assert(insertedResult.second);
  return ArrayRef<DataflowPath>(insertedResult.first->second);
}

LogicalResult LocalVisitor::visitValue(Value value, size_t bitPos,
                                       SmallVectorImpl<DataflowPath> &results) {
  LLVM_DEBUG({
    llvm::dbgs() << "Visiting: ";
    llvm::dbgs() << " " << value << "[" << bitPos << "]\n";
  });

  DataflowPath point({}, value, bitPos);
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(value))
    return visit(blockArg, bitPos, results);

  auto *op = value.getDefiningOp();
  auto result =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp,
                aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::MuxOp,
                comb::XorOp, seq::FirRegOp, seq::CompRegOp, hw::ConstantOp,
                hw::WireOp>([&](auto op) { return visit(op, bitPos, results); })
          .Case<hw::InstanceOp>([&](hw::InstanceOp op) {
            return visit(op, bitPos, cast<OpResult>(value).getResultNumber(),
                         results);
          })
          .Default([&](auto op) { return visitDefault(op, bitPos, results); });

  return result;
}

LogicalResult LocalVisitor::initializeAndRun() {
  auto start = std::chrono::high_resolution_clock::now();
  ctx->notifyStart(module.getModuleNameAttr());
  for (auto blockArgument : module.getBodyBlock()->getArguments())
    for (size_t i = 0, e = getBitWidth(blockArgument); i < e; ++i)
      (void)getOrComputeResults(blockArgument, i);

  auto walkResult = module->walk([&](Operation *op) {
    auto result =
        mlir::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<seq::FirRegOp>(
                [&](auto op) { return markFanOut(op, op.getNext()); })
            .Case<aig::AndInverterOp>([&](auto op) {
              // NOTE: Visiting and-inverter is not necessary but useful to
              // reduce recursion depth.
              for (size_t i = 0, e = getBitWidth(op); i < e; ++i)
                if (failed(getOrComputeResults(op, i)))
                  return failure();
              return success();
            })
            .Case<hw::InstanceOp>([&](auto instance) {
              const auto &it = ctx->localvisitors.find(
                  instance.getReferencedModuleNameAttr());

              // If this is a black box, skip it.
              if (it == ctx->localvisitors.end())
                return success();

              // Wait other threads to finish this.
              it->second->waitUntilDone();

              for (const auto &[object, dataflowPath] :
                   it->second->fromInputPortToFanOut) {
                auto [arg, argBitPos] = object;
                for (auto [point, state] : dataflowPath) {
                  auto [instancePath, fanout, fanoutBitPos] = point;
                  assert(isa<BlockArgument>(fanout) ||
                         isa<seq::FirRegOp>(fanout.getDefiningOp()));
                  auto [delay, history] = state;
                  auto newPath = instancePathCache->prependInstance(
                      instance, instancePath);
                  auto computedResults = getOrComputeResults(
                      instance.getOperand(arg.getArgNumber()), argBitPos);
                  if (failed(computedResults))
                    return failure();

                  for (auto &result : *computedResults) {
                    // Path to <newPath, start, startBitPos>
                    auto newHistory = mapList(debugPointFactory.get(), history,
                                              [&](DebugPoint p) {
                                                p.path = newPath;
                                                p.delay += result.delay;
                                                return p;
                                              });
                    if (auto newPort = dyn_cast<BlockArgument>(result.value)) {
                      saveInputPort(
                          newPath, fanout, fanoutBitPos, result.delay + delay,
                          newHistory,
                          fromInputPortToFanOut[{newPort, result.bitPos}]);
                    } else {
                      fanoutResults[{newPath, fanout, fanoutBitPos}]
                          .emplace_back(newPath, result.value, result.bitPos,
                                        result.delay + delay,
                                        concatList(debugPointFactory.get(),
                                                   newHistory, result.history));
                    }
                  }
                }
              }

              return success();
            })
            .Case<hw::OutputOp>([&](auto output) {
              for (OpOperand &operand : output->getOpOperands()) {
                for (size_t i = 0, e = getBitWidth(operand.get()); i < e; ++i) {
                  LLVM_DEBUG(llvm::dbgs() << "Running " << operand.get() << "["
                                          << i << "]\n");
                  auto &recordOutput =
                      fromOutputPortToFanIn[{operand.getOperandNumber(), i}];
                  auto computedResults = getOrComputeResults(operand.get(), i);
                  if (failed(computedResults))
                    return failure();
                  for (const auto &result : *computedResults) {
                    saveInputPort(result.instancePath, result.value,
                                  result.bitPos, result.delay, result.history,
                                  recordOutput);
                  }
                }
              }
              return success();
            })
            .Default([](auto op) { return success(); });
    if (failed(result))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  ctx->notifyEnd(module.getModuleNameAttr());
  llvm::errs() << "[Timing] Done " << module.getModuleNameAttr() << " in "
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::high_resolution_clock::now() - start)
                      .count()
               << "ms\n";
  done.store(true);
  return success();
}

const LocalVisitor *Context::getLocalVisitor(StringAttr name) const {
  auto *it = localvisitors.find(name);
  if (it == localvisitors.end())
    return nullptr;
  assert(it->second->isDone());
  return it->second.get();
}

LogicalResult LongestPathAnalysis::Impl::getResultsImpl(
    const Object &originalObject, Value value, size_t bitPos,
    SmallVectorImpl<PathResult> &results,
    circt::igraph::InstancePathCache *instancePathCache,
    llvm::ImmutableListFactory<DebugPoint> *debugPointFactory) const {
  auto parentHWModule =
      value.getParentRegion()->getParentOfType<hw::HWModuleOp>();
  if (!parentHWModule)
    return mlir::emitError(value.getLoc())
           << "query value is not in a HWModuleOp";
  auto *localvisitor = ctx.getLocalVisitor(parentHWModule.getModuleNameAttr());
  if (!localvisitor)
    return success();

  size_t oldIndex = results.size();
  auto *node = ctx.instanceGraph->lookup(parentHWModule.getModuleNameAttr());
  LLVM_DEBUG({
    llvm::dbgs() << "Running " << parentHWModule.getModuleNameAttr() << " "
                 << value << " " << bitPos << "\n";
  });

  for (auto &path : localvisitor->getResults(value, bitPos)) {
    auto arg = dyn_cast<BlockArgument>(path.value);
    if (!arg || localvisitor->isTopLevel()) {
      // If the value is not a block argument, then we are done.
      results.push_back({originalObject, path, parentHWModule});
      continue;
    }

    auto newObject = originalObject;
    for (auto *inst : node->uses()) {
      auto startIndex = results.size();
      if (instancePathCache)
        newObject.instancePath = instancePathCache->appendInstance(
            originalObject.instancePath, inst->getInstance());

      auto result = getResultsImpl(
          newObject, inst->getInstance()->getOperand(arg.getArgNumber()),
          path.bitPos, results, instancePathCache, debugPointFactory);
      if (failed(result))
        return result;
      for (auto i = startIndex, e = results.size(); i < e; ++i)
        results[i].fanin.delay += path.delay;
    }
  }

  deduplicatePaths(results, oldIndex);
  return success();
}

void LongestPathAnalysis::Impl::getResultsForFF(
    SmallVectorImpl<PathResult> &results) {
  for (auto &[key, localvisitor] : ctx.localvisitors) {
    for (auto &[point, state] : localvisitor->getFanoutResults()) {
      for (auto dataFlow : state)
        results.emplace_back(point, dataFlow, localvisitor->getHWModule());
    }
  }

  for (auto *topNode : topNodes) {
    for (auto &[key, value] :
         ctx.localvisitors[topNode->getModule().getModuleNameAttr()]
             ->getFromInputPortToFanOut()) {
      auto [arg, argBitPos] = key;
      for (auto [point, state] : value) {
        auto [path, start, startBitPos] = point;
        auto [delay, history] = state;
        results.emplace_back(
            Object(path, start, startBitPos),
            DataflowPath({}, arg, argBitPos, delay, history),
            ctx.localvisitors[topNode->getModule().getModuleNameAttr()]
                ->getHWModule());
      }
    }
  }
  std::sort(results.begin(), results.end(), [](PathResult &a, PathResult &b) {
    return a.fanin.delay > b.fanin.delay;
    // Implement sorting. Make sure they are sorted by names as well
    // when tie off.
    if (a.fanin.delay != b.fanin.delay)
      return a.fanin.delay > b.fanin.delay;
    if (a.root != b.root)
      return a.root.getModuleNameAttr().compare(b.root.getModuleNameAttr()) < 0;

    std::string aStr, bStr;
    llvm::raw_string_ostream osA(aStr);
    llvm::raw_string_ostream osB(bStr);
    a.fanin.print(osA);
    b.fanin.print(osB);
    osA.flush();
    osB.flush();
    return aStr.compare(bStr) < 0;
  });
}

LogicalResult
LongestPathAnalysis::Impl::initializeAndRun(hw::HWModuleOp module) {
  auto it =
      ctx.localvisitors.insert({module.getModuleNameAttr(),
                                std::make_unique<LocalVisitor>(module, &ctx)});
  assert(it.second);
  return it.first->second->initializeAndRun();
}

LogicalResult
LongestPathAnalysis::Impl::initializeAndRun(mlir::ModuleOp module) {
  StringRef topName;
  if (auto topNameAttr = module->getAttrOfType<FlatSymbolRefAttr>(
          getTopModuleNameAttrName())) {
    topName = topNameAttr.getAttr().getValue();
    llvm::errs() << "Using top module " << topName
                 << " from module attribute\n";
  }

  topNodes.clear();
  llvm::SetVector<Operation *> visited;
  auto *instanceGraph = ctx.instanceGraph;
  if (!topName.empty()) {
    auto *topNode =
        instanceGraph->lookup(StringAttr::get(module.getContext(), topName));
    if (!topNode || !topNode->getModule() ||
        !isa<hw::HWModuleOp>(topNode->getModule())) {
      module.emitError() << "top module not found in instance graph "
                         << topName;
      return failure();
    }
    topNodes.push_back(topNode);
  } else {
    auto inferredResults = instanceGraph->getInferredTopLevelNodes();
    if (failed(inferredResults))
      return inferredResults;
    for (auto *node : *inferredResults)
      topNodes.push_back(node);
  }

  SmallVector<igraph::InstanceGraphNode *> worklist = topNodes;
  // Get a set of modules that are reachable from the top nodes, excluding
  // bound instances.
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

  SmallVector<hw::HWModuleOp> underHierarchy;
  for (auto *topNode : topNodes) {
    for (auto *node : llvm::post_order(topNode))
      if (node && node->getModule())
        if (auto hwMod = dyn_cast<hw::HWModuleOp>(*node->getModule())) {
          if (visited.contains(hwMod))
            ctx.localvisitors.insert(
                {hwMod.getModuleNameAttr(),
                 std::make_unique<LocalVisitor>(hwMod, &ctx)});
        }

    ctx.localvisitors[topNode->getModule().getModuleNameAttr()]->setTopLevel();
  }

  auto result = mlir::failableParallelForEach(
      module.getContext(), ctx.localvisitors,
      [&](auto &it) { return it.second->initializeAndRun(); });

  if (failed(result))
    return failure();

  return success();
}

void LongestPathAnalysis::getResultsForFF(
    SmallVectorImpl<PathResult> &results) {
  impl->getResultsForFF(results);
}

void PrintLongestPathAnalysisPass::runOnOperation() {
  auto &sym = getAnalysis<mlir::SymbolTable>();
  auto &analysis = getAnalysis<circt::aig::LongestPathAnalysis>();
  SmallVector<PathResult> results;
  analysis.getResultsForFF(results);

  llvm::errs() << "Found " << results.size() << " paths\n";
  if (showTopKPercent == 0)
    return;

  size_t topK = llvm::divideCeil(results.size(), 100) *
                std::max(0, showTopKPercent.getValue());
  llvm::errs() << "Showing top " << topK << " paths\n";
  for (size_t i = 0; i < std::min<size_t>(topK, results.size()); ++i) {
    llvm::errs() << "Path #" << i << ": ";
    results[i].print(llvm::errs());
    llvm::errs() << "\n";
  }
}

bool LongestPathAnalysis::isAnalysisAvaiable(hw::HWModuleOp module) const {
  return impl->isAnalysisAvailable(module);
}

int64_t LongestPathAnalysis::getAverageMaxDelay(Value value) {
  return impl->getAverageMaxDelay(value);
}

bool LongestPathAnalysis::Impl::isAnalysisAvailable(
    hw::HWModuleOp module) const {
  return ctx.localvisitors.find(module.getModuleNameAttr()) !=
         ctx.localvisitors.end();
}

int64_t LongestPathAnalysis::Impl::getAverageMaxDelay(Value value) const {
  auto parentHWModule =
      value.getParentRegion()->getParentOfType<hw::HWModuleOp>();
  if (!parentHWModule)
    return 0;
  SmallVector<PathResult> results;
  size_t bitWidth = getBitWidth(value);
  if (bitWidth == 0)
    return 0;
  int64_t totalDelay = 0; 
  for (size_t i = 0; i < bitWidth; ++i) {
    results.clear();
    auto result = getResults(value, i, results);
    if (failed(result))
      return 0;

    int64_t maxDelay = 0;
    for (auto &path : results)
      maxDelay = std::max(maxDelay, path.fanin.delay);
    totalDelay += maxDelay;
  }
  return llvm::divideCeil(totalDelay, bitWidth);
}