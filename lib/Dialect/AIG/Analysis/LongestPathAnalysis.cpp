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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/InstanceGraph.h"
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
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Mutex.h"
#include <memory>
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
  using PrintLongestPathAnalysisBase::outputJSONFile;
  using PrintLongestPathAnalysisBase::PrintLongestPathAnalysisBase;
  using PrintLongestPathAnalysisBase::printSummary;
  using PrintLongestPathAnalysisBase::topModuleName;
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

struct Impl {
  LogicalResult run(mlir::ModuleOp module, StringRef topModuleName);

  struct ThreadWorker {};
  struct Context;
  struct Object {
    circt::igraph::InstancePath instancePath;
    Value value;
    size_t bitPos;
    Object(circt::igraph::InstancePath path, Value value, size_t bitPos)
        : instancePath(path), value(value), bitPos(bitPos) {}
    Object() = default;
    void print(llvm::raw_ostream &os) const {
      os << "Object(";
      instancePath.print(os);
      os << "." << getNameImpl(value, bitPos).getValue() << "[" << bitPos
         << "])";
    }
  };

  struct PathResult {
    Object fanout;
    DataflowPath fanin;
    hw::HWModuleOp root;
    PathResult(Object fanout, DataflowPath fanin, hw::HWModuleOp root)
        : fanout(fanout), fanin(fanin), root(root) {}

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
        llvm::interleaveComma(fanin.history, os, [&](DebugPoint p) {
          printImpl(os, getNameImpl(p.value, p.bitPos), p.path, p.bitPos,
                    p.delay, p.comment);
        });
        os << "]";
      }
      os << ")";
    }
  };

  struct Runner {
    Runner(hw::HWModuleOp module, igraph::InstanceGraph *instanceGraph,
           Context *ctx)
        : module(module), instanceGraph(instanceGraph), ctx(ctx) {
      debugPointFactory =
          std::make_unique<llvm::ImmutableListFactory<DebugPoint>>();
      instancePathCache =
          std::make_unique<circt::igraph::InstancePathCache>(*instanceGraph);

      done = false;
    }

    hw::HWModuleOp module;
    LogicalResult run();
    int64_t size = 0;

    std::pair<Value, size_t> findLeader(Value value, size_t bitpos) {
      return ec.getLeaderValue({value, bitpos});
    }

    void waitForDone() {
      while (!done.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    SmallVector<DataflowPath> startingPoints;
    std::priority_queue<DataflowPath, std::vector<DataflowPath>,
                        std::greater<DataflowPath>>
        savedFanIn;
    ArrayRef<DataflowPath> getOrCompute(Value value, size_t bitPos);

    // A map from the value point to the longest paths.
    DenseMap<std::pair<Value, size_t>, SmallVector<DataflowPath>> cachedResults;

    SmallVector<std::pair<Object, DataflowPath>> results;
    Context *ctx;

    DataflowPath longestPath;
    using SavedPoint =
        llvm::MapVector<std::tuple<circt::igraph::InstancePath, Value, size_t>,
                        std::pair<int64_t, llvm::ImmutableList<DebugPoint>>>;

    llvm::MapVector<std::pair<BlockArgument, size_t>, SavedPoint>
        fromInputPortToFanOut;
    llvm::MapVector<std::tuple<size_t, size_t>, SavedPoint>
        fromOutputPortToFanIn;

  private:
    DataflowPath currentStartingPoint;
    DenseMap<std::tuple<circt::igraph::InstancePath, Value, size_t>,
             std::pair<int64_t, bool>>
        visited;

    // llvm::DenseMap<std::pair<Value, size_t>, SmallVector<DataflowPath>>
    //     valueCache;

  public:
    llvm::DenseMap<std::tuple<igraph::InstancePath, Value, size_t>,
                   SmallVector<DataflowPath>>
        closedResults;

    void insertPathForFanout(igraph::InstancePath path, Value value,
                             size_t bitPos, const DataflowPath &dataflowPath);

    LogicalResult run(Value value, size_t bitPos,
                      SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, mlir::BlockArgument argument,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, hw::InstanceOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, hw::WireOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, comb::ConcatOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, comb::ExtractOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, comb::ReplicateOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, aig::AndInverterOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, comb::AndOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, comb::XorOp op,
                        SmallVectorImpl<DataflowPath> &results);
    LogicalResult visit(const DataflowPath &point, comb::OrOp op,
                        SmallVectorImpl<DataflowPath> &results);

    // Pass through.
    LogicalResult visit(const DataflowPath &point, seq::FirRegOp op,
                        SmallVectorImpl<DataflowPath> &results) {
      markFanIn(point, results);
      return success();
    }
    LogicalResult visit(const DataflowPath &point, seq::CompRegOp op,
                        SmallVectorImpl<DataflowPath> &results) {
      markFanIn(point, results);
      return success();
    }
    LogicalResult visitDefault(const DataflowPath &point, Operation *op,
                               SmallVectorImpl<DataflowPath> &results);

    LogicalResult addLogicOp(const DataflowPath &point, Operation *op,
                             SmallVectorImpl<DataflowPath> &results);
    void addEdge(Value to, size_t toBitPos, int64_t delay,
                 SmallVectorImpl<DataflowPath> &results);

    // This tracks equivalence classes of values. If an operation is purely a
    // data movement, then we can treat the output as equivalent to the input.
    llvm::EquivalenceClasses<std::pair<Value, size_t>> ec;
    DenseMap<std::pair<Value, size_t>, std::pair<Value, size_t>> ecMap;

    void markEquivalent(Value from, size_t fromBitPos, Value to,
                        size_t toBitPos,
                        SmallVectorImpl<DataflowPath> &results) {
      auto leader = ec.getOrInsertLeaderValue({to, toBitPos});
      auto &slot = ecMap[leader];
      if (!slot.first)
        slot = {to, toBitPos};

      auto newLeader = ec.unionSets({to, toBitPos}, {from, fromBitPos});
      assert(leader == *newLeader);
      (void)run(to, toBitPos, results);
      // addEdge(to, toBitPos, 0, results);
    }

    void markFanIn(const DataflowPath &point,
                   SmallVectorImpl<DataflowPath> &results);

    igraph::InstanceGraph *instanceGraph;

    std::unique_ptr<circt::igraph::InstancePathCache> instancePathCache;

    // Thread-local debug point factory.
    std::unique_ptr<llvm::ImmutableListFactory<DebugPoint>> debugPointFactory;
    std::atomic_bool done;
    bool searchingOutput = false;
  };

  struct Context {
    Context(size_t numAll) : numAll(numAll) {};
    llvm::sys::SmartMutex<true> mutex;
    std::atomic<size_t> numAllProcessed = 0;
    const size_t numAll;
    void add(size_t t, std::optional<DataflowPath> point,
             DataflowPath longest) {
      std::lock_guard<llvm::sys::SmartMutex<true>> lock(mutex);
      numAllProcessed += t;
      llvm::errs() << "[Timing] Processed " << numAllProcessed << "/" << numAll
                   << "(" << double(numAllProcessed) * 100 / numAll << "%) ";

      if (point) {
        llvm::errs() << "Current: ";
        point->print(llvm::errs());
      }
      if (longest.delay == -1)
        return;
      llvm::errs() << " Longest: ";
      longest.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::SetVector<StringAttr> running;

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

    llvm::MapVector<StringAttr, std::unique_ptr<Runner>> runners;
  };
};

void Impl::Runner::addEdge(Value to, size_t bitPos, int64_t delay,
                           SmallVectorImpl<DataflowPath> &results) {
  for (auto &point : getOrCompute(to, bitPos)) {
    auto newPoint = point;
    newPoint.delay += delay;
    results.push_back(newPoint);
  }
}

LogicalResult Impl::Runner::visit(const DataflowPath &point,
                                  aig::AndInverterOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(point, op, results);
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, comb::AndOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(point, op, results);
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, comb::OrOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(point, op, results);
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, comb::XorOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  return addLogicOp(point, op, results);
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, comb::ExtractOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  assert(getBitWidth(op.getInput()) > point.bitPos + op.getLowBit());
  markEquivalent(point.value, point.bitPos, op.getInput(),
                 point.bitPos + op.getLowBit(), results);
  return success();
}

LogicalResult Impl::Runner::visit(const DataflowPath &point,
                                  comb::ReplicateOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  markEquivalent(point.value, point.bitPos, op.getInput(),
                 point.bitPos % getBitWidth(op.getInput()), results);
  return success();
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

void Impl::Runner::markFanIn(const DataflowPath &point,
                             SmallVectorImpl<DataflowPath> &results) {
  results.push_back(point);
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, hw::WireOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  markEquivalent(point.value, point.bitPos, op.getInput(), point.bitPos,
                 results);
  return success();
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, hw::InstanceOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  auto resultNum = cast<mlir::OpResult>(point.value).getResultNumber();
  auto moduleName = op.getReferencedModuleNameAttr();
  auto *node = instanceGraph->lookup(moduleName);
  if (!node || !node->getModule()) {
    op.emitError() << "module " << moduleName << " not found";
    return failure();
  }
  // Consider black box results as fanin.
  if (!isa<hw::HWModuleOp>(node->getModule())) {
    markFanIn(point, results);
    return success();
  }

  // auto moduleOp = node->getModule<hw::HWModuleOp>();
  DebugPoint debugPoint(point.instancePath, point.value, point.bitPos,
                        point.delay, "output port");

  // Point newPoint = point;
  // newPoint.history = debugPointFactory->add(debugPoint, newPoint.history);

  // newPoint.path = instancePathCache->appendInstance(point.path, op);
  // newPoint.value =
  //     moduleOp.getBodyBlock()->getTerminator()->getOperand(resultNum);

  auto *it = ctx->runners.find(moduleName);
  assert(it != ctx->runners.end());
  it->second->waitForDone();

  auto *fanInIt =
      it->second->fromOutputPortToFanIn.find({resultNum, point.bitPos});

  // It means output is constant, so it's ok.
  if (fanInIt == it->second->fromOutputPortToFanIn.end())
    return success();

  // Output.
  // %a = hw.instance @A(b.c.d)

  assert(point.instancePath.empty() && "must be empty?");
  // Concat History.
  for (auto &[faninPoint, state] : fanInIt->second) {
    auto [path, start, startBitPos] = faninPoint;
    auto [delay, history] = state;
    auto newPath = instancePathCache->prependInstance(op, path);

    // Update the history.
    auto newHistory = concatList(debugPointFactory.get(),
                                 mapList(debugPointFactory.get(), history,
                                         [&](DebugPoint p) {
                                           p.path = newPath;
                                           p.delay += point.delay;
                                           return p;
                                         }),
                                 point.history);

    if (auto arg = dyn_cast<BlockArgument>(start)) {
      for (auto path :
           getOrCompute(op->getOperand(arg.getArgNumber()), startBitPos)) {
        path.delay += point.delay;
        path.history =
            concatList(debugPointFactory.get(), newHistory, path.history);
        results.push_back(path);
      }
    } else {
      assert((isa<seq::FirRegOp, hw::InstanceOp>(start.getDefiningOp())));

      results.push_back(DataflowPath(start, startBitPos, newPath,
                                     delay + point.delay, newHistory));
    }
  }

  return success();
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, comb::ConcatOp op,
                                  SmallVectorImpl<DataflowPath> &results) {
  // Find bit pos.
  DataflowPath newPoint = point;
  for (auto operand : llvm::reverse(op.getInputs())) {
    auto size = getBitWidth(operand);
    if (newPoint.bitPos >= size) {
      newPoint.bitPos -= size;
      continue;
    }
    markEquivalent(point.value, point.bitPos, operand, newPoint.bitPos,
                   results);
    return success();
  }
  assert(false);
  return failure();
}

LogicalResult Impl::Runner::addLogicOp(const DataflowPath &point, Operation *op,
                                       SmallVectorImpl<DataflowPath> &results) {
  auto size = op->getNumOperands();
  // Create edges each operand with cost ceil(log(size)).
  for (auto operand : op->getOperands())
    addEdge(operand, point.bitPos, llvm::Log2_64_Ceil(size), results);

  llvm::sort(results, std::greater<DataflowPath>());
  results.erase(std::unique(results.begin(), results.end(),
                            [](const DataflowPath &a, const DataflowPath &b) {
                              return a.instancePath == b.instancePath &&
                                     a.value == b.value && a.bitPos == b.bitPos;
                            }),
                results.end());

  return success();
}

LogicalResult
Impl::Runner::visitDefault(const DataflowPath &point, Operation *op,
                           SmallVectorImpl<DataflowPath> &results) {
  return success();
}

LogicalResult Impl::Runner::visit(const DataflowPath &point, // Start.
                                  mlir::BlockArgument arg,
                                  SmallVectorImpl<DataflowPath> &results) {
  auto path = point.instancePath;
  assert(arg.getOwner() == module.getBodyBlock() && path.size() == 0);

  // Record a debug point.
  DebugPoint debug(path, point.value, point.bitPos, point.delay, "input port");
  auto newHistory = debugPointFactory->add(debug, point.history);
  assert(arg.getParentBlock() == module.getBodyBlock());

  DataflowPath newPoint = point;
  newPoint.history = newHistory;

  results.push_back(newPoint);
  return success();
}

static void uniqifyPaths(SmallVectorImpl<DataflowPath> &results) {
  // Take only maximum for each path destination.
  DenseMap<std::tuple<circt::igraph::InstancePath, Value, size_t>, size_t>
      saved;
  for (auto [i, path] : llvm::enumerate(results)) {
    auto &slot = saved[{path.instancePath, path.value, path.bitPos}];
    if (slot == 0) {
      slot = i + 1;
      continue;
    }
    if (results[slot - 1].delay < path.delay)
      results[slot - 1] = path;
  }

  results.resize(saved.size());
}

ArrayRef<DataflowPath> Impl::Runner::getOrCompute(Value value, size_t bitPos) {
  if (ec.contains({value, bitPos})) {
    auto leader = ec.findLeader({value, bitPos});
    // If this is not the leader, then use the leader.
    if (*leader != std::pair(value, bitPos)) {
      auto root = ecMap.at(*leader);
      return getOrCompute(root.first, root.second);
    }
  }

  auto it = cachedResults.find({value, bitPos});
  if (it != cachedResults.end())
    return it->second;

  SmallVector<DataflowPath> results;
  if (failed(run(value, bitPos, results)))
    return {};

  uniqifyPaths(results);
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
  return insertedResult.first->second;
}

LogicalResult Impl::Runner::run(Value value, size_t bitPos,
                                SmallVectorImpl<DataflowPath> &results) {
  LLVM_DEBUG({
    llvm::dbgs() << "Visiting: ";
    llvm::dbgs() << " " << value << "[" << bitPos << "]\n";
  });

  DataflowPath point(value, bitPos, {});
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(value))
    return visit(point, blockArg, results);

  auto *op = value.getDefiningOp();
  auto result =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<hw::InstanceOp, comb::ConcatOp, comb::ExtractOp,
                comb::ReplicateOp, aig::AndInverterOp, comb::AndOp, comb::OrOp,
                comb::XorOp, seq::FirRegOp, seq::CompRegOp>(
              [&](auto op) { return visit(point, op, results); })
          .Default([&](auto op) { return visitDefault(point, op, results); });

  return result;
}

LogicalResult Impl::Runner::run() {
  // Show elapsed time.
  auto start = std::chrono::steady_clock::now();
  ctx->notifyStart(module.getModuleNameAttr());
  size_t e = startingPoints.size();
  auto result = module->walk([&](Operation *op) {
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      for (size_t i = 0, e = getBitWidth(reg); i < e; ++i) {
        auto result = getOrCompute(reg.getNext(), i);
        for (auto &path : result) {
          if (auto blockArg = dyn_cast<BlockArgument>(path.value)) {
            // Not closed.
            auto &slot =
                fromInputPortToFanOut[{blockArg, path.bitPos}][{{}, reg, i}];
            if (slot.first >= path.delay)
              continue;
            slot = {path.delay, path.history};
          } else {
            // Closed.
            closedResults[{{}, reg, i}].push_back(path);
          }
        }
      }
    }

    if (auto aig = dyn_cast<aig::AndInverterOp>(op)) {
      for (size_t i = 0, e = getBitWidth(aig); i < e; ++i)
        (void)getOrCompute(aig, i);
    }

    if (auto instance = dyn_cast<hw::InstanceOp>(op)) {
      const auto &it =
          ctx->runners.find(instance.getReferencedModuleNameAttr());

      // If this is a black box, skip it.
      if (it == ctx->runners.end())
        return WalkResult::advance();

      // Wait other threads to finish this.
      it->second->waitForDone();

      for (const auto &[object, dataflowPath] :
           it->second->fromInputPortToFanOut) {
        auto [arg, argBitPos] = object;
        for (auto [point, state] : dataflowPath) {
          auto [instancePath, fanout, fanoutBitPos] = point;
          assert(isa<BlockArgument>(fanout) ||
                 isa<seq::FirRegOp>(fanout.getDefiningOp()));
          auto [delay, history] = state;
          auto newPath =
              instancePathCache->prependInstance(instance, instancePath);

          for (auto &result : getOrCompute(
                   instance.getOperand(arg.getArgNumber()), argBitPos)) {
            // Path to <newPath, start, startBitPos>
            auto newHistory =
                mapList(debugPointFactory.get(), history, [&](DebugPoint p) {
                  p.path = newPath;
                  p.delay += result.delay;
                  return p;
                });
            if (auto newPort = dyn_cast<BlockArgument>(result.value)) {
              auto &slot =
                  fromInputPortToFanOut[{newPort, result.bitPos}]
                                       [{newPath, fanout, fanoutBitPos}];
              if (slot.first >= result.delay + delay)
                continue;
              slot = {result.delay + delay, newHistory};
            } else {
              closedResults[{newPath, fanout, fanoutBitPos}].emplace_back(
                  newPath, result.value, result.bitPos, result.delay + delay,
                  concatList(debugPointFactory.get(), newHistory,
                             result.history));
            }
          }
        }
      }
    }
    if (auto output = dyn_cast<hw::OutputOp>(op)) {
      searchingOutput = true;
      for (OpOperand &operand : output->getOpOperands()) {
        for (size_t i = 0, e = getBitWidth(operand.get()); i < e; ++i) {
          size_t oldResultIdx = results.size();
          LLVM_DEBUG(llvm::dbgs()
                     << "Running " << operand.get() << "[" << i << "]\n");
          auto results = getOrCompute(operand.get(), i);
          auto &recordOutput =
              fromOutputPortToFanIn[{operand.getOperandNumber(), i}];
          for (const auto &result : results) {
            auto &resultSlot = recordOutput[{result.instancePath, result.value,
                                             result.bitPos}];
            if (resultSlot.first >= result.delay)
              continue;
            resultSlot = {result.delay, result.history};
          }
        }
      }
      searchingOutput = false;
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  auto end = std::chrono::steady_clock::now();
  {
    std::lock_guard<llvm::sys::SmartMutex<true>> lock(ctx->mutex);
    llvm::errs() << "[Timing] Done " << module.getModuleNameAttr() << "\n";
    llvm::errs()
        << "[Timing] Done in "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << "s\n";

    if (longestPath.delay != -1) {
      llvm::errs() << "[Timing] Longest path: ";
      longestPath.print(llvm::errs());
      llvm::errs() << "\n";
    }
  }
  ctx->notifyEnd(module.getModuleNameAttr());

  cachedResults.clear();

  done.store(true);
  return success();
}

LogicalResult Impl::run(mlir::ModuleOp top, StringRef topName) {
  igraph::InstanceGraph instanceGraph(top);
  igraph::InstancePathCache instancePathCache(instanceGraph);
  // 1. Enumerate all instance paths and registers
  auto *topNode =
      instanceGraph.lookup(StringAttr::get(top.getContext(), topName));
  if (!topNode || !topNode->getModule() ||
      !isa<hw::HWModuleOp>(topNode->getModule())) {
    top.emitError() << "top module not found in instance graph " << topName;
    return failure();
  }
  SmallVector<circt::igraph::InstancePath> instancePaths;
  SmallVector<igraph::InstanceGraphNode *> worklist;
  llvm::SetVector<Operation *> visited;
  worklist.push_back(topNode);
  Context ctx(1);
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
  for (auto *node : llvm::post_order(topNode))
    if (node && node->getModule())
      if (auto hwMod = dyn_cast<hw::HWModuleOp>(*node->getModule())) {
        if (visited.contains(hwMod))
          ctx.runners.insert(
              {hwMod.getModuleNameAttr(),
               std::make_unique<Runner>(hwMod, &instanceGraph, &ctx)});
      }

  // std::vector<Runner> runners;
  // for (size_t i = 0, e = top.getContext()->getNumThreads(); i < e;
  // ++i)
  //   runners.emplace_back(&instanceGraph, &ctx);

  // Calculate bucket size with ceiling division to ensure all points
  // are distributed auto bucketSize =
  //     (allStartPoints.size() + runners.size() - 1) / runners.size();
  // for (auto [i, start] : llvm::enumerate(allStartPoints)) {
  //   size_t group = std::min(i / bucketSize, runners.size() - 1);
  //   runners[group].startingPoints.push_back(start);
  // }
  // for (auto [i, start] : llvm::enumerate(allStartPoints)) {
  //   // size_t group = std::min(i / bucketSize, runners.size() - 1);
  //   runners[i % runners.size()].startingPoints.push_back(start);
  // }

  // NOTE: Reverse order to perform bottom-up traversal.

  auto result =
      mlir::failableParallelForEach(top.getContext(), ctx.runners,
                                    [&](auto &it) { return it.second->run(); });
  // for (auto &runner : runners)
  //   runner.run();

  if (failed(result))
    return failure();

  SmallVector<PathResult> results;
  // TODO: Sort in multithread and merge instead of sorting here.
  for (auto &[key, runner] : ctx.runners) {
    for (auto &[point, state] : runner->closedResults) {
      auto [path, start, startBitPos] = point;
      for (auto dataFlow : state)
        results.emplace_back(Object(path, start, startBitPos), dataFlow,
                             runner->module);
    }
  }

  for (auto &[key, value] :
       ctx.runners[topNode->getModule().getModuleNameAttr()]
           ->fromInputPortToFanOut) {
    auto [arg, argBitPos] = key;
    for (auto [point, state] : value) {
      auto [path, start, startBitPos] = point;
      auto [delay, history] = state;
      results.emplace_back(
          Object(path, start, startBitPos),
          DataflowPath(arg, argBitPos, {}, delay, history),
          ctx.runners[topNode->getModule().getModuleNameAttr()]->module);
    }
  }

  llvm::sort(results, [](const auto &a, const auto &b) {
    return a.fanin.delay > b.fanin.delay;
  });

  llvm::errs() << "Found " << results.size() << " paths\n";
  llvm::errs() << "Top 10 longest paths:\n";
  for (size_t i = 0; i < std::min<size_t>(100, results.size()); ++i) {
    results[i].print(llvm::errs());
    llvm::errs() << "\n";
  }

  return success();
}

void PrintLongestPathAnalysisPass::runOnOperation() {
  if (topModuleName.empty()) {
    getOperation().emitError()
        << "'top-name' option is required for PrintLongestPathAnalysis";
    return signalPassFailure();
  }

  Impl impl;
  if (failed(impl.run(getOperation(), topModuleName)))
    return signalPassFailure();
}
