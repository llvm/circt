//===- ResourceUsageAnalysis.cpp - resource usage analysis ---------*- C++
//-*-===//
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
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Mutex.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mutex>

#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <queue>

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

void Point::print(llvm::raw_ostream &os) const {
  os << "Point(";
  path.print(os);
  os << "." << getNameImpl(value, bitPos).getValue() << "[" << bitPos << "] @ "
     << delay << ")";
}

// struct Output {
//   size_t bitPos;
//   circt::igraph::InstancePath path;
//   Value value;
//   StringAttr comment;
//
//   Output(Value value, size_t bitPos, circt::igraph::InstancePath path)
//       : value(value), bitPos(bitPos), path(path) {}
//   Output() = default;
//   void print(llvm::raw_ostream &os) const {
//     os << "Output(";
//     path.print(os);
//     os << "." << getNameImpl(value, bitPos).getValue() << "[" << bitPos <<
//     "]"
//        << ")";
//   }
// };

struct Impl {
  LogicalResult run(mlir::ModuleOp module, StringRef topModuleName);
  struct Context {
    Context(size_t numAll) : numAll(numAll){};
    llvm::sys::SmartMutex<true> mutex;
    std::atomic<size_t> numAllProcessed = 0;
    const size_t numAll;
    void add(size_t t, std::optional<Point> point, Point longest) {
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
  };

  struct Runner {
    Runner(igraph::InstanceGraph *instanceGraph, Context *ctx)
        : instanceGraph(instanceGraph), ctx(ctx) {
      debugPointFactory =
          std::make_unique<llvm::ImmutableListFactory<DebugPoint>>();
      instancePathCache =
          std::make_unique<circt::igraph::InstancePathCache>(*instanceGraph);
    }
    LogicalResult run();
    Runner() = delete;

    SmallVector<Point> startingPoints;
    llvm::PriorityQueue<Point, std::vector<Point>, std::less<Point>> savedFanIn;

    SmallVector<std::pair<Point, Point>> results;
    Context *ctx;

    Point longestPath;

  private:
    Point currentStartingPoint;
    DenseMap<std::tuple<circt::igraph::InstancePath, Value, size_t>, int64_t>
        visited;
    llvm::PriorityQueue<Point, std::vector<Point>, std::greater<Point>>
        worklist;

    DenseMap<
        std::tuple<circt::igraph::InstancePath, Value, size_t>,
        llvm::MapVector<std::tuple<circt::igraph::InstancePath, Value, size_t>,
                        std::pair<int64_t, llvm::ImmutableList<DebugPoint>>>>
        cache;
    bool useCache(const Point &point);
    LogicalResult run(Point point);

    LogicalResult visit(const Point &point, mlir::BlockArgument argument);
    LogicalResult visit(const Point &point, hw::InstanceOp op);
    LogicalResult visit(const Point &point, hw::WireOp op);
    LogicalResult visit(const Point &point, comb::ConcatOp op);
    LogicalResult visit(const Point &point, comb::ExtractOp op);
    LogicalResult visit(const Point &point, comb::ReplicateOp op);
    LogicalResult visit(const Point &point, aig::AndInverterOp op);
    // Pass through.
    LogicalResult visit(const Point &point, seq::FirRegOp op) {
      markFanIn(point);
      return success();
    }
    LogicalResult visit(const Point &point, seq::CompRegOp op) {
      markFanIn(point);
      return success();
    }
    LogicalResult visitDefault(const Point &point, Operation *op);

    LogicalResult addLogicOp(const Point &point, Operation *op);
    void addEdge(Point from, int64_t delay, Value to);
    void addEdge(Point from, int64_t delay, Value to, size_t bitPos);
    void markFanIn(const Point &point);
    void addToWorklist(Point point);
    LogicalResult setFanOut(Point point);

    igraph::InstanceGraph *instanceGraph;

    std::unique_ptr<circt::igraph::InstancePathCache> instancePathCache;

    // Thread-local debug point factory.
    std::unique_ptr<llvm::ImmutableListFactory<DebugPoint>> debugPointFactory;
  };
};

LogicalResult Impl::Runner::setFanOut(Point output) {

  currentStartingPoint = output;
  return TypeSwitch<Operation *, LogicalResult>(output.value.getDefiningOp())
      .Case<seq::FirRegOp>([&](seq::FirRegOp op) {
        // TODO: Add dependency to reset line.
        Point point(op.getNext(), output.bitPos, output.path, 0);
        addToWorklist(point);
        return success();
      })
      .Default([&](auto) { return failure(); });
}

void Impl::Runner::addToWorklist(Point point) {
  auto it = visited.find({point.path, point.value, point.bitPos});
  if (it != visited.end() && it->second >= point.delay)
    return;
  visited[{point.path, point.value, point.bitPos}] = point.delay;
  worklist.push(point);
}

void Impl::Runner::addEdge(Point from, int64_t delay, Value to) {
  from.value = to;
  from.delay += delay;
  addToWorklist(from);
}

void Impl::Runner::addEdge(Point from, int64_t delay, Value to, size_t bitPos) {
  from.value = to;
  from.delay += delay;
  from.bitPos = bitPos;
  addToWorklist(from);
}

LogicalResult Impl::Runner::visit(const Point &point, aig::AndInverterOp op) {
  return addLogicOp(point, op);
}

LogicalResult Impl::Runner::visit(const Point &point, comb::ExtractOp op) {
  assert(getBitWidth(op.getInput()) > point.bitPos + op.getLowBit());
  addEdge(point, 0, op.getInput(), point.bitPos + op.getLowBit());
  return success();
}

LogicalResult Impl::Runner::visit(const Point &point, comb::ReplicateOp op) {
  addEdge(point, 0, op.getInput(), point.bitPos % getBitWidth(op.getInput()));
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

bool Impl::Runner::useCache(const Point &point) {
  auto it = cache.find({point.path, point.value, point.bitPos});
  if (it == cache.end())
    return false;
  auto &m = it->second;
  for (auto &[key, value] : m) {
    auto history = value.second;
    history = mapList(debugPointFactory.get(), history, [&](DebugPoint p) {
      return DebugPoint(p.path, p.value, p.bitPos, p.delay + point.delay,
                        p.comment);
    });

    Point newPoint(std::get<1>(key), std::get<2>(key), std::get<0>(key),
                   value.first + point.delay, history);
    addToWorklist(newPoint);
  }
  return true;
}

void Impl::Runner::markFanIn(const Point &point) {
  auto currentHistory = point.history;
  while (!currentHistory.isEmpty()) {
    auto head = currentHistory.getHead();
    currentHistory = currentHistory.getTail();
    auto &m = cache[{head.path, head.value, head.bitPos}];
    if (!m.count({point.path, point.value, point.bitPos})) {
      m[{point.path, point.value, point.bitPos}] = {
          point.delay - head.delay,
          mapList(debugPointFactory.get(), currentHistory, [&](DebugPoint p) {
            return DebugPoint(p.path, p.value, p.bitPos, p.delay - head.delay,
                              p.comment);
          })};
    }
  }
  if (point.delay > longestPath.delay) {
    longestPath = point;
  }

  savedFanIn.push(point);
  if (savedFanIn.size() > 5) {
    savedFanIn.pop();
  }
}

LogicalResult Impl::Runner::visit(const Point &point, hw::WireOp op) {
  addEdge(point, 0, op.getInput());
  return success();
}

LogicalResult Impl::Runner::visit(const Point &point, hw::InstanceOp op) {
  if (useCache(point))
    return success();
  auto resultNum = cast<mlir::OpResult>(point.value).getResultNumber();
  auto moduleName = op.getReferencedModuleNameAttr();
  auto *node = instanceGraph->lookup(moduleName);
  if (!node || !node->getModule()) {
    op.emitError() << "module " << moduleName << " not found";
    return failure();
  }
  // Consider black box results as fanin.
  if (!isa<hw::HWModuleOp>(node->getModule())) {
    markFanIn(point);
    return success();
  }

  auto moduleOp = node->getModule<hw::HWModuleOp>();
  if (!moduleOp) {
    markFanIn(point);
    return success();
  }

  DebugPoint debugPoint(point.path, point.value, point.bitPos, point.delay,
                        "output port");

  Point newPoint = point;
  newPoint.history = debugPointFactory->add(debugPoint, newPoint.history);

  newPoint.path = instancePathCache->appendInstance(point.path, op);
  newPoint.value =
      moduleOp.getBodyBlock()->getTerminator()->getOperand(resultNum);
  addToWorklist(newPoint);
  return success();
}

LogicalResult Impl::Runner::visit(const Point &point, comb::ConcatOp op) {
  // Find bit pos.
  Point newPoint = point;
  for (auto operand : llvm::reverse(op.getInputs())) {
    auto size = getBitWidth(operand);
    if (newPoint.bitPos >= size) {
      newPoint.bitPos -= size;
      continue;
    }
    newPoint.value = operand;
    addToWorklist(newPoint);
    return success();
  }
  assert(false);
  return failure();
}

LogicalResult Impl::Runner::addLogicOp(const Point &point, Operation *op) {
  auto size = op->getNumOperands();
  // Create edges each operand with cost ceil(log(size)).
  for (auto operand : op->getOperands())
    addEdge(point, llvm::Log2_64_Ceil(size), operand);
  return success();
}

LogicalResult Impl::Runner::visitDefault(const Point &point, Operation *op) {
  return success();
}

LogicalResult Impl::Runner::visit(const Point &point, mlir::BlockArgument arg) {
  if (useCache(point))
    return success();

  auto path = point.path;
  // Top module.
  if (path.size() == 0) {
    markFanIn(point);
    return success();
  }

  auto instance = path.leaf();
  if (instance->getNumOperands() <= arg.getArgNumber()) {
    llvm::dbgs() << instance << "\n";
    llvm::dbgs() << *arg.getParentBlock()->getParentOp() << "\n";
  }
  auto newPoint = point;
  newPoint.path = path.dropBack();
  newPoint.value = instance->getOperand(arg.getArgNumber());

  // Record a debug point.
  DebugPoint debug(path, point.value, point.bitPos, point.delay, "input port");
  newPoint.history = debugPointFactory->add(debug, point.history);

  addToWorklist(newPoint);
  return success();
}

LogicalResult Impl::Runner::run(Point top) {
  assert(worklist.empty());
  if (failed(setFanOut(top)))
    return failure();

  while (!worklist.empty()) {
    auto current = worklist.top();
    LLVM_DEBUG({
      llvm::dbgs() << "Visiting: ";
      current.path.print(llvm::dbgs());
      llvm::dbgs() << " " << current.value << "[" << current.bitPos << "] @ "
                   << current.delay << "\n";
    });

    worklist.pop();
    auto value = current.value;
    if (visited[{current.path, value, current.bitPos}] > current.delay)
      continue;

    if (auto blockArg = dyn_cast<mlir::BlockArgument>(value)) {
      if (failed(visit(current, blockArg)))
        return failure();
      continue;
    }

    auto *op = value.getDefiningOp();
    auto result =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<hw::InstanceOp, comb::ConcatOp, comb::ExtractOp,
                  comb::ReplicateOp, aig::AndInverterOp, seq::FirRegOp,
                  seq::CompRegOp>([&](auto op) { return visit(current, op); })
            .Default([&](auto op) { return visitDefault(current, op); });
    if (failed(result))
      return failure();
  }

  visited.clear();

  while (!savedFanIn.empty()) {
    auto point = savedFanIn.top();
    savedFanIn.pop();
    results.push_back({currentStartingPoint, point});
  }
  return success();
}

LogicalResult Impl::Runner::run() {
  size_t e = startingPoints.size();
  size_t reportSize = (e + 19) / 20;
  size_t index = 0;
  for (auto &point : startingPoints) {
    index++;
    if (failed(run(point)))
      return failure();

    if (reportSize == index) {
      ctx->add(index, point, longestPath);
      index = 0;
    }
  }
  ctx->add(index, {}, longestPath);

  return success();
}

static void printImpl(llvm::raw_ostream &os, StringAttr name,
                      circt::igraph::InstancePath path, size_t bitPos,
                      StringRef comment) {
  std::string pathString;
  llvm::raw_string_ostream osPath(pathString);
  path.print(osPath);
  os << "DebugNode(" << pathString << "." << name.getValue() << "[" << bitPos
     << "])";
  if (!comment.empty())
    os << " // " << comment;
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
  SmallVector<Point> allStartPoints;
  SmallVector<igraph::InstanceGraphNode *> worklist;
  llvm::SetVector<Operation *> visited;
  worklist.push_back(topNode);
  while (!worklist.empty()) {
    auto *node = worklist.pop_back_val();
    assert(node && "node should not be null");
    auto op = node->getModule();
    if (!isa_and_nonnull<hw::HWModuleOp>(op) || !visited.insert(op))
      continue;

    auto paths = instancePathCache.getAbsolutePaths(
        cast<igraph::ModuleOpInterface>(*op), topNode);
    LLVM_DEBUG(llvm::dbgs() << "path: " << op.getModuleName() << " "
                            << paths.size() << "\n";);
    op->walk([&](Operation *op) {
      if (isa<seq::FirRegOp>(op)) {
        for (auto result : op->getResults()) {
          for (auto path : paths) {
            for (size_t i = 0, e = getBitWidth(result); i < e; ++i) {
              allStartPoints.push_back(Point(result, i, path));
            }
          }
        }
      }
    });

    for (auto *child : *node) {
      auto childOp = child->getInstance();
      if (!childOp || childOp->hasAttr("doNotPrint"))
        continue;

      worklist.push_back(child->getTarget());
    }
  }

  std::vector<Runner> runners;
  Context ctx(allStartPoints.size());
  for (size_t i = 0, e = top.getContext()->getNumThreads(); i < e; ++i)
    runners.emplace_back(&instanceGraph, &ctx);

  llvm::errs() << "Total start points: " << allStartPoints.size() << "\n";
  // Calculate bucket size with ceiling division to ensure all points are
  // distributed
  auto bucketSize =
      (allStartPoints.size() + runners.size() - 1) / runners.size();
  for (auto [i, start] : llvm::enumerate(allStartPoints)) {
    size_t group = std::min(i / bucketSize, runners.size() - 1);
    runners[group].startingPoints.push_back(start);
  }
  // for (auto [i, start] : llvm::enumerate(allStartPoints)) {
  //   // size_t group = std::min(i / bucketSize, runners.size() - 1);
  //   runners[i % runners.size()].startingPoints.push_back(start);
  // }

  auto result = mlir::failableParallelForEach(
      top.getContext(), runners, [&](Runner &runner) { return runner.run(); });
  // for (auto &runner : runners)
  //   runner.run();

  if (failed(result))
    return failure();

  SmallVector<std::pair<Point, Point>> results;
  // TODO: Sort in multithread and merge instead of sorting here.
  for (auto &runner : runners)
    results.append(runner.results.begin(), runner.results.end());

  llvm::sort(results, [](const std::pair<Point, Point> &a,
                         const std::pair<Point, Point> &b) {
    if (a.second.delay == b.second.delay) {
    }
    return a.second.delay > b.second.delay;
  });
  llvm::errs() << "Found " << results.size() << " paths\n";
  llvm::errs() << "Top 10 longest paths:\n";
  for (size_t i = 0; i < std::min<size_t>(10, results.size()); ++i) {
    llvm::errs() << i + 1 << ": delay= " << results[i].second.delay
                 << " fanout=";
    results[i].first.print(llvm::errs());
    llvm::errs() << " -> fanin=";
    results[i].second.print(llvm::errs());
    llvm::errs() << "\n";
    for (auto point : results[i].second.history) {
      llvm::errs() << "  " << point.delay << " ";
      printImpl(llvm::errs(), getNameImpl(point.value, point.bitPos),
                point.path, point.bitPos, point.comment);
      llvm::errs() << "\n";
    }
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
