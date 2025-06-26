//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass prints the longest path analysis results to a file.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/AIG/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <variant>

#define DEBUG_TYPE "aig-longest-path-analysis"
using namespace circt;
using namespace aig;

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTLONGESTPATHANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

//===----------------------------------------------------------------------===//
// PrintLongestPathAnalysisPass
//===----------------------------------------------------------------------===//

namespace {
struct PrintLongestPathAnalysisPass
    : public impl::PrintLongestPathAnalysisBase<PrintLongestPathAnalysisPass> {
  using PrintLongestPathAnalysisBase::outputFile;
  using PrintLongestPathAnalysisBase::PrintLongestPathAnalysisBase;
  using PrintLongestPathAnalysisBase::showTopKPercent;

  // Type alias for timing path variant
  using TimingPathVariant =
      std::variant<DataflowPath, std::tuple<size_t, size_t, OpenPath>>;

  void runOnOperation() override;
  LogicalResult printAnalysisResult(const LongestPathAnalysis &analysis,
                                    igraph::InstancePathCache &pathCache,
                                    hw::HWModuleOp top, llvm::raw_ostream &os);

private:
  /// Print timing level statistics showing delay distribution
  void printTimingLevelStatistics(
      SmallVectorImpl<TimingPathVariant> &allTimingPaths,
      const std::function<int64_t(const TimingPathVariant &)> &extractDelay,
      llvm::raw_ostream &os);

  /// Print detailed information for the top K critical paths
  void printTopKPathDetails(
      SmallVectorImpl<TimingPathVariant> &allTimingPaths,
      const std::function<int64_t(const TimingPathVariant &)> &extractDelay,
      hw::HWModuleOp top, llvm::raw_ostream &os);

  /// Print detailed history of a timing path showing intermediate debug points
  void printPathHistory(const OpenPath &timingPath, llvm::raw_ostream &os);
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

  os << "# Longest Path Analysis result for " << top.getModuleNameAttr() << "\n"
     << "Found " << closedPaths.size() << " closed paths\n";

  os << "## Showing Levels\n";

  // Data structures for collecting timing statistics and path information
  // Using variant to handle both internal timing paths and output port paths
  using TimingPathVariant =
      std::variant<DataflowPath, std::tuple<size_t, size_t, OpenPath>>;
  SmallVector<TimingPathVariant> allTimingPaths;
  auto *topLevelNode = pathCache.instanceGraph.lookup(top);

  // Map from (value, bit position) to instance paths and their timing
  // information This tracks the timing level for each signal bit across
  // different instances
  DenseMap<std::pair<Value, size_t>,
           DenseMap<circt::igraph::InstancePath, OpenPath>>
      signalTimingLevels;

  llvm::ImmutableListFactory<DebugPoint> debugPointFactory;

  // Process closed paths and build timing level map
  for (auto &closedPath : closedPaths) {
    auto fanOutSignal = closedPath.getFanOut();
    auto signalKey = std::make_pair(fanOutSignal.value, fanOutSignal.bitPos);
    auto &timingLevelMap = signalTimingLevels[signalKey];

    // Find the module containing the fan-out signal
    auto *fanOutOperation = fanOutSignal.value.getDefiningOp();
    assert(fanOutOperation && "Fan-out value must have a defining operation");
    auto fanOutModule =
        fanOutOperation->template getParentOfType<hw::HWModuleOp>();
    assert(fanOutModule && "Fan-out operation must be within a HWModuleOp");

    auto topToFanOutModulePaths =
        pathCache.getRelativePaths(fanOutModule, topLevelNode);

    // Skip if we already have all possible paths for this signal
    // (optimization to avoid redundant processing)
    if (timingLevelMap.size() == topToFanOutModulePaths.size())
      continue;

    // Get paths from root to fan-out module and from top to root
    auto *rootNode = pathCache.instanceGraph.lookup(closedPath.getRoot());
    auto rootToFanOutModulePaths =
        pathCache.getRelativePaths(fanOutModule, rootNode);
    auto topToRootPaths =
        pathCache.getRelativePaths(closedPath.getRoot(), topLevelNode);

    // Combine all possible path combinations to build complete timing picture
    for (const auto &topToRootPath : topToRootPaths) {
      for (const auto &rootToFanOutPath : rootToFanOutModulePaths) {
        auto concatenatedPath =
            pathCache.concatPath(topToRootPath, rootToFanOutPath);

        // Update timing information if this path has better (higher) delay
        auto pathIterator = timingLevelMap.find(concatenatedPath);
        if (pathIterator == timingLevelMap.end() ||
            pathIterator->second.getDelay() < closedPath.getDelay()) {
          auto newTimingPath = closedPath.getPath();
          newTimingPath.prependPaths(pathCache, &debugPointFactory,
                                     topToRootPath);
          timingLevelMap.insert_or_assign(concatenatedPath, newTimingPath);
        }
      }
    }
  }

  // Process open paths ending at flip-flops
  for (const auto &[signalObject, openPath] : openPathsToFF) {
    auto signalKey = std::make_pair(signalObject.value, signalObject.bitPos);
    auto &timingLevelMap = signalTimingLevels[signalKey];

    // Update the maximum delay for the path to this flip-flop
    auto instancePath = openPath.getFanIn().instancePath;
    auto insertResult = timingLevelMap.try_emplace(instancePath, openPath);
    if (!insertResult.second &&
        insertResult.first->second.getDelay() < openPath.getDelay())
      insertResult.first->second = openPath;
  }

  // Process open paths from output ports and collect delay statistics
  DenseMap<std::pair<size_t, size_t>, OpenPath> outputPortTimingInfo;
  for (const auto &[resultNumber, bitPosition, openPath] :
       openPathsFromOutputPorts) {
    auto outputKey = std::make_pair(resultNumber, bitPosition);
    auto insertResult = outputPortTimingInfo.try_emplace(outputKey, openPath);
    if (!insertResult.second &&
        insertResult.first->second.getDelay() < openPath.getDelay())
      insertResult.first->second = openPath;
  }

  // Convert output port timing data to the unified format
  for (const auto &[outputKey, openPath] : outputPortTimingInfo) {
    allTimingPaths.push_back(
        std::make_tuple(outputKey.first, outputKey.second, openPath));
  }

  // Collect all internal timing level data into the unified format
  for (const auto &[signalKey, timingLevelMap] : signalTimingLevels) {
    for (const auto &[instancePath, timingPath] : timingLevelMap) {
      auto fanOutObject =
          Object(instancePath, signalKey.first, signalKey.second);
      assert(fanOutObject.value && "Fan-out object must have a valid value");
      assert(timingPath.getFanIn().value &&
             "Timing path must have a valid fan-in");
      allTimingPaths.push_back(DataflowPath(fanOutObject, timingPath, top));
    }
  }

  // Helper lambda to extract delay from variant timing path data
  auto extractDelay = [](const TimingPathVariant &pathVariant) -> int64_t {
    if (auto *dataflowPath = std::get_if<DataflowPath>(&pathVariant))
      return dataflowPath->getDelay();
    auto &[resultNumber, bitPosition, openPath] =
        std::get<std::tuple<size_t, size_t, OpenPath>>(pathVariant);
    return openPath.getDelay();
  };

  // Sort all timing paths by delay value (ascending order)
  llvm::sort(allTimingPaths,
             [&extractDelay](const auto &first, const auto &second) {
               return extractDelay(first) < extractDelay(second);
             });

  // Print timing distribution statistics
  printTimingLevelStatistics(allTimingPaths, extractDelay, os);

  // Print detailed information for top K paths if requested
  if (showTopKPercent)
    printTopKPathDetails(allTimingPaths, extractDelay, top, os);

  return success();
}

/// Print timing level statistics showing delay distribution
void PrintLongestPathAnalysisPass::printTimingLevelStatistics(
    SmallVectorImpl<TimingPathVariant> &allTimingPaths,
    const std::function<int64_t(const TimingPathVariant &)> &extractDelay,
    llvm::raw_ostream &os) {

  int64_t totalTimingPoints = allTimingPaths.size();
  int64_t cumulativeCount = 0;

  for (size_t index = 0; index < allTimingPaths.size();) {
    auto currentDelay = extractDelay(allTimingPaths[index++]);
    int64_t pathsWithSameDelay = 1;

    // Count all paths with the same delay value
    while (index < allTimingPaths.size() &&
           extractDelay(allTimingPaths[index]) == currentDelay) {
      pathsWithSameDelay++;
      index++;
    }

    cumulativeCount += pathsWithSameDelay;

    // Calculate cumulative percentage
    double cumulativePercentage =
        (double)cumulativeCount / totalTimingPoints * 100.0;

    // Print formatted timing level statistics
    os << llvm::format("Level = %-10d. Count = %-10d. %-10.2f%%\n",
                       currentDelay, pathsWithSameDelay, cumulativePercentage);
  }
}

/// Print detailed information for the top K critical paths
void PrintLongestPathAnalysisPass::printTopKPathDetails(
    SmallVectorImpl<TimingPathVariant> &allTimingPaths,
    const std::function<int64_t(const TimingPathVariant &)> &extractDelay,
    hw::HWModuleOp top, llvm::raw_ostream &os) {

  auto topKPercent = showTopKPercent.getValue();
  auto topKCount = allTimingPaths.size() * topKPercent / 100;

  os << "## Top " << topKCount << " (" << topKPercent << "% of "
     << allTimingPaths.size() << ") fan-out points\n\n";

  // Process paths from highest delay to lowest (reverse order)
  for (size_t i = 0; i < std::min<size_t>(topKCount, allTimingPaths.size());
       ++i) {
    auto &currentPathVariant = allTimingPaths[allTimingPaths.size() - i - 1];

    // Extract fan-out information and timing path
    SmallString<128> fanOutDescription;
    llvm::raw_svector_ostream fanOutStream(fanOutDescription);
    OpenPath currentTimingPath;

    if (auto *dataflowPath = std::get_if<DataflowPath>(&currentPathVariant)) {
      // Internal dataflow path
      dataflowPath->getFanOut().print(fanOutStream);
      currentTimingPath = dataflowPath->getPath();
    } else {
      // Output port path
      auto &[resultNumber, bitPosition, openPath] =
          std::get<std::tuple<size_t, size_t, OpenPath>>(currentPathVariant);

      bool needsBitIndex =
          hw::getBitWidth(top.getOutputTypes()[resultNumber]) > 1;
      fanOutStream << "Object($root." << top.getOutputName(resultNumber);
      if (needsBitIndex)
        fanOutStream << "[" << bitPosition << "]";
      fanOutStream << ")";
      currentTimingPath = openPath;
    }

    // Print path header information
    os << "==============================================\n";
    os << "#" << i + 1 << ": Distance=" << extractDelay(currentPathVariant)
       << "\n"
       << "FanOut=" << fanOutDescription << "\n"
       << "FanIn=";
    currentTimingPath.getFanIn().print(os);
    os << "\n";

    // Print detailed path history if available
    printPathHistory(currentTimingPath, os);
  }
}

/// Print detailed history of a timing path showing intermediate debug points
void PrintLongestPathAnalysisPass::printPathHistory(const OpenPath &timingPath,
                                                    llvm::raw_ostream &os) {
  int64_t remainingDelay = timingPath.getDelay();

  if (!timingPath.getHistory().isEmpty()) {
    os << "== History Start (closer to fanout) ==\n";

    for (auto &debugPoint : timingPath.getHistory()) {
      int64_t stepDelay = remainingDelay - debugPoint.delay;
      remainingDelay = debugPoint.delay;

      os << "<--- (logic delay " << stepDelay << ") ---\n";
      debugPoint.print(os);
      os << "\n";
    }

    os << "== History End (closer to fanin) ==\n";
  }
}

void PrintLongestPathAnalysisPass::runOnOperation() {
  auto &analysis = getAnalysis<aig::LongestPathAnalysisWithTrace>();
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
