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
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

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
  void runOnOperation() override;
  using PrintLongestPathAnalysisBase::PrintLongestPathAnalysisBase;
  LogicalResult printAnalysisResult(const LongestPathAnalysis &analysis,
                                    igraph::InstancePathCache &pathCache,
                                    hw::HWModuleOp top, llvm::raw_ostream *os,
                                    llvm::json::OStream *jsonOS);

private:
  /// Print timing level statistics showing delay distribution
  void printTimingLevelStatistics(SmallVectorImpl<DataflowPath> &allTimingPaths,
                                  llvm::raw_ostream *os,
                                  llvm::json::OStream *jsonOS);

  /// Print detailed information for the top K critical paths
  void printTopKPathDetails(SmallVectorImpl<DataflowPath> &allTimingPaths,
                            hw::HWModuleOp top, llvm::raw_ostream *os,
                            llvm::json::OStream *jsonOS);

  /// Print detailed history of a timing path showing intermediate debug points
  void printPathHistory(const OpenPath &timingPath, llvm::raw_ostream &os);
};

} // namespace

/// Main method to print comprehensive longest path analysis results.
LogicalResult PrintLongestPathAnalysisPass::printAnalysisResult(
    const LongestPathAnalysis &analysis, igraph::InstancePathCache &pathCache,
    hw::HWModuleOp top, llvm::raw_ostream *os, llvm::json::OStream *jsonOS) {
  auto moduleName = top.getModuleNameAttr();

  LongestPathCollection collection(top.getContext());

  // Get all timing paths with full hierarchical elaboration
  if (failed(analysis.getAllPaths(moduleName, collection.paths, true)))
    return failure();

  // Emit diagnostics if testing is enabled (for regression testing)
  if (test) {
    for (auto &result : collection.paths) {
      auto fanOutLoc = result.getFanOutLoc();
      auto diag = mlir::emitRemark(fanOutLoc);
      SmallString<128> buf;
      llvm::raw_svector_ostream os(buf);
      result.print(os);
      diag << buf;
    }
  }

  size_t oldPathCount = collection.paths.size();
  collection.sortAndDropNonCriticalPathsPerFanOut();
  auto &longestPathForEachFanOut = collection.paths;

  // Print analysis header
  if (os) {
    *os << "# Longest Path Analysis result for " << top.getModuleNameAttr()
        << "\n"
        << "Found " << oldPathCount << " paths\n"
        << "Found " << longestPathForEachFanOut.size()
        << " unique fanout points\n"
        << "Maximum path delay: "
        << (longestPathForEachFanOut.empty()
                ? 0
                : longestPathForEachFanOut.front().getDelay())
        << "\n";

    *os << "## Showing Levels\n";
  }

  // Handle JSON output.
  if (jsonOS) {
    jsonOS->objectBegin();
    jsonOS->attribute("module_name", top.getModuleNameAttr().getValue());
  }

  auto deferClose = llvm::make_scope_exit([&]() {
    if (jsonOS)
      jsonOS->objectEnd();
  });

  // Print timing distribution statistics (histogram of delay levels)
  printTimingLevelStatistics(longestPathForEachFanOut, os, jsonOS);

  // Print detailed information for top K critical paths if requested
  if (showTopKPercent.getValue() > 0)
    printTopKPathDetails(longestPathForEachFanOut, top, os, jsonOS);

  return success();
}

/// Print timing level statistics showing delay distribution across all paths.
/// This provides a histogram-like view of timing levels in the design, showing
/// how many paths exist at each delay level and the cumulative percentage.
/// This is useful for understanding the overall timing characteristics of the
/// design.
void PrintLongestPathAnalysisPass::printTimingLevelStatistics(
    SmallVectorImpl<DataflowPath> &allTimingPaths, llvm::raw_ostream *os,
    llvm::json::OStream *jsonOS) {

  int64_t totalTimingPoints = allTimingPaths.size();
  int64_t cumulativeCount = 0;

  if (jsonOS) {
    jsonOS->attributeBegin("timing_levels");
    jsonOS->arrayBegin();
  }
  auto closeJson = llvm::make_scope_exit([&]() {
    if (jsonOS) {
      jsonOS->arrayEnd();
      jsonOS->attributeEnd();
    }
  });

  // Process paths grouped by delay level (paths are already sorted by delay)
  if (allTimingPaths.empty())
    return;
  for (int64_t index = allTimingPaths.size() - 1; index >= 0;) {
    int64_t oldIndex = index;
    auto currentDelay = allTimingPaths[index--].getDelay();

    // Count all paths with the same delay value to create histogram bins
    while (index >= 0 && allTimingPaths[index].getDelay() == currentDelay)
      --index;

    int64_t pathsWithSameDelay = oldIndex - index;

    cumulativeCount += pathsWithSameDelay;

    // Calculate cumulative percentage to show timing distribution
    double cumulativePercentage =
        (double)cumulativeCount / totalTimingPoints * 100.0;

    // Print formatted timing level statistics in tabular format
    // Format: Level = delay_value . Count = path_count . percentage%
    if (os)
      *os << llvm::format("Level = %-10d. Count = %-10d. %-10.2f%%\n",
                          currentDelay, pathsWithSameDelay,
                          cumulativePercentage);
    if (jsonOS) {
      jsonOS->objectBegin();
      jsonOS->attribute("level", currentDelay);
      jsonOS->attribute("count", pathsWithSameDelay);
      jsonOS->attribute("percentage", cumulativePercentage);
      jsonOS->objectEnd();
    }
  }
}

/// Print detailed information for the top K critical paths.
/// This shows the most critical timing paths in the design, providing detailed
/// information about each path including fanout/fanin points and path history.
void PrintLongestPathAnalysisPass::printTopKPathDetails(
    SmallVectorImpl<DataflowPath> &allTimingPaths, hw::HWModuleOp top,
    llvm::raw_ostream *os, llvm::json::OStream *jsonOS) {

  auto topKCount = static_cast<uint64_t>(allTimingPaths.size()) *
                   std::clamp(showTopKPercent.getValue(), 0, 100) / 100;

  if (os)
    *os << "## Top " << topKCount << " (out of " << allTimingPaths.size()
        << ") fan-out points\n\n";

  if (jsonOS) {
    jsonOS->attributeBegin("top_paths");
    jsonOS->arrayBegin();
  }
  auto closeJson = llvm::make_scope_exit([&]() {
    if (jsonOS) {
      jsonOS->arrayEnd();
      jsonOS->attributeEnd();
    }
  });

  // Process paths from highest delay to lowest (reverse order since paths are
  // sorted ascending)
  for (size_t i = 0; i < std::min<size_t>(topKCount, allTimingPaths.size());
       ++i) {
    auto &path = allTimingPaths[i];
    if (jsonOS)
      jsonOS->value(toJSON(path));

    if (!os)
      continue;
    // Print path header with ranking and delay information
    *os << "==============================================\n"
        << "#" << i + 1 << ": Distance=" << path.getDelay() << "\n";

    // Print fanout point (where the critical path starts)
    *os << "FanOut=";
    path.printFanOut(*os);

    // Print fanin point (where the critical path ends)
    *os << "\n"
        << "FanIn=";
    path.getFanIn().print(*os);
    *os << "\n";

    // Print detailed path history showing intermediate logic stages
    printPathHistory(path.getPath(), *os);
  }
}

/// Print detailed history of a timing path showing intermediate debug points.
/// This traces the path from fanout to fanin, showing each logic stage and
/// the delay contribution of each stage. This is crucial for understanding
/// where delay is being accumulated along the critical path and identifying
/// optimization opportunities.
void PrintLongestPathAnalysisPass::printPathHistory(const OpenPath &timingPath,
                                                    llvm::raw_ostream &os) {
  int64_t remainingDelay = timingPath.getDelay();

  if (!timingPath.getHistory().isEmpty()) {
    os << "== History Start (closer to fanout) ==\n";

    // Walk through debug points in order from fanout to fanin
    for (auto &debugPoint : timingPath.getHistory()) {
      // Calculate delay contribution of this logic stage
      int64_t stepDelay = remainingDelay - debugPoint.delay;
      remainingDelay = debugPoint.delay;

      // Show the delay contribution and the debug point information
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

  auto &os = file->os();
  std::unique_ptr<llvm::json::OStream> jsonOS;
  if (emitJSON.getValue()) {
    jsonOS = std::make_unique<llvm::json::OStream>(os);
    jsonOS->arrayBegin();
  }

  auto closeJson = llvm::make_scope_exit([&]() {
    if (jsonOS)
      jsonOS->arrayEnd();
  });

  for (auto top : analysis.getTopModules())
    if (failed(printAnalysisResult(analysis, pathCache, top,
                                   jsonOS ? nullptr : &os, jsonOS.get())))
      return signalPassFailure();
  file->keep();
  return markAllAnalysesPreserved();
}
