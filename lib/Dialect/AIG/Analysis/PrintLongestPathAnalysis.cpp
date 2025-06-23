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
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

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
