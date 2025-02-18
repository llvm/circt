//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HierarchicalRunner pass which runs a pass pipeline
// on specific hierarchichy.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/InstanceGraph.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"

using namespace circt;
using namespace mlir;

namespace circt {
#define GEN_PASS_DEF_HIERARCHICALRUNNER
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

namespace {
struct HierarchicalRunnerPass
    : public circt::impl::HierarchicalRunnerBase<HierarchicalRunnerPass> {
  using circt::impl::HierarchicalRunnerBase<
      HierarchicalRunnerPass>::HierarchicalRunnerBase;
  void runOnOperation() override;
  HierarchicalRunnerPass(const std::string &topName,
                         llvm::function_ref<void(OpPassManager &)> populateFunc,
                         bool includeBoundInstances) {
    this->topName = topName;
    this->includeBoundInstances = includeBoundInstances;
    populateFunc(dynamicPM);
    llvm::raw_string_ostream os(pipelineStr);
    dynamicPM.printAsTextualPipeline(os);
  }

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(
            HierarchicalRunnerBase::initializeOptions(options, errorHandler)))
      return failure();

    if (failed(parsePassPipeline(pipelineStr, dynamicPM)))
      return errorHandler("Failed to parse composite pass pipeline");

    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    dynamicPM.getDependentDialects(registry);
  }

private:
  OpPassManager dynamicPM;
};
} // namespace

void HierarchicalRunnerPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<circt::igraph::InstanceGraph>();

  llvm::SetVector<Operation *> visited;
  auto name = mlir::StringAttr::get(getOperation()->getContext(), topName);
  auto *top = instanceGraph.lookupOrNull(name);
  if (!top) {
    mlir::emitError(mlir::UnknownLoc::get(&getContext()))
        << "top module not found in instance graph " << topName;
    return signalPassFailure();
  }

  SmallVector<igraph::InstanceGraphNode *> worklist;

  auto am = getAnalysisManager();
  worklist.push_back(top);
  while (!worklist.empty()) {
    auto *node = worklist.pop_back_val();
    if (!node)
      continue;
    auto op = node->getModule();
    if (!op || !visited.insert(op))
      continue;

    // Ensure an analysis manager has been constructed for each of the nodes.
    // This prevents thread races when running the nested pipelines.
    am.nest(op);

    for (auto *child : *node) {
      auto childOp = child->getInstance();
      if (!childOp ||
          (!includeBoundInstances && childOp->hasAttr("doNotPrint")))
        continue;

      worklist.push_back(child->getTarget());
    }
  }

  // We must maintain a fixed pool of pass managers which is at least as large
  // as the maximum parallelism of the failableParallelForEach below.
  // Note: The number of pass managers here needs to remain constant
  // to prevent issues with pass instrumentations that rely on having the same
  // pass manager for the main thread.
  size_t numThreads = getContext().getNumThreads();

  llvm::SmallVector<OpPassManager> pipelines(numThreads, dynamicPM);

  // An atomic failure variable for the async executors.
  std::vector<std::atomic<bool>> activePMs(pipelines.size());
  std::fill(activePMs.begin(), activePMs.end(), false);
  auto result = mlir::failableParallelForEach(
      &getContext(), visited, [&](Operation *node) -> LogicalResult {
        // Find a pass manager for this operation.
        auto it = llvm::find_if(activePMs, [](std::atomic<bool> &isActive) {
          bool expectedInactive = false;
          return isActive.compare_exchange_strong(expectedInactive, true);
        });
        assert(it != activePMs.end() &&
               "could not find inactive pass manager for thread");
        unsigned pmIndex = it - activePMs.begin();
        auto result = runPipeline(pipelines[pmIndex], node);
        // Reset the active bit for this pass manager.
        activePMs[pmIndex].store(false);
        return result;
      });
  if (failed(result))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createHierarchicalRunner(
    const std::string &topName,
    llvm::function_ref<void(mlir::OpPassManager &)> pipeline,
    bool includeBoundInstances) {
  return std::make_unique<HierarchicalRunnerPass>(topName, pipeline,
                                                  includeBoundInstances);
}
