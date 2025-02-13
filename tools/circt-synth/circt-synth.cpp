//===- circt-synth.cpp - The circt-synth driver -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initializes the 'circt-synth' tool, which performs logic
/// synthesis.
///
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AIGToComb.h"
#include "circt/Conversion/CombToAIG.h"
#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Threading.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-synth Options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::init("-"),
                                          cl::desc("Specify an input file"),
                                          cl::value_desc("filename"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

// Options to control early-out from pipeline.
enum Until { UntilAIGLowering, UntilEnd };

static auto runUntilValues = llvm::cl::values(
    clEnumValN(UntilAIGLowering, "aig-lowering", "Lowering of AIG"),
    clEnumValN(UntilEnd, "all", "Run entire pipeline (default)"));

static llvm::cl::opt<Until> runUntilBefore(
    "until-before", llvm::cl::desc("Stop pipeline before a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));
static llvm::cl::opt<Until> runUntilAfter(
    "until-after", llvm::cl::desc("Stop pipeline after a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));

static cl::opt<bool>
    convertToComb("convert-to-comb",
                  cl::desc("Convert AIG to Comb at the end of the pipeline"),
                  cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string> topName("top", cl::desc("Top module name"),
                                    cl::value_desc("name"), cl::init(""),
                                    cl::cat(mainCategory));

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialects",
                              cl::desc("Allow unregistered dialects"),
                              cl::init(false), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

static bool untilReached(Until until) {
  return until >= runUntilBefore || until > runUntilAfter;
}

//===----------------------------------------------------------------------===//
// Instance Graph Pass Manager
//===----------------------------------------------------------------------===//

struct ConditionallyRunPass
    : public PassWrapper<ConditionallyRunPass, OperationPass<void>> {
  ConditionallyRunPass(llvm::function_ref<void(OpPassManager &)> pipeline,
                       llvm::function_ref<bool(Operation *)> shouldRunPass)
      : PassWrapper<ConditionallyRunPass, OperationPass<void>>(),
        pipeline(pipeline), shouldRunPass(shouldRunPass) {}

  LogicalResult initialize(MLIRContext *context) override { return success(); }

  void runOnOperation() override {
    if (!shouldRunPass(getOperation()))
      return markAllAnalysesPreserved();

    auto pm = OpPassManager(getOperation()->getName());
    pipeline(pm);
    if (failed(runPipeline(pm, getOperation())))
      return signalPassFailure();
  }

  llvm::function_ref<void(OpPassManager &)> pipeline;
  llvm::function_ref<bool(Operation *)> shouldRunPass;
};

struct NestPassesUnderHierarchyPass
    : public PassWrapper<NestPassesUnderHierarchyPass,
                         OperationPass<mlir::ModuleOp>> {
  NestPassesUnderHierarchyPass(
      llvm::function_ref<void(OpPassManager &)> pipeline)
      : PassWrapper<NestPassesUnderHierarchyPass,
                    OperationPass<mlir::ModuleOp>>(),
        pipeline(pipeline) {}

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<circt::igraph::InstanceGraph>();
    std::function<bool(Operation *)> runPass;
    llvm::SetVector<Operation *> visited;
    if (topName.empty()) {
      runPass = [&](Operation *op) { return true; };
    } else {
      auto name = mlir::StringAttr::get(getOperation()->getContext(), topName);
      auto *top = instanceGraph.lookup(name);
      if (!top)
        return signalPassFailure();
      SmallVector<igraph::InstanceGraphNode *> worklist;
      worklist.push_back(top);
      while (!worklist.empty()) {
        auto *node = worklist.pop_back_val();
        auto op = node->getModule();
        if (!op || !visited.insert(op).second)
          continue;
        for (auto *child : *node) {
          auto childOp = child->getInstance();
          if (!childOp || childOp->hasAttr("doNotPrint"))
            continue;
          worklist.push_back(child->getTarget());
        }
      }

      runPass = [&](Operation *op) -> bool { return visited.count(op); };
    }

    auto dynamicPM = OpPassManager(getOperation()->getName());
    auto &pm = dynamicPM.nest<hw::HWModuleOp>();
    pm.addPass(std::make_unique<ConditionallyRunPass>(pipeline, runPass));
    if (failed(runPipeline(dynamicPM, getOperation())))
      return signalPassFailure();

    // We must maintain a fixed pool of pass managers which is at least as large
    // as the maximum parallelism of the failableParallelForEach below.
    // Note: The number of pass managers here needs to remain constant
    // to prevent issues with pass instrumentations that rely on having the same
    // pass manager for the main thread.
    size_t numThreads = getContext().getNumThreads();
    OpPassManager nestedPipelines;

    llvm::SmallVector<OpPassManager> pipelines;
    pipeline(nestedPipelines);
    const auto &opPipelines = nestedPipelines;
    if (pipelines.size() < numThreads) {
      pipelines.reserve(numThreads);
      pipelines.resize(numThreads, opPipelines);
    }

    // An atomic failure variable for the async executors.
    std::vector<std::atomic<bool>> activePMs(pipelines.size());
    std::fill(activePMs.begin(), activePMs.end(), false);
    auto result = mlir::failableParallelForEach(
        getContext(), visited.getArrayRef(), [&](Operation *node) -> LogicalResult {
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
  }

  llvm::function_ref<void(OpPassManager &)> pipeline;
};

static std::unique_ptr<Pass> createNestPassesUnderHierarchyPass(
    llvm::function_ref<void(OpPassManager &)> pipeline) {
  return std::make_unique<NestPassesUnderHierarchyPass>(pipeline);
}

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

static void populateSynthesisPipeline(PassManager &pm) {
  pm.addPass(createNestPassesUnderHierarchyPass([](OpPassManager &mpm) {
    // Add the AIG to Comb at the scope exit if requested.
    auto addAIGToComb = llvm::make_scope_exit([&]() {
      if (convertToComb) {
        mpm.addPass(circt::createConvertAIGToComb());
        mpm.addPass(createCSEPass());
      }
    });
    mpm.addPass(circt::hw::createHWAggregateToCombPass());
    mpm.addPass(circt::createConvertCombToAIG());
    mpm.addPass(createCSEPass());
    if (untilReached(UntilAIGLowering))
      return;
    mpm.addPass(createSimpleCanonicalizerPass());
    mpm.addPass(createCSEPass());
    mpm.addPass(aig::createLowerVariadic());
    // TODO: LowerWordToBits is not scalable for large designs. Change to
    // conditionally enable the pass once the rest of the pipeline was able to
    // handle multibit operands properly.
    mpm.addPass(aig::createLowerWordToBits());
    mpm.addPass(createCSEPass());
    mpm.addPass(createSimpleCanonicalizerPass());
    // TODO: Add balancing, rewriting, FRAIG conversion, etc.
    if (untilReached(UntilEnd))
      return;
  }));

  // TODO: Add LUT mapping, etc.
}

/// This function initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeSynthesis(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  OwningOpRef<ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    // Parse the provided input files.
    module = parseSourceFile<ModuleOp>(inputFilename, &context);
  }
  if (!module)
    return failure();
  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "circt-synth"));
  populateSynthesisPipeline(pm);
  if (failed(pm.run(module.get())))
    return failure();

  auto timer = ts.nest("Print MLIR output");
  OpPrintingFlags printingFlags;
  module->print(outputFile.value()->os(), printingFlags);
  outputFile.value()->keep();
  return success();
}

/// The entry point for the `circt-synth` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeSynthesis` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv, "Logic synthesis tool\n\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  registry.insert<circt::aig::AIGDialect, circt::comb::CombDialect,
                  circt::hw::HWDialect, circt::seq::SeqDialect,
                  circt::om::OMDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the synthesis; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeSynthesis(context)));
}
