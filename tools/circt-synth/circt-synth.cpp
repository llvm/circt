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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
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

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialect",
                              cl::desc("Allow unknown dialects in the input"),
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

static cl::opt<bool>
    enableWordToBits("enable-word-to-bits",
                     cl::desc("Perform bit-blasting in the pipeline"),
                     cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    printResourceUsage("print-resource-usage",
                       cl::desc("Print resource usage of AIG operations"),
                       cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string>
    resourceUsageOutputFile("resource-usage-output",
                            cl::desc("Output file for resource usage"),
                            cl::init("-"), cl::cat(mainCategory));

static cl::opt<bool>
    printLongestPath("print-longest-path",
                     cl::desc("Print longest path of AIG operations"),
                     cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string>
    longestPathOutputFile("longest-path-output",
                          cl::desc("Output file for longest path"),
                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> topName("top", cl::desc("Top module name"),
                                    cl::value_desc("name"), cl::init(""),
                                    cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

static bool untilReached(Until until) {
  return until >= runUntilBefore || until > runUntilAfter;
}

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

template <typename... AllowedOpTy>
static void partiallyLegalizeCombToAIG(SmallVectorImpl<std::string> &ops) {
  (ops.push_back(AllowedOpTy::getOperationName().str()), ...);
}

static void populateSynthesisPipeline(PassManager &pm) {
  auto pipeline = [](OpPassManager &mpm) {
    // Add the AIG to Comb at the scope exit if requested.
    auto addAIGToComb = llvm::make_scope_exit([&]() {
      if (convertToComb) {
        mpm.addPass(circt::createConvertAIGToComb());
        mpm.addPass(createCSEPass());
      }
    });
    mpm.addPass(circt::comb::createCombIntRangeNarrowing());

    {
      // Partially legalize Comb to AIG, run CSE and canonicalization.
      circt::ConvertCombToAIGOptions options;
      partiallyLegalizeCombToAIG<comb::AndOp, comb::OrOp, comb::XorOp,
                                 comb::MuxOp, comb::ICmpOp, hw::ArrayGetOp,
                                 hw::ArraySliceOp, hw::ArrayCreateOp,
                                 hw::ArrayConcatOp, hw::AggregateConstantOp>(
          options.additionalLegalOps);
      mpm.addPass(circt::createConvertCombToAIG(options));
    }
    mpm.addPass(comb::createCombIntRangeNarrowing());
    mpm.addPass(createCSEPass());
    mpm.addPass(createSimpleCanonicalizerPass());

    // Fully legalize AIG to Comb.
    mpm.addPass(circt::hw::createHWAggregateToCombPass());
    // Partially legalize Comb to AIG, run CSE and canonicalization.
    circt::ConvertCombToAIGOptions options;
    // partiallyLegalizeCombToAIG<comb::AndOp, comb::OrOp, comb::XorOp,
    //                            comb::MuxOp>(options.additionalLegalOps);
    mpm.addPass(circt::createConvertCombToAIG(options));
    mpm.addPass(createCSEPass());
    if (untilReached(UntilAIGLowering))
      return;
    mpm.addPass(createSimpleCanonicalizerPass());
    mpm.addPass(createCSEPass());
    // mpm.addPass(aig::createLowerVariadic());
    // TODO: LowerWordToBits is not scalable for large designs. Change to
    // conditionally enable the pass once the rest of the pipeline was able to
    // handle multibit operands properly.
    if (enableWordToBits)
      mpm.addPass(aig::createLowerWordToBits());
    mpm.addPass(createCSEPass());
    mpm.addPass(createSimpleCanonicalizerPass());
  };

  if (topName.empty()) {
    pipeline(pm.nest<hw::HWModuleOp>());
  } else {
    pm.addPass(circt::createHierarchicalRunner(topName, pipeline));
  }

  if (printResourceUsage) {
    circt::aig::PrintResourceUsageAnalysisOptions options;
    options.topModuleName = topName;
    options.outputJSONFile = resourceUsageOutputFile;
    options.printSummary = true;
    pm.addPass(circt::aig::createPrintResourceUsageAnalysis(options));
  }

  if (printLongestPath) {
    circt::aig::PrintLongestPathAnalysisOptions options;
    options.topModuleName = topName;
    options.outputJSONFile = longestPathOutputFile;
    options.showTopKPercent = 5;
    pm.addPass(circt::aig::createPrintLongestPathAnalysis(options));
  }
  // TODO: Add balancing, rewriting, FRAIG conversion, etc.
  if (untilReached(UntilEnd))
    return;
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
                  circt::hw::HWDialect, circt::verif::VerifDialect,
                  circt::sv::SVDialect, emit::EmitDialect, seq::SeqDialect,
                  om::OMDialect, sv::SVDialect, verif::VerifDialect,
                  ltl::LTLDialect, debug::DebugDialect, sim::SimDialect>();
  MLIRContext context(registry);
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
