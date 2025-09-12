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

#include "circt/Conversion/SynthToComb.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Dialect/Synth/Transforms/SynthesisPipeline.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
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
using namespace synth;

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
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));

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

enum Until { UntilCombLowering, UntilMapping, UntilEnd };

static auto runUntilValues = llvm::cl::values(
    clEnumValN(UntilCombLowering, "comb-lowering", "Lowering Comb to AIG/MIG"),
    clEnumValN(UntilMapping, "mapping", "Run technology/lut mapping"),
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

static cl::opt<std::string>
    outputLongestPath("output-longest-path",
                      cl::desc("Output file for longest path analysis "
                               "results. The analysis is only run "
                               "if file name is specified"),
                      cl::init(""), cl::cat(mainCategory));

static cl::opt<bool>
    outputLongestPathJSON("output-longest-path-json",
                          cl::desc("Output longest path analysis results in "
                                   "JSON format"),
                          cl::init(false), cl::cat(mainCategory));
static cl::opt<int>
    outputLongestPathTopKPercent("output-longest-path-top-k-percent",
                                 cl::desc("Output top K percent of longest "
                                          "paths in the analysis results"),
                                 cl::init(5), cl::cat(mainCategory));

static cl::opt<std::string> topName("top", cl::desc("Top module name"),
                                    cl::value_desc("name"), cl::init(""),
                                    cl::cat(mainCategory));

static cl::list<std::string> abcCommands("abc-commands",
                                         cl::desc("ABC passes to run"),
                                         cl::CommaSeparated,
                                         cl::cat(mainCategory));
static cl::opt<std::string> abcPath("abc-path", cl::desc("Path to ABC"),
                                    cl::value_desc("path"), cl::init("abc"),
                                    cl::cat(mainCategory));

static cl::opt<bool>
    ignoreAbcFailures("ignore-abc-failures",
                      cl::desc("Continue on ABC failure instead of aborting"),
                      cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> disableWordToBits("disable-word-to-bits",
                                       cl::desc("Disable LowerWordToBits pass"),
                                       cl::init(false), cl::cat(mainCategory));
static cl::opt<bool>
    disableDatapath("disable-datapath",
                    cl::desc("Disable datapath optimization passes"),
                    cl::init(false), cl::cat(mainCategory));
static cl::opt<bool>
    disableTimingAware("disable-timing-aware",
                       cl::desc("Disable datapath optimization passes"),
                       cl::init(false), cl::cat(mainCategory));

static cl::opt<int> maxCutSizePerRoot("max-cut-size-per-root",
                                      cl::desc("Maximum cut size per root"),
                                      cl::init(6), cl::cat(mainCategory));

static cl::opt<synth::OptimizationStrategy> synthesisStrategy(
    "synthesis-strategy", cl::desc("Synthesis strategy to use"),
    cl::values(clEnumValN(synth::OptimizationStrategyArea, "area",
                          "Optimize for area"),
               clEnumValN(synth::OptimizationStrategyTiming, "timing",
                          "Optimize for timing")),
    cl::init(synth::OptimizationStrategyTiming), cl::cat(mainCategory));

static cl::opt<int>
    lowerToKLUTs("lower-to-k-lut",
                 cl::desc("Lower to generic a truth table op with K inputs"),
                 cl::init(0), cl::cat(mainCategory));

static cl::opt<TargetIR>
    targetIR("target-ir", cl::desc("Target IR to lower to"),
             cl::values(clEnumValN(TargetIR::AIG, "aig", "AIG operation"),
                        clEnumValN(TargetIR::MIG, "mig", "MIG operation")),
             cl::init(TargetIR::AIG), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

static bool untilReached(Until until) {
  return until >= runUntilBefore || until > runUntilAfter;
}

static void
nestOrAddToHierarchicalRunner(OpPassManager &pm,
                              std::function<void(OpPassManager &pm)> pipeline,
                              const std::string &topName) {
  if (topName.empty()) {
    pipeline(pm.nest<hw::HWModuleOp>());
  } else {
    pm.addPass(circt::createHierarchicalRunner(topName, pipeline));
  }
}

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

// Add a default synthesis pipeline and analysis.
static void populateCIRCTSynthPipeline(PassManager &pm) {
  // ExtractTestCode is used to move verification code from design to
  // remove registers/logic used only for verification.
  pm.addPass(sv::createSVExtractTestCodePass(
      /*disableInstanceExtraction=*/false, /*disableRegisterExtraction=*/false,
      /*disableModuleInlining=*/false));
  auto pipeline = [](OpPassManager &pm) {
    circt::synth::CombLoweringPipelineOptions loweringOptions;
    loweringOptions.disableDatapath = disableDatapath;
    loweringOptions.timingAware = !disableTimingAware;
    loweringOptions.targetIR = targetIR;
    circt::synth::buildCombLoweringPipeline(pm, loweringOptions);
    if (untilReached(UntilCombLowering))
      return;

    circt::synth::SynthOptimizationPipelineOptions optimizationOptions;
    optimizationOptions.abcCommands = abcCommands;
    optimizationOptions.abcPath.setValue(abcPath);
    optimizationOptions.ignoreAbcFailures.setValue(ignoreAbcFailures);
    optimizationOptions.disableWordToBits.setValue(disableWordToBits);

    circt::synth::buildSynthOptimizationPipeline(pm, optimizationOptions);
    if (untilReached(UntilMapping))
      return;
    if (lowerToKLUTs) {
      circt::synth::GenericLutMapperOptions lutOptions;
      lutOptions.maxLutSize = lowerToKLUTs;
      lutOptions.maxCutsPerRoot = maxCutSizePerRoot;
      pm.addPass(circt::synth::createGenericLutMapper(lutOptions));
    }
  };

  nestOrAddToHierarchicalRunner(pm, pipeline, topName);

  if (!untilReached(UntilMapping)) {
    synth::TechMapperOptions options;
    options.maxCutsPerRoot = maxCutSizePerRoot;
    options.strategy = synthesisStrategy;
    pm.addPass(synth::createTechMapper(options));
  }

  // Run analysis if requested.
  if (!outputLongestPath.empty()) {
    circt::synth::PrintLongestPathAnalysisOptions options;
    options.outputFile = outputLongestPath;
    options.showTopKPercent = outputLongestPathTopKPercent;
    options.emitJSON = outputLongestPathJSON;
    pm.addPass(circt::synth::createPrintLongestPathAnalysis(options));
  }

  if (convertToComb)
    nestOrAddToHierarchicalRunner(
        pm,
        [&](OpPassManager &pm) {
          pm.addPass(circt::createConvertSynthToComb());
          if (lowerToKLUTs)
            pm.addPass(circt::comb::createLowerComb());
          pm.addPass(createCSEPass());
        },
        topName);
}

/// Check output stream before writing bytecode to it.
/// Warn and return true if output is known to be displayed.
static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    llvm::errs() << "WARNING: You're attempting to print out a bytecode file.\n"
                    "This is inadvisable as it may cause display problems. If\n"
                    "you REALLY want to taste MLIR bytecode first-hand, you\n"
                    "can force output with the `-f' option.\n\n";
    return true;
  }
  return false;
}

/// Print the operation to the specified stream, emitting bytecode when
/// requested and politely avoiding dumping to terminal unless forced.
static LogicalResult printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    return writeBytecodeToFile(op, os,
                               mlir::BytecodeWriterConfig(getCirctVersion()));
  op->print(os);
  return success();
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
  populateCIRCTSynthPipeline(pm);

  if (!topName.empty()) {
    // Set a top module name for the longest path analysis.
    module.get()->setAttr(
        circt::synth::LongestPathAnalysis::getTopModuleNameAttrName(),
        FlatSymbolRefAttr::get(&context, topName));
  }

  if (failed(pm.run(module.get())))
    return failure();

  auto timer = ts.nest("Print MLIR output");
  if (failed(printOp(module.get(), outputFile.value()->os())))
    return failure();
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
  circt::synth::registerSynthAnalysisPrerequisitePasses();

  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv, "Logic synthesis tool\n\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  registry.insert<comb::CombDialect, debug::DebugDialect, emit::EmitDialect,
                  hw::HWDialect, ltl::LTLDialect, om::OMDialect,
                  seq::SeqDialect, sim::SimDialect, synth::SynthDialect,
                  sv::SVDialect, verif::VerifDialect>();
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
