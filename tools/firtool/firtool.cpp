//===- firtool.cpp - The firtool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'firtool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Translation/ExportVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

/// Allow the user to specify the input file format.  This can be used to
/// override the input, and can be used to specify ambiguous cases like standard
/// input.
enum InputFormatKind { InputUnspecified, InputFIRFile, InputMLIRFile };

static cl::opt<InputFormatKind> inputFormat(
    "format", cl::desc("Specify input file format:"),
    cl::values(clEnumValN(InputUnspecified, "autodetect",
                          "Autodetect input format"),
               clEnumValN(InputFIRFile, "fir", "Parse as .fir file"),
               clEnumValN(InputMLIRFile, "mlir", "Parse as .mlir file")),
    cl::init(InputUnspecified));

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
    outputFilename("o",
                   cl::desc("Output filename, or directory for split output"),
                   cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden);

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden);

static cl::opt<bool> disableOptimization("disable-opt",
                                         cl::desc("disable optimizations"));

static cl::opt<bool> inliner("inline",
                             cl::desc("Run the FIRRTL module inliner"),
                             cl::init(true));

static cl::opt<bool> lowerToHW("lower-to-hw",
                               cl::desc("run the lower-to-hw pass"));

static cl::opt<bool> enableAnnotationWarning(
    "warn-on-unprocessed-annotations",
    cl::desc("Warn about annotations that were not removed by lower-to-hw"),
    cl::init(false));

static cl::opt<bool> imconstprop(
    "imconstprop",
    cl::desc(
        "Enable intermodule constant propagation and dead code elimination"),
    cl::init(true));

static cl::opt<bool>
    lowerTypes("lower-types",
               cl::desc("run the lower-types pass within lower-to-hw"),
               cl::init(true));

static cl::opt<bool> expandWhens("expand-whens",
                                 cl::desc("disable the expand-whens pass"),
                                 cl::init(true));

static cl::opt<bool>
    blackBoxMemory("blackbox-memory",
                   cl::desc("Create a black box for all memory operations"),
                   cl::init(false));

static cl::opt<bool>
    ignoreFIRLocations("ignore-fir-locators",
                       cl::desc("ignore the @info locations in the .fir file"),
                       cl::init(false));

static cl::opt<bool>
    inferWidths("infer-widths",
                cl::desc("run the width inference pass on firrtl"),
                cl::init(true));

static cl::opt<bool> extractTestCode("extract-test-code",
                                     cl::desc("run the extract test code pass"),
                                     cl::init(false));
static cl::opt<bool>
    grandCentral("firrtl-grand-central",
                 cl::desc("create interfaces and data/memory taps from SiFive "
                          "Grand Central annotations"),
                 cl::init(false));

static cl::opt<bool>
    checkCombCycles("firrtl-check-comb-cycles",
                    cl::desc("check combinational cycles on firrtl"),
                    cl::init(false));

enum OutputFormatKind {
  OutputMLIR,
  OutputVerilog,
  OutputSplitVerilog,
  OutputDisabled
};

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(clEnumValN(OutputMLIR, "mlir", "Emit MLIR dialect"),
               clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
               clEnumValN(OutputSplitVerilog, "split-verilog",
                          "Emit Verilog (one file per module; specify "
                          "directory with -o=<dir>)"),
               clEnumValN(OutputDisabled, "disable-output",
                          "Do not output anything")),
    cl::init(OutputMLIR));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<std::string>
    inputAnnotationFilename("annotation-file",
                            cl::desc("Optional input annotation file"),
                            cl::value_desc("filename"));

static cl::opt<std::string> blackBoxRootPath(
    "blackbox-path",
    cl::desc("Optional path to use as the root of black box annotations"),
    cl::value_desc("path"), cl::init(""));

static cl::opt<std::string> blackBoxRootResourcePath(
    "blackbox-resource-path",
    cl::desc(
        "Optional path to use as the root of black box resource annotations"),
    cl::value_desc("path"), cl::init(""));

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return mlir::createCanonicalizerPass(config);
}

/// Process a single buffer of the input.
static LogicalResult
processBuffer(MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
              llvm::function_ref<LogicalResult(ModuleOp)> callback) {
  // Add the annotation file if one was explicitly specified.
  std::string annotationFilenameDetermined;
  if (!inputAnnotationFilename.empty()) {
    if (!(sourceMgr.AddIncludeFile(inputAnnotationFilename, llvm::SMLoc(),
                                   annotationFilenameDetermined))) {
      llvm::errs() << "cannot open input annotation file '"
                   << inputAnnotationFilename
                   << "': No such file or directory\n";
      return failure();
    }
  }

  // Parse the input.
  OwningModuleRef module;
  if (inputFormat == InputFIRFile) {
    auto parserTimer = ts.nest("FIR Parser");
    firrtl::FIRParserOptions options;
    options.ignoreInfoLocators = ignoreFIRLocations;
    module = importFIRFile(sourceMgr, &context, options);
  } else {
    auto parserTimer = ts.nest("MLIR Parser");
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  if (!disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createCSEPass());
  }

  // Width inference creates canonicalization opportunities.
  if (inferWidths)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferWidthsPass());

  // The input mlir file could be firrtl dialect so we might need to clean
  // things up.
  if (lowerTypes) {
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass());
    // Only enable expand whens if lower types is also enabled.
    if (expandWhens) {
      auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
      modulePM.addPass(firrtl::createExpandWhensPass());
    }
  }

  if (checkCombCycles)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCheckCombCyclesPass());

  // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
  if (!disableOptimization) {
    auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
    modulePM.addPass(createSimpleCanonicalizerPass());
  }

  if (inliner)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());

  if (imconstprop)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());

  if (blackBoxMemory)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createBlackBoxMemoryPass());

  // Read black box source files into the IR.
  StringRef blackBoxRoot = blackBoxRootPath.empty()
                               ? llvm::sys::path::parent_path(inputFilename)
                               : blackBoxRootPath;
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createBlackBoxReaderPass(
      blackBoxRoot, blackBoxRootResourcePath.empty()
                        ? blackBoxRoot
                        : blackBoxRootResourcePath));

  if (grandCentral) {
    auto &circuitPM = pm.nest<firrtl::CircuitOp>();
    circuitPM.addPass(firrtl::createGrandCentralPass());
    circuitPM.addPass(firrtl::createGrandCentralTapsPass());
  }

  // Lower if we are going to verilog or if lowering was specifically requested.
  if (lowerToHW || outputFormat == OutputVerilog ||
      outputFormat == OutputSplitVerilog) {
    pm.addPass(createLowerFIRRTLToHWPass(enableAnnotationWarning.getValue()));
    pm.addPass(sv::createHWMemSimImplPass());

    if (extractTestCode)
      pm.addPass(sv::createSVExtractTestCodePass());

    // If enabled, run the optimizer.
    if (!disableOptimization) {
      auto &modulePM = pm.nest<hw::HWModuleOp>();
      modulePM.addPass(sv::createHWCleanupPass());
      modulePM.addPass(createCSEPass());
      modulePM.addPass(createSimpleCanonicalizerPass());
    }
  }

  // Add passes specific to Verilog emission if we're going there.
  if (outputFormat == OutputVerilog || outputFormat == OutputSplitVerilog) {
    // Legalize the module names.
    pm.addPass(sv::createHWLegalizeNamesPass());

    // Tidy up the IR to improve verilog emission quality.
    if (!disableOptimization) {
      auto &modulePM = pm.nest<hw::HWModuleOp>();
      modulePM.addPass(sv::createPrettifyVerilogPass());
    }
  }

  // Load the emitter options from the command line. Command line options if
  // specified will override any module options.
  applyLoweringCLOptions(module.get());

  if (failed(pm.run(module.get())))
    return failure();

  auto outputTimer = ts.nest("Output");

  // Note that we intentionally "leak" the Module into the MLIRContext instead
  // of deallocating it.  There is no need to deallocate it right before
  // process exit.
  return callback(module.release());
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult
processInputSplit(MLIRContext &context, TimingScope &ts,
                  std::unique_ptr<llvm::MemoryBuffer> buffer,
                  llvm::function_ref<LogicalResult(ModuleOp)> emitCallback) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (verifyDiagnostics) {
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    context.printOpOnDiagnostic(false);
    (void)processBuffer(context, ts, sourceMgr, emitCallback);
    return sourceMgrHandler.verify();
  } else {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, emitCallback);
  }
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             llvm::function_ref<LogicalResult(ModuleOp)> emitCallback) {
  if (splitInputFile) {
    // Emit an error if the user provides a separate annotation file alongside
    // split input. This is technically not a problem, but the user likely
    // expects the annotation file to be split as well, which is not the case.
    // To prevent any frustration, we detect this constellation and emit an
    // error here. The user can provide annotations for each split using the
    // inline JSON syntax in FIRRTL.
    if (!inputAnnotationFilename.empty()) {
      llvm::errs() << "annotation file cannot be used with split input: "
                      "use inline JSON syntax on FIRRTL `circuit` to specify "
                      "per-split annotations\n";
      return failure();
    }

    return splitAndProcessBuffer(
        std::move(input),
        [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
          return processInputSplit(context, ts, std::move(buffer),
                                   emitCallback);
        },
        llvm::outs());
  } else {
    return processInputSplit(context, ts, std::move(input), emitCallback);
  }
}

/// This implements the top-level logic for the firtool command, invoked once
/// command line options are parsed and LLVM/MLIR are all set up and ready to
/// go.
static LogicalResult executeFirtool(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Figure out the input format if unspecified.
  if (inputFormat == InputUnspecified) {
    if (StringRef(inputFilename).endswith(".fir"))
      inputFormat = InputFIRFile;
    else if (StringRef(inputFilename).endswith(".mlir"))
      inputFormat = InputMLIRFile;
    else {
      llvm::errs() << "unknown input format: "
                      "specify with -format=fir or -format=mlir\n";
      return failure();
    }
  }

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Create the output directory or output file depending on our mode.
  Optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  if (outputFormat != OutputSplitVerilog) {
    // Create an output file.
    outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
    if (!outputFile.getValue()) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
  } else {
    // Create an output directory.
    if (outputFilename.isDefaultOption() || outputFilename == "-") {
      llvm::errs() << "missing output directory: specify with -o=<dir>\n";
      return failure();
    }
    auto error = llvm::sys::fs::create_directory(outputFilename);
    if (error) {
      llvm::errs() << "cannot create output directory '" << outputFilename
                   << "': " << error.message() << "\n";
      return failure();
    }
  }

  // Emit a single file or multiple files depending on the output format.
  auto emitCallback = [&](ModuleOp module) -> LogicalResult {
    switch (outputFormat) {
    case OutputMLIR:
      module->print(outputFile.getValue()->os());
      return success();
    case OutputDisabled:
      return success();
    case OutputVerilog:
      return exportVerilog(module, outputFile.getValue()->os());
    case OutputSplitVerilog:
      return exportSplitVerilog(module, outputFilename);
    }
    return failure();
  };

  // Register our dialects.
  context.loadDialect<firrtl::FIRRTLDialect, hw::HWDialect, comb::CombDialect,
                      sv::SVDialect>();

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), emitCallback)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.hasValue())
    outputFile.getValue()->keep();

  return success();
}

/// Main driver for firtool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeFirtool'.  This is set up
/// so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  registerLoweringCLOptions();
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR-based FIRRTL compiler\n");

  // -disable-opt turns off constant propagation (unless it was explicitly
  // enabled).
  if (disableOptimization && imconstprop.getNumOccurrences() == 0)
    imconstprop = false;

  MLIRContext context;

  // Do the guts of the firtool process.
  auto result = executeFirtool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
