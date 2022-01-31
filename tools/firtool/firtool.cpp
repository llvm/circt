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

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LoweringOptions.h"
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
#include "llvm/Support/FileSystem.h"
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

static cl::opt<bool> enableAnnotationWarning(
    "warn-on-unprocessed-annotations",
    cl::desc("Warn about annotations that were not removed by lower-to-hw"),
    cl::init(false));

static cl::opt<bool> disableAnnotationsClassless(
    "disable-annotation-classless",
    cl::desc("Ignore annotations without a class when parsing"),
    cl::init(false));

static cl::opt<bool> disableAnnotationsUnknown(
    "disable-annotation-unknown",
    cl::desc("Ignore unknown annotations when parsing"), cl::init(false));

static cl::opt<bool>
    emitMetadata("emit-metadata",
                 cl::desc("emit metadata for metadata annotations"),
                 cl::init(true));

static cl::opt<bool> emitOMIR("emit-omir",
                              cl::desc("emit OMIR annotations to a JSON file"),
                              cl::init(true));

static cl::opt<bool> replSeqMem(
    "repl-seq-mem",
    cl::desc(
        "replace the seq mem for macro replacement and emit relevant metadata"),
    cl::init(false));
static cl::opt<bool>
    preserveAggregate("preserve-aggregate",
                      cl::desc("preserve aggregate types in lower types"),
                      cl::init(false));

static cl::opt<bool> preservePublicTypes(
    "preserve-public-types",
    cl::desc("force to lower ports of toplevel and external modules"),
    cl::init(true));

static cl::opt<std::string>
    replSeqMemCircuit("repl-seq-mem-circuit",
                      cl::desc("circuit root for seq mem metadata"),
                      cl::init(""));
static cl::opt<std::string>
    replSeqMemFile("repl-seq-mem-file",
                   cl::desc("file name for seq mem metadata"), cl::init(""));

static cl::opt<bool>
    ignoreReadEnableMem("ignore-read-enable-mem",
                        cl::desc("ignore the read enable signal, instead of "
                                 "assigning X on read disable"),
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
    lowerCHIRRTL("lower-chirrtl",
                 cl::desc("lower CHIRRTL memories to FIRRTL memories"),
                 cl::init(true));

static cl::opt<bool> wireDFT("wire-dft", cl::desc("wire the DFT ports"),
                             cl::init(true));

static cl::opt<bool>
    inferWidths("infer-widths",
                cl::desc("run the width inference pass on firrtl"),
                cl::init(true));

static cl::opt<bool>
    inferResets("infer-resets",
                cl::desc("run the reset inference pass on firrtl"),
                cl::init(true));

static cl::opt<bool>
    prefixModules("prefix-modules",
                  cl::desc("prefix modules with NestedPrefixAnnotation"),
                  cl::init(true));

static cl::opt<bool> extractTestCode("extract-test-code",
                                     cl::desc("run the extract test code pass"),
                                     cl::init(false));
static cl::opt<bool>
    grandCentral("firrtl-grand-central",
                 cl::desc("create interfaces and data/memory taps from SiFive "
                          "Grand Central annotations"),
                 cl::init(false));
static cl::opt<bool> exportModuleHierarchy(
    "export-module-hierarchy",
    cl::desc("export module and instance hierarchy as JSON"), cl::init(false));

static cl::opt<bool>
    checkCombCycles("firrtl-check-comb-cycles",
                    cl::desc("check combinational cycles on firrtl"),
                    cl::init(false));
static cl::opt<bool> newAnno("new-anno",
                             cl::desc("enable new annotation handling"),
                             cl::init(false));
static cl::opt<bool> removeUnusedPorts("remove-unused-ports",
                                       cl::desc("enable unused ports pruning"),
                                       cl::init(true));

/// Enable the pass to merge the read and write ports of a memory, if their
/// enable conditions are mutually exclusive.
static cl::opt<bool>
    inferMemReadWrite("infer-rw",
                      cl::desc("enable infer read write ports for memory"),
                      cl::init(true));

enum OutputFormatKind {
  OutputParseOnly,
  OutputIRFir,
  OutputIRHW,
  OutputIRVerilog,
  OutputVerilog,
  OutputSplitVerilog,
  OutputDisabled
};

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(
        clEnumValN(OutputParseOnly, "parse-only",
                   "Emit FIR dialect after parsing"),
        clEnumValN(OutputIRFir, "ir-fir", "Emit FIR dialect after pipeline"),
        clEnumValN(OutputIRHW, "ir-hw", "Emit HW dialect"),
        clEnumValN(OutputIRVerilog, "ir-verilog",
                   "Emit IR after Verilog lowering"),
        clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
        clEnumValN(OutputSplitVerilog, "split-verilog",
                   "Emit Verilog (one file per module; specify "
                   "directory with -o=<dir>)"),
        clEnumValN(OutputDisabled, "disable-output", "Do not output anything")),
    cl::init(OutputVerilog));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::list<std::string>
    inputAnnotationFilenames("annotation-file",
                             cl::desc("Optional input annotation file"),
                             cl::CommaSeparated, cl::value_desc("filename"));

static cl::list<std::string>
    inputOMIRFilenames("omir-file",
                       cl::desc("Optional input object model 2.0 file"),
                       cl::CommaSeparated, cl::value_desc("filename"));

static cl::opt<std::string>
    omirOutFile("output-omir", cl::desc("file name for the output omir"),
                cl::init(""));

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
              Optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Add the annotation file if one was explicitly specified.
  unsigned numAnnotationFiles = 0;
  for (auto inputAnnotationFilename : inputAnnotationFilenames) {
    std::string annotationFilenameDetermined;
    if (!sourceMgr.AddIncludeFile(inputAnnotationFilename, llvm::SMLoc(),
                                  annotationFilenameDetermined)) {
      llvm::errs() << "cannot open input annotation file '"
                   << inputAnnotationFilename
                   << "': No such file or directory\n";
      return failure();
    }
    ++numAnnotationFiles;
  }

  for (auto file : inputOMIRFilenames) {
    std::string filename;
    if (!sourceMgr.AddIncludeFile(file, llvm::SMLoc(), filename)) {
      llvm::errs() << "cannot open input annotation file '" << file
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
    options.rawAnnotations = newAnno;
    options.numAnnotationFiles = numAnnotationFiles;
    module = importFIRFile(sourceMgr, &context, options);
  } else {
    auto parserTimer = ts.nest("MLIR Parser");
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // If the user asked for just a parse, stop here.
  if (outputFormat == OutputParseOnly) {
    mlir::ModuleOp theModule = module.release();
    auto outputTimer = ts.nest("Print .mlir output");
    theModule->print(outputFile.getValue()->os());
    return success();
  }

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  if (newAnno)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createLowerFIRRTLAnnotationsPass(disableAnnotationsUnknown,
                                                 disableAnnotationsClassless));

  if (!disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createCSEPass());
  }

  if (lowerCHIRRTL)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createLowerCHIRRTLPass());

  // Width inference creates canonicalization opportunities.
  if (inferWidths)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferWidthsPass());

  if (inferResets)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferResetsPass());

  if (wireDFT)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createWireDFTPass());

  if (prefixModules)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createPrefixModulesPass());

  if (blackBoxMemory)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createBlackBoxMemoryPass());

  // The input mlir file could be firrtl dialect so we might need to clean
  // things up.
  if (lowerTypes) {
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
        replSeqMem, preserveAggregate, preservePublicTypes));
    // Only enable expand whens if lower types is also enabled.
    if (expandWhens) {
      auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
      modulePM.addPass(firrtl::createExpandWhensPass());
      modulePM.addPass(firrtl::createRemoveResetsPass());
    }
  }

  if (checkCombCycles)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCheckCombCyclesPass());

  // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
  if (!disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());

  // Run the infer-rw pass, which merges read and write ports of a memory with
  // mutually exclusive enables.
  if (inferMemReadWrite)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createInferReadWritePass());

  if (inliner)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());

  bool nonConstAsyncResetValueIsError = false;
  if (imconstprop && !disableOptimization) {
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());
    nonConstAsyncResetValueIsError = true;
  }

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
    circuitPM.nest<firrtl::FModuleOp>().addPass(
        firrtl::createGrandCentralSignalMappingsPass());
  }

  // The above passes, IMConstProp in particular, introduce additional
  // canonicalization opportunities that we should pick up here before we
  // proceed to output-specific pipelines.
  if (!disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());
    if (removeUnusedPorts)
      pm.nest<firrtl::CircuitOp>().addPass(
          firrtl::createRemoveUnusedPortsPass());
  }

  if (emitMetadata)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCreateSiFiveMetadataPass(
        replSeqMem, replSeqMemCircuit, replSeqMemFile));

  if (emitOMIR)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createEmitOMIRPass(omirOutFile));

  // Lower if we are going to verilog or if lowering was specifically requested.
  if (outputFormat != OutputIRFir) {
    pm.addPass(createLowerFIRRTLToHWPass(enableAnnotationWarning.getValue(),
                                         nonConstAsyncResetValueIsError));
    pm.addPass(sv::createHWMemSimImplPass(replSeqMem, ignoreReadEnableMem));

    if (extractTestCode)
      pm.addPass(sv::createSVExtractTestCodePass());

    // If enabled, run the optimizer.
    if (!disableOptimization) {
      auto &modulePM = pm.nest<hw::HWModuleOp>();
      modulePM.addPass(createCSEPass());
      modulePM.addPass(createSimpleCanonicalizerPass());
      modulePM.addPass(sv::createHWCleanupPass());
    }
  }

  // Add passes specific to Verilog emission if we're going there.
  if (outputFormat == OutputVerilog || outputFormat == OutputSplitVerilog ||
      outputFormat == OutputIRVerilog) {
    // Legalize unsupported operations within the modules.
    pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

    // Tidy up the IR to improve verilog emission quality.
    if (!disableOptimization) {
      auto &modulePM = pm.nest<hw::HWModuleOp>();
      modulePM.addPass(sv::createPrettifyVerilogPass());
    }

    // Emit a single file or multiple files depending on the output format.
    switch (outputFormat) {
    default:
      llvm_unreachable("can't reach this");
    case OutputVerilog:
      pm.addPass(createExportVerilogPass(outputFile.getValue()->os()));
      break;
    case OutputSplitVerilog:
      pm.addPass(createExportSplitVerilogPass(outputFilename));
      break;
    case OutputIRVerilog:
      // Run the ExportVerilog pass to get its lowering, but discard the output.
      pm.addPass(createExportVerilogPass(llvm::nulls()));
      break;
    }

    // Run module hierarchy emission after verilog emission, which ensures we
    // pick up any changes that verilog emission made.
    if (exportModuleHierarchy)
      pm.addPass(sv::createHWExportModuleHierarchyPass(outputFilename));
  }

  // Load the emitter options from the command line. Command line options if
  // specified will override any module options.
  applyLoweringCLOptions(module.get());

  if (failed(pm.run(module.get())))
    return failure();

  if (outputFormat == OutputIRFir || outputFormat == OutputIRHW ||
      outputFormat == OutputIRVerilog) {
    auto outputTimer = ts.nest("Print .mlir output");
    module->print(outputFile.getValue()->os());
  }

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult
processInputSplit(MLIRContext &context, TimingScope &ts,
                  std::unique_ptr<llvm::MemoryBuffer> buffer,
                  Optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             Optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  // Emit an error if the user provides a separate annotation file alongside
  // split input. This is technically not a problem, but the user likely
  // expects the annotation file to be split as well, which is not the case.
  // To prevent any frustration, we detect this constellation and emit an
  // error here. The user can provide annotations for each split using the
  // inline JSON syntax in FIRRTL.
  if (!inputAnnotationFilenames.empty()) {
    llvm::errs() << "annotation file cannot be used with split input: "
                    "use inline JSON syntax on FIRRTL `circuit` to specify "
                    "per-split annotations\n";
    return failure();
  }

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
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

  // Register our dialects.
  context.loadDialect<chirrtl::CHIRRTLDialect, firrtl::FIRRTLDialect,
                      hw::HWDialect, comb::CombDialect, sv::SVDialect>();

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
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

  MLIRContext context;

  // Do the guts of the firtool process.
  auto result = executeFirtool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
