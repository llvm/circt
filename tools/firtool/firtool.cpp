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
#include "circt/Support/Version.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Chrono.h"
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

static cl::OptionCategory mainCategory("firtool Options");

static cl::opt<InputFormatKind> inputFormat(
    "format", cl::desc("Specify input file format:"),
    cl::values(clEnumValN(InputUnspecified, "autodetect",
                          "Autodetect input format"),
               clEnumValN(InputFIRFile, "fir", "Parse as .fir file"),
               clEnumValN(InputMLIRFile, "mlir", "Parse as .mlir file")),
    cl::init(InputUnspecified), cl::cat(mainCategory));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename, or directory for split output"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(mainCategory));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableOptimization("disable-opt",
                                         cl::desc("disable optimizations"),
                                         cl::cat(mainCategory));

static cl::opt<bool> inliner("inline",
                             cl::desc("Run the FIRRTL module inliner"),
                             cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> enableAnnotationWarning(
    "warn-on-unprocessed-annotations",
    cl::desc("Warn about annotations that were not removed by lower-to-hw"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    emitChiselAssertsAsSVA("emit-chisel-asserts-as-sva",
                           cl::desc("Convert all chisel asserts into SVA"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> disableAnnotationsClassless(
    "disable-annotation-classless",
    cl::desc("Ignore annotations without a class when parsing"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> disableAnnotationsUnknown(
    "disable-annotation-unknown",
    cl::desc("Ignore unknown annotations when parsing"), cl::init(false),
    cl::cat(mainCategory));

static cl::opt<bool>
    emitMetadata("emit-metadata",
                 cl::desc("emit metadata for metadata annotations"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> emitOMIR("emit-omir",
                              cl::desc("emit OMIR annotations to a JSON file"),
                              cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> replSeqMem(
    "repl-seq-mem",
    cl::desc(
        "replace the seq mem for macro replacement and emit relevant metadata"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    preserveAggregate("preserve-aggregate",
                      cl::desc("preserve aggregate types in lower types"),
                      cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> preservePublicTypes(
    "preserve-public-types",
    cl::desc("force to lower ports of toplevel and external modules"),
    cl::init(true), cl::cat(mainCategory));

static cl::opt<std::string>
    replSeqMemCircuit("repl-seq-mem-circuit",
                      cl::desc("circuit root for seq mem metadata"),
                      cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    replSeqMemFile("repl-seq-mem-file",
                   cl::desc("file name for seq mem metadata"), cl::init(""),
                   cl::cat(mainCategory));

static cl::opt<bool>
    ignoreReadEnableMem("ignore-read-enable-mem",
                        cl::desc("ignore the read enable signal, instead of "
                                 "assigning X on read disable"),
                        cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> imconstprop(
    "imconstprop",
    cl::desc(
        "Enable intermodule constant propagation and dead code elimination"),
    cl::init(true), cl::cat(mainCategory));

// TODO: this pass is temporarily off by default, while we migrate over to the
// new memory lowering pipeline.
static cl::opt<bool> lowerMemory("lower-memory",
                                 cl::desc("run the lower-memory pass"),
                                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    lowerTypes("lower-types",
               cl::desc("run the lower-types pass within lower-to-hw"),
               cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> expandWhens("expand-whens",
                                 cl::desc("disable the expand-whens pass"),
                                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    addSeqMemPorts("add-seqmem-ports",
                   cl::desc("add user defined ports to sequential memories"),
                   cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    dedup("dedup", cl::desc("deduplicate structurally identical modules"),
          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    ignoreFIRLocations("ignore-fir-locators",
                       cl::desc("ignore the @info locations in the .fir file"),
                       cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    lowerCHIRRTL("lower-chirrtl",
                 cl::desc("lower CHIRRTL memories to FIRRTL memories"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> wireDFT("wire-dft", cl::desc("wire the DFT ports"),
                             cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    inferWidths("infer-widths",
                cl::desc("run the width inference pass on firrtl"),
                cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    inferResets("infer-resets",
                cl::desc("run the reset inference pass on firrtl"),
                cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    injectDUTHierarchy("inject-dut-hierarchy",
                       cl::desc("add a level of hierarchy to the DUT"),
                       cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    extractInstances("extract-instances",
                     cl::desc("extract black boxes, seq mems, and clock gates"),
                     cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    memToRegOfVec("mem-to-reg-of-vec",
                  cl::desc("convert combinational memories to registers"),
                  cl::init(true));

static cl::opt<bool>
    prefixModules("prefix-modules",
                  cl::desc("prefix modules with NestedPrefixAnnotation"),
                  cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> extractTestCode("extract-test-code",
                                     cl::desc("run the extract test code pass"),
                                     cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    grandCentral("firrtl-grand-central",
                 cl::desc("create interfaces and data/memory taps from SiFive "
                          "Grand Central annotations"),
                 cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> exportModuleHierarchy(
    "export-module-hierarchy",
    cl::desc("export module and instance hierarchy as JSON"), cl::init(false),
    cl::cat(mainCategory));

static cl::opt<bool>
    checkCombCycles("firrtl-check-comb-cycles",
                    cl::desc("check combinational cycles on firrtl"),
                    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    imdeadcodeelim("imdeadcodeelim",
                   cl::desc("inter-module dead code elimination."),
                   cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> mergeConnections(
    "merge-connections",
    cl::desc("merge field-level connections into full aggregate connections"),
    cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    mergeConnectionsAgggresively("merge-connections-aggressive-merging",
                                 cl::desc("merge connections aggressively"),
                                 cl::init(false), cl::cat(mainCategory));

/// Enable the pass to merge the read and write ports of a memory, if their
/// enable conditions are mutually exclusive.
static cl::opt<bool>
    inferMemReadWrite("infer-rw",
                      cl::desc("enable infer read write ports for memory"),
                      cl::init(true), cl::cat(mainCategory));

enum OutputFormatKind {
  OutputParseOnly,
  OutputIRFir,
  OutputIRHW,
  OutputIRSV,
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
        clEnumValN(OutputIRSV, "ir-sv", "Emit SV dialect"),
        clEnumValN(OutputIRVerilog, "ir-verilog",
                   "Emit IR after Verilog lowering"),
        clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
        clEnumValN(OutputSplitVerilog, "split-verilog",
                   "Emit Verilog (one file per module; specify "
                   "directory with -o=<dir>)"),
        clEnumValN(OutputDisabled, "disable-output", "Do not output anything")),
    cl::init(OutputVerilog), cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::list<std::string> inputAnnotationFilenames(
    "annotation-file", cl::desc("Optional input annotation file"),
    cl::CommaSeparated, cl::value_desc("filename"), cl::cat(mainCategory));

static cl::list<std::string> inputOMIRFilenames(
    "omir-file", cl::desc("Optional input object model 2.0 file"),
    cl::CommaSeparated, cl::value_desc("filename"), cl::cat(mainCategory));

static cl::opt<std::string>
    omirOutFile("output-omir", cl::desc("file name for the output omir"),
                cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    mlirOutFile("output-final-mlir",
                cl::desc("Optional file name to output the final MLIR into, in "
                         "addition to the output requested by -o"),
                cl::init(""), cl::value_desc("filename"),
                cl::cat(mainCategory));

static cl::opt<std::string> blackBoxRootPath(
    "blackbox-path",
    cl::desc("Optional path to use as the root of black box annotations"),
    cl::value_desc("path"), cl::init(""), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> stripDebugInfo(
    "strip-debug-info",
    cl::desc("Disable source locator information in output Verilog"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> dropName(
    "drop-names",
    cl::desc("Disable full name preservation by dropping interesting names"),
    cl::init(false), cl::cat(mainCategory));

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return mlir::createCanonicalizerPass(config);
}

// This class prints logs before and after of pass executions. This
// insrumentation assumes that passes are not parallelized for firrtl::CircuitOp
// and mlir::ModuleOp.
class FirtoolPassInstrumentation : public mlir::PassInstrumentation {
  // This stores start time points of passes.
  using TimePoint = llvm::sys::TimePoint<>;
  llvm::SmallVector<TimePoint> timePoints;
  int level = 0;

public:
  void runBeforePass(Pass *pass, Operation *op) override {
    // This assumes that it is safe to log messages to stderr if the operation
    // is circuit or module op.
    if (isa<firrtl::CircuitOp, mlir::ModuleOp>(op)) {
      timePoints.push_back(TimePoint::clock::now());
      auto &os = llvm::errs();
      os << "[firtool] ";
      os.indent(2 * level++);
      os << "Running \"";
      pass->printAsTextualPipeline(llvm::errs());
      os << "\"\n";
    }
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    using namespace std::chrono;
    // This assumes that it is safe to log messages to stderr if the operation
    // is circuit or module op.
    if (isa<firrtl::CircuitOp, mlir::ModuleOp>(op)) {
      auto &os = llvm::errs();
      auto elpased = duration<double>(TimePoint::clock::now() -
                                      timePoints.pop_back_val()) /
                     seconds(1);
      os << "[firtool] ";
      os.indent(2 * --level);
      os << "-- Done in " << llvm::format("%.3f", elpased) << " sec\n";
    }
  }
};

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
  mlir::OwningOpRef<mlir::ModuleOp> module;

  llvm::sys::TimePoint<> parseStartTime;
  if (verbosePassExecutions) {
    llvm::errs() << "[firtool] Running "
                 << (inputFormat == InputFIRFile ? "fir" : "mlir")
                 << " parser\n";
    parseStartTime = llvm::sys::TimePoint<>::clock::now();
  }

  if (inputFormat == InputFIRFile) {
    auto parserTimer = ts.nest("FIR Parser");
    firrtl::FIRParserOptions options;
    options.ignoreInfoLocators = ignoreFIRLocations;
    options.numAnnotationFiles = numAnnotationFiles;
    module = importFIRFile(sourceMgr, &context, parserTimer, options);
  } else {
    auto parserTimer = ts.nest("MLIR Parser");
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  }
  if (!module)
    return failure();

  if (verbosePassExecutions) {
    auto elpased = std::chrono::duration<double>(
                       llvm::sys::TimePoint<>::clock::now() - parseStartTime) /
                   std::chrono::seconds(1);
    llvm::errs() << "[firtool] -- Done in " << llvm::format("%.3f", elpased)
                 << " sec\n";
  }

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
  if (verbosePassExecutions)
    pm.addInstrumentation(std::make_unique<FirtoolPassInstrumentation>());
  applyPassManagerCLOptions(pm);

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerFIRRTLAnnotationsPass(
      disableAnnotationsUnknown, disableAnnotationsClassless));

  // TODO: Move this to the O1 pipeline.
  if (dropName)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createDropNamesPass());

  if (!disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createCSEPass());

  if (injectDUTHierarchy)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createInjectDUTHierarchyPass());

  if (lowerCHIRRTL)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createLowerCHIRRTLPass());

  // Width inference creates canonicalization opportunities.
  if (inferWidths)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferWidthsPass());

  if (memToRegOfVec)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createMemToRegOfVecPass(replSeqMem, ignoreReadEnableMem));

  if (inferResets)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferResetsPass());

  if (!disableOptimization && dedup)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createDedupPass());

  if (wireDFT)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createWireDFTPass());

  if (replSeqMem)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createFlattenMemoryPass());
  // The input mlir file could be firrtl dialect so we might need to clean
  // things up.
  if (lowerTypes) {
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
        preserveAggregate, preservePublicTypes));
    // Only enable expand whens if lower types is also enabled.
    if (expandWhens) {
      auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
      modulePM.addPass(firrtl::createExpandWhensPass());
      modulePM.addPass(firrtl::createSFCCompatPass());
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

  if (replSeqMem && lowerMemory)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerMemoryPass());

  if (prefixModules)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createPrefixModulesPass());

  if (inliner)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());

  if (imconstprop && !disableOptimization)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());

  if (addSeqMemPorts)
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createAddSeqMemPortsPass());

  if (emitMetadata)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCreateSiFiveMetadataPass(
        replSeqMem, replSeqMemCircuit, replSeqMemFile));

  if (extractInstances)
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createExtractInstancesPass());

  // Run passes to resolve Grand Central features.  This should run before
  // BlackBoxReader because Grand Central needs to inform BlackBoxReader where
  // certain black boxes should be placed.
  if (grandCentral) {
    auto &circuitPM = pm.nest<firrtl::CircuitOp>();
    circuitPM.addPass(firrtl::createGrandCentralPass());
    circuitPM.addPass(firrtl::createGrandCentralTapsPass());
    circuitPM.addPass(
        firrtl::createGrandCentralSignalMappingsPass(outputFilename));
  }

  // Read black box source files into the IR.
  StringRef blackBoxRoot = blackBoxRootPath.empty()
                               ? llvm::sys::path::parent_path(inputFilename)
                               : blackBoxRootPath;
  pm.nest<firrtl::CircuitOp>().addPass(
      firrtl::createBlackBoxReaderPass(blackBoxRoot));

  // Drop names introduced by middle-end passes (e.g. GrandCentral).
  // TODO: Move this to the O1 pipeline.
  if (dropName)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createDropNamesPass());

  // The above passes, IMConstProp in particular, introduce additional
  // canonicalization opportunities that we should pick up here before we
  // proceed to output-specific pipelines.
  if (!disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());
    if (imdeadcodeelim)
      pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMDeadCodeElimPass());
  }

  if (emitOMIR)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createEmitOMIRPass(omirOutFile));

  if (!disableOptimization && preserveAggregate && mergeConnections)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createMergeConnectionsPass(mergeConnectionsAgggresively));

  // Lower if we are going to verilog or if lowering was specifically requested.
  if (outputFormat != OutputIRFir) {
    pm.addPass(createLowerFIRRTLToHWPass(enableAnnotationWarning.getValue(),
                                         emitChiselAssertsAsSVA.getValue()));

    if (outputFormat == OutputIRHW) {
      if (!disableOptimization) {
        auto &modulePM = pm.nest<hw::HWModuleOp>();
        modulePM.addPass(createCSEPass());
        modulePM.addPass(createSimpleCanonicalizerPass());
      }
    } else {
      pm.addPass(sv::createHWMemSimImplPass(replSeqMem, ignoreReadEnableMem));

      if (extractTestCode)
        pm.addPass(sv::createSVExtractTestCodePass());

      // If enabled, run the optimizer.
      if (!disableOptimization) {
        auto &modulePM = pm.nest<hw::HWModuleOp>();
        modulePM.addPass(createCSEPass());
        modulePM.addPass(createSimpleCanonicalizerPass());
        modulePM.addPass(createCSEPass());
        modulePM.addPass(sv::createHWCleanupPass());
      }
    }
  }

  // Load the emitter options from the command line. Command line options if
  // specified will override any module options.
  applyLoweringCLOptions(module.get());

  if (failed(pm.run(module.get())))
    return failure();

  // Add passes specific to Verilog emission if we're going there.
  if (outputFormat == OutputVerilog || outputFormat == OutputSplitVerilog ||
      outputFormat == OutputIRVerilog) {
    PassManager exportPm(&context);
    exportPm.enableTiming(ts);
    applyPassManagerCLOptions(exportPm);
    if (verbosePassExecutions)
      exportPm.addInstrumentation(
          std::make_unique<FirtoolPassInstrumentation>());
    // Legalize unsupported operations within the modules.
    exportPm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

    // Tidy up the IR to improve verilog emission quality.
    if (!disableOptimization) {
      auto &modulePM = exportPm.nest<hw::HWModuleOp>();
      modulePM.addPass(sv::createPrettifyVerilogPass());
    }

    if (stripDebugInfo)
      exportPm.addPass(mlir::createStripDebugInfoPass());

    // Emit a single file or multiple files depending on the output format.
    switch (outputFormat) {
    default:
      llvm_unreachable("can't reach this");
    case OutputVerilog:
      exportPm.addPass(createExportVerilogPass(outputFile.getValue()->os()));
      break;
    case OutputSplitVerilog:
      exportPm.addPass(createExportSplitVerilogPass(outputFilename));
      break;
    case OutputIRVerilog:
      // Run the ExportVerilog pass to get its lowering, but discard the output.
      exportPm.addPass(createExportVerilogPass(llvm::nulls()));
      break;
    }

    // Run module hierarchy emission after verilog emission, which ensures we
    // pick up any changes that verilog emission made.
    if (exportModuleHierarchy)
      exportPm.addPass(sv::createHWExportModuleHierarchyPass(outputFilename));

    // Load the emitter options from the command line. Command line options if
    // specified will override any module options.
    applyLoweringCLOptions(module.get());

    if (failed(exportPm.run(module.get())))
      return failure();
  }

  if (outputFormat == OutputIRFir || outputFormat == OutputIRHW ||
      outputFormat == OutputIRSV || outputFormat == OutputIRVerilog) {
    auto outputTimer = ts.nest("Print .mlir output");
    module->print(outputFile.getValue()->os());
  }

  // If requested, print the final MLIR into mlirOutFile.
  if (!mlirOutFile.empty()) {
    std::string mlirOutError;
    auto mlirFile = openOutputFile(mlirOutFile, &mlirOutError);
    if (!mlirFile) {
      llvm::errs() << mlirOutError;
      return failure();
    }

    module->print(mlirFile->os());
    mlirFile->keep();
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
    auto error = llvm::sys::fs::create_directories(outputFilename);
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

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  registerLoweringCLOptions();
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });
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
