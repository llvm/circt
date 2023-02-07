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
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
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
#include "llvm/Support/PrettyStackTrace.h"
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
    cl::values(
        clEnumValN(InputUnspecified, "autodetect", "Autodetect input format"),
        clEnumValN(InputFIRFile, "fir", "Parse as .fir file"),
        clEnumValN(InputMLIRFile, "mlir", "Parse as .mlir or .mlirbc file")),
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

static cl::list<std::string> includeDirs(
    "include-dir",
    cl::desc("Directory to search in when resolving source references"),
    cl::value_desc("directory"), cl::cat(mainCategory));
static cl::alias includeDirsShort(
    "I", cl::desc("Alias for --include-dir.  Example: -I<directory>"),
    cl::aliasopt(includeDirs), cl::Prefix, cl::NotHidden,
    cl::cat(mainCategory));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableOptimization("disable-opt",
                                         cl::desc("Disable optimizations"),
                                         cl::cat(mainCategory));

static cl::opt<bool> disableInliner("disable-inliner",
                                    cl::desc("Disable the Inliner pass"),
                                    cl::init(false), cl::Hidden,
                                    cl::cat(mainCategory));

static cl::opt<bool> enableAnnotationWarning(
    "warn-on-unprocessed-annotations",
    cl::desc("Warn about annotations that were not removed by lower-to-hw"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    emitChiselAssertsAsSVA("emit-chisel-asserts-as-sva",
                           cl::desc("Convert all chisel asserts into SVA"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> addMuxPragmas("add-mux-pragmas",
                                   cl::desc("Annotate mux pragmas"),
                                   cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    stripMuxPragmas("strip-mux-pragmas",
                    cl::desc("Strip mux pragmas. This option was deprecated "
                             "since mux pragma annotatations are "
                             "not emitted by default"),
                    cl::init(true), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableAnnotationsClassless(
    "disable-annotation-classless",
    cl::desc("Ignore annotations without a class when parsing"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> disableAnnotationsUnknown(
    "disable-annotation-unknown",
    cl::desc("Ignore unknown annotations when parsing"), cl::init(false),
    cl::cat(mainCategory));

static cl::opt<bool> lowerAnnotationsNoRefTypePorts(
    "lower-annotations-no-ref-type-ports",
    cl::desc("Create real ports instead of ref type ports when resolving "
             "wiring problems inside the LowerAnnotations pass"),
    cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    emitMetadata("emit-metadata",
                 cl::desc("Emit metadata for metadata annotations"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> emitOMIR("emit-omir",
                              cl::desc("Emit OMIR annotations to a JSON file"),
                              cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> replSeqMem(
    "repl-seq-mem",
    cl::desc(
        "Replace the seq mem for macro replacement and emit relevant metadata"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    lowerMemories("lower-memories",
                  cl::desc("Lower memories to have memories with masks as an "
                           "array with one memory per ground type"),
                  cl::init(false), cl::cat(mainCategory));

static cl::opt<circt::firrtl::PreserveAggregate::PreserveMode>
    preserveAggregate(
        "preserve-aggregate", cl::desc("Specify input file format:"),
        llvm::cl::values(clEnumValN(circt::firrtl::PreserveAggregate::None,
                                    "none", "Preserve no aggregate"),
                         clEnumValN(circt::firrtl::PreserveAggregate::OneDimVec,
                                    "1d-vec",
                                    "Preserve only 1d vectors of ground type"),
                         clEnumValN(circt::firrtl::PreserveAggregate::Vec,
                                    "vec", "Preserve only vectors"),
                         clEnumValN(circt::firrtl::PreserveAggregate::All,
                                    "all", "Preserve vectors and bundles")),
        cl::init(circt::firrtl::PreserveAggregate::None),
        cl::cat(mainCategory));

static cl::opt<bool> preservePublicTypes(
    "preserve-public-types",
    cl::desc("Force to lower ports of toplevel and external modules"),
    cl::init(true), cl::cat(mainCategory));

static cl::opt<firrtl::PreserveValues::PreserveMode>
    preserveMode("preserve-values",
                 cl::desc("Specify the values which can be optimized away"),
                 cl::values(clEnumValN(firrtl::PreserveValues::None, "none",
                                       "Preserve no values"),
                            clEnumValN(firrtl::PreserveValues::Named, "named",
                                       "Preserve values with meaningful names"),
                            clEnumValN(firrtl::PreserveValues::All, "all",
                                       "Preserve all values")),
                 cl::init(firrtl::PreserveValues::None), cl::cat(mainCategory));

static cl::opt<std::string>
    replSeqMemCircuit("repl-seq-mem-circuit",
                      cl::desc("Circuit root for seq mem metadata"),
                      cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    replSeqMemFile("repl-seq-mem-file",
                   cl::desc("File name for seq mem metadata"), cl::init(""),
                   cl::cat(mainCategory));

static cl::opt<bool>
    ignoreReadEnableMem("ignore-read-enable-mem",
                        cl::desc("Ignore the read enable signal, instead of "
                                 "assigning X on read disable"),
                        cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> disableIMCP("disable-imcp",
                                 cl::desc("Disable the IMCP pass"),
                                 cl::init(false), cl::Hidden,
                                 cl::cat(mainCategory));

static cl::opt<bool>
    disableLowerMemory("disable-lower-memory",
                       cl::desc("Disable the LowerMemory pass"),
                       cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableLowerTypes("disable-lower-types",
                                       cl::desc("Disable the LowerTypes pass"),
                                       cl::init(false), cl::Hidden,
                                       cl::cat(mainCategory));

static cl::opt<bool>
    disableExpandWhens("disable-expand-whens",
                       cl::desc("Disable the ExpandWhens pass"),
                       cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    disableAddSeqMemPorts("disable-add-seqmem-ports",
                          cl::desc("Disable the AddSeqMemPorts pass"),
                          cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    dedup("dedup", cl::desc("Deduplicate structurally identical modules"),
          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    ignoreFIRLocations("ignore-fir-locators",
                       cl::desc("Ignore the @info locations in the .fir file"),
                       cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    disableLowerChirrtl("disable-lower-chirrtl",
                        cl::desc("Disable the LowerCHIRRTL pass"),
                        cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableWireDFT("disable-wire-dft",
                                    cl::desc("Disable the WireDFT pass"),
                                    cl::init(false), cl::Hidden,
                                    cl::cat(mainCategory));

static cl::opt<bool>
    disableInferWidths("disable-infer-widths",
                       cl::desc("Disable the InferWidths pass"),
                       cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    disableInferResets("disable-infer-resets",
                       cl::desc("Disable the InferResets pass"),
                       cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> exportChiselInterface(
    "export-chisel-interface",
    cl::desc("Generate a Scala Chisel interface to the top level "
             "module of the firrtl circuit"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string> chiselInterfaceOutDirectory(
    "chisel-interface-out-dir",
    cl::desc("The output directory for generated Chisel interface files"),
    cl::init(""), cl::cat(mainCategory));

static cl::opt<bool>
    disableInjectDutHierarchy("disable-inject-dut-hierarchy",
                              cl::desc("Disable the InjectDutHierarchy pass"),
                              cl::init(false), cl::Hidden,
                              cl::cat(mainCategory));

static cl::opt<bool>
    disableExtractInstances("disable-extract-instances",
                            cl::desc("Disable the ExtractInstances pass"),
                            cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    disableMemToRegOfVec("disable-mem-to-reg-of-vec",
                         cl::desc("Disable the MemToRegOfVec pass"),
                         cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    disablePrefixModules("disable-prefix-modules",
                         cl::desc("Disable the PrefixModules pass"),
                         cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> extractTestCode("extract-test-code",
                                     cl::desc("Run the extract test code pass"),
                                     cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    disableGrandCentral("disable-grand-central",
                        cl::desc("Disable the Grand Central passes"),
                        cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> grandCentralInstantiateCompanionOnly(
    "grand-central-instantiate-companion",
    cl::desc(
        "Run Grand Central in a mode where the companion module is "
        "instantiated and not bound in and the interface is dropped.  This is "
        "intended for situations where there is useful assertion logic inside "
        "the companion, but you don't care about the actual interface."),
    cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> exportModuleHierarchy(
    "export-module-hierarchy",
    cl::desc("Export module and instance hierarchy as JSON"), cl::init(false),
    cl::cat(mainCategory));

static cl::opt<bool>
    disableCheckCombCycles("disable-check-comb-cycles",
                           cl::desc("Disable the CheckCombCycles pass"),
                           cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> useOldCheckCombCycles(
    "use-old-check-comb-cycles",
    cl::desc("Use old CheckCombCycles pass, that does not support aggregates"),
    cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableIMDCE("disable-imdce",
                                  cl::desc("Disable the IMDCE pass"),
                                  cl::init(false), cl::Hidden,
                                  cl::cat(mainCategory));

static cl::opt<bool>
    disableMergeConnections("disable-merge-connections",
                            cl::desc("Disable the MergeConnections pass"),
                            cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableAggressiveMergeConnections(
    "disable-aggressive-merge-connections",
    cl::desc("Disable aggressive merge connections (i.e. merge all field-level "
             "connections into bulk connections)"),
    cl::init(false), cl::cat(mainCategory));

/// Enable the pass to merge the read and write ports of a memory, if their
/// enable conditions are mutually exclusive.
static cl::opt<bool> disableInferRW("disable-infer-rw",
                                    cl::desc("Disable the InferRW pass"),
                                    cl::init(false), cl::Hidden,
                                    cl::cat(mainCategory));

static cl::opt<bool> etcDisableInstanceExtraction(
    "etc-disable-instance-extraction",
    cl::desc("Disable extracting instances only that feed test code"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> etcDisableModuleInlining(
    "etc-disable-module-inlining",
    cl::desc("Disable inlining modules that only feed test code"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> addVivadoRAMAddressConflictSynthesisBugWorkaround(
    "add-vivado-ram-address-conflict-synthesis-bug-workaround",
    cl::desc(
        "Add a vivado specific SV attribute (* ram_style = \"distributed\" *) "
        "to array registers as a workaronud for a vivado synthesis bug that "
        "incorrectly modifies address conflict behavivor of combinational "
        "memories"),
    cl::init(false), cl::cat(mainCategory));

enum class RandomKind { None, Mem, Reg, All };

static cl::opt<RandomKind> disableRandom(
    cl::desc("Disable random initialization code (may break semantics!)"),
    cl::values(clEnumValN(RandomKind::Mem, "disable-mem-randomization",
                          "Disable emission of memory randomization code"),
               clEnumValN(RandomKind::Reg, "disable-reg-randomization",
                          "Disable emission of register randomization code"),
               clEnumValN(RandomKind::All, "disable-all-randomization",
                          "Disable emission of all randomization code")),
    cl::init(RandomKind::None), cl::cat(mainCategory));

static bool isRandomEnabled(RandomKind kind) {
  return disableRandom != RandomKind::All && disableRandom != kind;
}

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
                   "Emit FIR dialect after parsing, verification, and "
                   "annotation lowering"),
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

static cl::opt<std::string> outputAnnotationFilename(
    "output-annotation-file", cl::desc("Optional output annotation file"),
    cl::CommaSeparated, cl::value_desc("filename"), cl::cat(mainCategory));

static cl::list<std::string> inputOMIRFilenames(
    "omir-file", cl::desc("Optional input object model 2.0 file"),
    cl::CommaSeparated, cl::value_desc("filename"), cl::cat(mainCategory));

static cl::opt<std::string>
    omirOutFile("output-omir", cl::desc("File name for the output omir"),
                cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    mlirOutFile("output-final-mlir",
                cl::desc("Optional file name to output the final MLIR into, in "
                         "addition to the output requested by -o"),
                cl::init(""), cl::value_desc("filename"),
                cl::cat(mainCategory));

static cl::opt<bool>
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string> blackBoxRootPath(
    "blackbox-path",
    cl::desc("Optional path to use as the root of black box annotations"),
    cl::value_desc("path"), cl::init(""), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> stripFirDebugInfo(
    "strip-fir-debug-info",
    cl::desc("Disable source fir locator information in output Verilog"),
    cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> stripDebugInfo(
    "strip-debug-info",
    cl::desc("Disable source locator information in output Verilog"),
    cl::init(false), cl::cat(mainCategory));

// Build mode options.
enum BuildMode { BuildModeDebug, BuildModeRelease };
static cl::opt<BuildMode> buildMode(
    "O", cl::desc("Controls how much optimization should be performed"),
    cl::values(clEnumValN(BuildModeDebug, "debug",
                          "Compile with only necessary optimizations"),
               clEnumValN(BuildModeRelease, "release",
                          "Compile with optimizations")),
    cl::init(BuildModeRelease), cl::cat(mainCategory),
    cl::callback([](const BuildMode &buildMode) {
      switch (buildMode) {
      case BuildModeDebug:
        preserveMode = firrtl::PreserveValues::Named;
        break;
      case BuildModeRelease:
        preserveMode = firrtl::PreserveValues::None;
        break;
      }
    }));

static LoweringOptionsOption loweringOptions(mainCategory);

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
      auto elapsed = duration<double>(TimePoint::clock::now() -
                                      timePoints.pop_back_val()) /
                     seconds(1);
      os << "[firtool] ";
      os.indent(2 * --level);
      os << "-- Done in " << llvm::format("%.3f", elapsed) << " sec\n";
    }
  }
};

/// Check output stream before writing bytecode to it.
/// Warn and return true if output is known to be displayed.
static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    errs() << "WARNING: You're attempting to print out a bytecode file.\n"
              "This is inadvisable as it may cause display problems. If\n"
              "you REALLY want to taste MLIR bytecode first-hand, you\n"
              "can force output with the `-f' option.\n\n";
    return true;
  }
  return false;
}

/// Print the operation to the specified stream, emitting bytecode when
/// requested and politely avoiding dumping to terminal unless forced.
static void printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    writeBytecodeToFile(op, os, mlir::BytecodeWriterConfig(getCirctVersion()));
  else
    op->print(os);
}

/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Add the annotation file if one was explicitly specified.
  unsigned numAnnotationFiles = 0;
  for (const auto &inputAnnotationFilename : inputAnnotationFilenames) {
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

  for (const auto &file : inputOMIRFilenames) {
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
    auto elapsed = std::chrono::duration<double>(
                       llvm::sys::TimePoint<>::clock::now() - parseStartTime) /
                   std::chrono::seconds(1);
    llvm::errs() << "[firtool] -- Done in " << llvm::format("%.3f", elapsed)
                 << " sec\n";
  }

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (verbosePassExecutions)
    pm.addInstrumentation(std::make_unique<FirtoolPassInstrumentation>());
  applyPassManagerCLOptions(pm);

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerFIRRTLAnnotationsPass(
      disableAnnotationsUnknown, disableAnnotationsClassless,
      lowerAnnotationsNoRefTypePorts));

  // If the user asked for --parse-only, stop after running LowerAnnotations.
  if (outputFormat == OutputParseOnly) {
    if (failed(pm.run(module.get())))
      return failure();
    auto outputTimer = ts.nest("Print .mlir output");
    printOp(*module, (*outputFile)->os());
    return success();
  }

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerIntrinsicsPass());

  // TODO: Move this to the O1 pipeline.
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createDropNamesPass(preserveMode));

  if (!disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createCSEPass());

  if (!disableInjectDutHierarchy)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createInjectDUTHierarchyPass());

  if (!disableLowerChirrtl)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createLowerCHIRRTLPass());

  // Width inference creates canonicalization opportunities.
  if (!disableInferWidths)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferWidthsPass());

  if (!disableMemToRegOfVec)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createMemToRegOfVecPass(replSeqMem, ignoreReadEnableMem));

  if (!disableInferResets)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferResetsPass());

  if (exportChiselInterface) {
    if (chiselInterfaceOutDirectory.empty()) {
      pm.nest<firrtl::CircuitOp>().addPass(createExportChiselInterfacePass());
    } else {
      pm.nest<firrtl::CircuitOp>().addPass(
          createExportSplitChiselInterfacePass(chiselInterfaceOutDirectory));
    }
  }

  if (!disableOptimization && dedup)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createDedupPass());

  if (!disableWireDFT)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createWireDFTPass());

  if (!lowerMemories)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createFlattenMemoryPass());

  // The input mlir file could be firrtl dialect so we might need to clean
  // things up.
  if (!disableLowerTypes) {
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
        preserveAggregate, preservePublicTypes));
    // Only enable expand whens if lower types is also enabled.
    if (!disableExpandWhens) {
      auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
      modulePM.addPass(firrtl::createExpandWhensPass());
      modulePM.addPass(firrtl::createSFCCompatPass());
    }
  }

  if (!disableInliner)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());

  // Preset the random initialization parameters for each module. The current
  // implementation assumes it can run at a time where every register is
  // currently in the final module it will be emitted in, all registers have
  // been created, and no registers have yet been removed.
  if (isRandomEnabled(RandomKind::Reg))
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createRandomizeRegisterInitPass());

  if (!disableCheckCombCycles) {
    if (useOldCheckCombCycles) {
      if (preserveAggregate == firrtl::PreserveAggregate::None)
        pm.nest<firrtl::CircuitOp>().addPass(
            firrtl::createCheckCombCyclesPass());
      else
        emitWarning(module->getLoc())
            << "CheckCombCyclesPass doens't support aggregate "
               "values yet so it is skipped\n";
    } else
      pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCheckCombLoopsPass());
  }

  // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
  if (!disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());

  // Run the infer-rw pass, which merges read and write ports of a memory with
  // mutually exclusive enables.
  if (!disableInferRW)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createInferReadWritePass());

  if (replSeqMem && !disableLowerMemory)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerMemoryPass());

  if (!disablePrefixModules)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createPrefixModulesPass());

  if (!disableIMCP && !disableOptimization)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());

  if (!disableAddSeqMemPorts)
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createAddSeqMemPortsPass());

  if (emitMetadata)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCreateSiFiveMetadataPass(
        replSeqMem, replSeqMemCircuit, replSeqMemFile));

  if (!disableExtractInstances)
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createExtractInstancesPass());

  // Run passes to resolve Grand Central features.  This should run before
  // BlackBoxReader because Grand Central needs to inform BlackBoxReader where
  // certain black boxes should be placed.  Note: all Grand Central Taps related
  // collateral is resolved entirely by LowerAnnotations.
  if (!disableGrandCentral)
    pm.addNestedPass<firrtl::CircuitOp>(
        firrtl::createGrandCentralPass(grandCentralInstantiateCompanionOnly));

  // Read black box source files into the IR.
  StringRef blackBoxRoot = blackBoxRootPath.empty()
                               ? llvm::sys::path::parent_path(inputFilename)
                               : blackBoxRootPath;
  pm.nest<firrtl::CircuitOp>().addPass(
      firrtl::createBlackBoxReaderPass(blackBoxRoot));

  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createDropNamesPass(preserveMode));

  // Run SymbolDCE as late as possible, but before InnerSymbolDCE. This is for
  // hierpathop's and just for general cleanup.
  pm.addNestedPass<firrtl::CircuitOp>(mlir::createSymbolDCEPass());

  // Run InnerSymbolDCE as late as possible, but before IMDCE.
  pm.addPass(firrtl::createInnerSymbolDCEPass());

  // The above passes, IMConstProp in particular, introduce additional
  // canonicalization opportunities that we should pick up here before we
  // proceed to output-specific pipelines.
  if (!disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());
    if (!disableIMDCE)
      pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMDeadCodeElimPass());
  }

  if (emitOMIR)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createEmitOMIRPass(omirOutFile));

  if (!disableOptimization &&
      preserveAggregate != firrtl::PreserveAggregate::None &&
      !disableMergeConnections)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createMergeConnectionsPass(
            !disableAggressiveMergeConnections.getValue()));

  // Lower if we are going to verilog or if lowering was specifically requested.
  if (outputFormat != OutputIRFir) {

    // Remove TraceAnnotations and write their updated paths to an output
    // annotation file.
    if (outputAnnotationFilename.empty())
      pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolveTracesPass());
    else
      pm.nest<firrtl::CircuitOp>().addPass(
          firrtl::createResolveTracesPass(outputAnnotationFilename.getValue()));

    // Lower the ref.resolve and ref.send ops and remove the RefType ports.
    // LowerToHW cannot handle RefType so, this pass must be run to remove all
    // RefType ports and ops.
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerXMRPass());

    pm.addPass(createLowerFIRRTLToHWPass(
        enableAnnotationWarning.getValue(), emitChiselAssertsAsSVA.getValue(),
        addMuxPragmas.getValue(), !isRandomEnabled(RandomKind::Mem),
        !isRandomEnabled(RandomKind::Reg)));

    if (outputFormat == OutputIRHW) {
      if (!disableOptimization) {
        auto &modulePM = pm.nest<hw::HWModuleOp>();
        modulePM.addPass(createCSEPass());
        modulePM.addPass(createSimpleCanonicalizerPass());
      }
    } else {
      // If enabled, run the optimizer.
      if (!disableOptimization) {
        auto &modulePM = pm.nest<hw::HWModuleOp>();
        modulePM.addPass(createCSEPass());
        modulePM.addPass(createSimpleCanonicalizerPass());
        modulePM.addPass(createCSEPass());
      }

      pm.nest<hw::HWModuleOp>().addPass(seq::createSeqFIRRTLLowerToSVPass(
          {/*disableRandomization=*/!isRandomEnabled(RandomKind::Reg),
           /*addVivadoRAMAddressConflictSynthesisBugWorkaround=*/
           addVivadoRAMAddressConflictSynthesisBugWorkaround}));
      pm.addPass(sv::createHWMemSimImplPass(
          replSeqMem, ignoreReadEnableMem, addMuxPragmas,
          !isRandomEnabled(RandomKind::Mem), !isRandomEnabled(RandomKind::Reg),
          addVivadoRAMAddressConflictSynthesisBugWorkaround));

      if (extractTestCode)
        pm.addPass(sv::createSVExtractTestCodePass(etcDisableInstanceExtraction,
                                                   etcDisableModuleInlining));

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
  if (loweringOptions.getNumOccurrences())
    loweringOptions.setAsAttribute(module.get());

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

    if (stripFirDebugInfo)
      exportPm.addPass(
          circt::createStripDebugInfoWithPredPass([](mlir::Location loc) {
            if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
              return fileLoc.getFilename().getValue().endswith(".fir");
            return false;
          }));

    if (stripDebugInfo)
      exportPm.addPass(circt::createStripDebugInfoWithPredPass(
          [](mlir::Location loc) { return true; }));

    // Emit a single file or multiple files depending on the output format.
    switch (outputFormat) {
    default:
      llvm_unreachable("can't reach this");
    case OutputVerilog:
      exportPm.addPass(createExportVerilogPass((*outputFile)->os()));
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

    if (failed(exportPm.run(module.get())))
      return failure();
  }

  if (outputFormat == OutputIRFir || outputFormat == OutputIRHW ||
      outputFormat == OutputIRSV || outputFormat == OutputIRVerilog) {
    auto outputTimer = ts.nest("Print .mlir output");
    printOp(*module, (*outputFile)->os());
  }

  // If requested, print the final MLIR into mlirOutFile.
  if (!mlirOutFile.empty()) {
    std::string mlirOutError;
    auto mlirFile = openOutputFile(mlirOutFile, &mlirOutError);
    if (!mlirFile) {
      llvm::errs() << mlirOutError;
      return failure();
    }

    printOp(*module, mlirFile->os());
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
static LogicalResult processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  sourceMgr.setIncludeDirs(includeDirs);
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
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
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
  if (stripMuxPragmas == addMuxPragmas) {
    llvm::errs()
        << "--strip-mux-pragmas and --add-mux-pragmas are conflicting.";
    return failure();
  }

  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Figure out the input format if unspecified.
  if (inputFormat == InputUnspecified) {
    if (StringRef(inputFilename).endswith(".fir"))
      inputFormat = InputFIRFile;
    else if (StringRef(inputFilename).endswith(".mlir") ||
             StringRef(inputFilename).endswith(".mlirbc") ||
             mlir::isBytecode(*input))
      inputFormat = InputMLIRFile;
    else {
      llvm::errs() << "unknown input format: "
                      "specify with -format=fir or -format=mlir\n";
      return failure();
    }
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  if (outputFormat != OutputSplitVerilog) {
    // Create an output file.
    outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
    if (!(*outputFile)) {
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
                      hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                      sv::SVDialect>();

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    (*outputFile)->keep();

  return success();
}

/// Main driver for firtool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeFirtool'.  This is set up
/// so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  {
    // MLIR transforms:
    // Don't use registerTransformsPasses, pulls in too much.
    registerCSEPass();
    registerCanonicalizerPass();
    registerStripDebugInfoPass();
    registerSymbolDCEPass();

    // Dialect passes:
    firrtl::registerPasses();
    sv::registerPasses();

    // Export passes:
    registerExportChiselInterfacePass();
    registerExportSplitChiselInterfacePass();
    registerExportSplitVerilogPass();
    registerExportVerilogPass();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
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
