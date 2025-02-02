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

#include "circt/Firtool/Firtool.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Target/DebugInfo.h"
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
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
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

using InfoLocHandling = firrtl::FIRParserOptions::InfoLocHandling;
static cl::opt<InfoLocHandling> infoLocHandling(
    cl::desc("Location tracking:"),
    cl::values(
        clEnumValN(InfoLocHandling::IgnoreInfo, "ignore-info-locators",
                   "Ignore the @info locations in the .fir file"),
        clEnumValN(InfoLocHandling::FusedInfo, "fuse-info-locators",
                   "@info locations are fused with .fir locations"),
        clEnumValN(
            InfoLocHandling::PreferInfo, "prefer-info-locators",
            "Use @info locations when present, fallback to .fir locations")),
    cl::init(InfoLocHandling::PreferInfo), cl::cat(mainCategory));

static cl::opt<bool>
    scalarizePublicModules("scalarize-public-modules",
                           cl::desc("Scalarize all public modules"),
                           cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    scalarizeIntModules("scalarize-internal-modules",
                        cl::desc("Scalarize the ports of any internal modules"),
                        cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    scalarizeExtModules("scalarize-ext-modules",
                        cl::desc("Scalarize the ports of any external modules"),
                        cl::init(true), cl::cat(mainCategory));

static cl::list<std::string>
    passPlugins("load-pass-plugin", cl::desc("Load passes from plugin library"),
                cl::CommaSeparated, cl::cat(mainCategory));

static cl::opt<std::string>
    highFIRRTLPassPlugin("high-firrtl-pass-plugin",
                         cl::desc("Insert passes after parsing FIRRTL. Specify "
                                  "passes with MLIR textual format."),
                         cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    lowFIRRTLPassPlugin("low-firrtl-pass-plugin",
                        cl::desc("Insert passes before lowering to HW. Specify "
                                 "passes with MLIR textual format."),
                        cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    hwPassPlugin("hw-pass-plugin",
                 cl::desc("Insert passes after lowering to HW. Specify "
                          "passes with MLIR textual format."),
                 cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    svPassPlugin("sv-pass-plugin",
                 cl::desc("Insert passes after lowering to SV. Specify "
                          "passes with MLIR textual format."),
                 cl::init(""), cl::cat(mainCategory));

enum OutputFormatKind {
  OutputParseOnly,
  OutputIRFir,
  OutputIRHW,
  OutputIRSV,
  OutputIRVerilog,
  OutputVerilog,
  OutputBTOR2,
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
        clEnumValN(OutputBTOR2, "btor2", "Emit BTOR2"),
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

static cl::opt<std::string>
    hwOutFile("output-hw-mlir",
              cl::desc("Optional file name to output the HW IR into, in "
                       "addition to the output requested by -o"),
              cl::init(""), cl::value_desc("filename"), cl::cat(mainCategory));

static cl::opt<std::string>
    mlirOutFile("output-final-mlir",
                cl::desc("Optional file name to output the final MLIR into, in "
                         "addition to the output requested by -o"),
                cl::init(""), cl::value_desc("filename"),
                cl::cat(mainCategory));

static cl::opt<bool> emitHGLDD("emit-hgldd", cl::desc("Emit HGLDD debug info"),
                               cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string>
    hglddSourcePrefix("hgldd-source-prefix",
                      cl::desc("Prefix for source file paths in HGLDD output"),
                      cl::init(""), cl::value_desc("path"),
                      cl::cat(mainCategory));

static cl::opt<std::string>
    hglddOutputPrefix("hgldd-output-prefix",
                      cl::desc("Prefix for output file paths in HGLDD output"),
                      cl::init(""), cl::value_desc("path"),
                      cl::cat(mainCategory));

static cl::opt<std::string> hglddOutputDirectory(
    "hgldd-output-dir", cl::desc("Directory into which to emit HGLDD files"),
    cl::init(""), cl::value_desc("path"), cl::cat(mainCategory));

static cl::opt<bool> hglddOnlyExistingFileLocs(
    "hgldd-only-existing-file-locs",
    cl::desc("Only consider locations in files that exist on disk"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static LoweringOptionsOption loweringOptions(mainCategory);

static cl::list<std::string>
    enableLayers("enable-layers", cl::desc("enable these layers permanently"),
                 cl::value_desc("layer-list"), cl::MiscFlags::CommaSeparated,
                 cl::cat(mainCategory));

static cl::list<std::string>
    disableLayers("disable-layers",
                  cl::desc("disable these layers permanently"),
                  cl::value_desc("layer-list"), cl::MiscFlags::CommaSeparated,
                  cl::cat(mainCategory));

enum class LayerSpecializationOpt { None, Enable, Disable };
static llvm::cl::opt<LayerSpecializationOpt> defaultLayerSpecialization{
    "default-layer-specialization",
    llvm::cl::desc("The default specialization for layers"),
    llvm::cl::values(
        clEnumValN(LayerSpecializationOpt::None, "none", "Layers are disabled"),
        clEnumValN(LayerSpecializationOpt::Disable, "disable",
                   "Layers are disabled"),
        clEnumValN(LayerSpecializationOpt::Enable, "enable",
                   "Layers are enabled")),
    cl::init(LayerSpecializationOpt::None), cl::cat(mainCategory)};

/// Specify the select option for specializing instance choice. Currently
/// firtool does not support partially specified instance choice.
static cl::list<std::string> selectInstanceChoice(
    "select-instance-choice",
    cl::desc("Options to specialize instance choice, in option=case format"),
    cl::MiscFlags::CommaSeparated, cl::cat(mainCategory));

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
static LogicalResult printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    return writeBytecodeToFile(op, os,
                               mlir::BytecodeWriterConfig(getCirctVersion()));
  op->print(os);
  return success();
}

static debug::EmitHGLDDOptions getHGLDDOptions() {
  debug::EmitHGLDDOptions opts;
  opts.sourceFilePrefix = hglddSourcePrefix;
  opts.outputFilePrefix = hglddOutputPrefix;
  opts.outputDirectory = hglddOutputDirectory;
  opts.onlyExistingFileLocs = hglddOnlyExistingFileLocs;
  return opts;
}

/// Wrapper pass to call the `emitHGLDD` translation.
struct EmitHGLDDPass
    : public PassWrapper<EmitHGLDDPass, OperationPass<mlir::ModuleOp>> {
  llvm::raw_ostream &os;
  EmitHGLDDPass(llvm::raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    markAllAnalysesPreserved();
    if (failed(debug::emitHGLDD(getOperation(), os, getHGLDDOptions())))
      return signalPassFailure();
  }
};

/// Wrapper pass to call the `emitSplitHGLDD` translation.
struct EmitSplitHGLDDPass
    : public PassWrapper<EmitSplitHGLDDPass, OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    markAllAnalysesPreserved();
    if (failed(debug::emitSplitHGLDD(getOperation(), getHGLDDOptions())))
      return signalPassFailure();
  }
};

/// Wrapper pass to dump IR.
struct DumpIRPass
    : public PassWrapper<DumpIRPass, OperationPass<mlir::ModuleOp>> {
  DumpIRPass(const std::string &outputFile)
      : PassWrapper<DumpIRPass, OperationPass<mlir::ModuleOp>>() {
    this->outputFile.setValue(outputFile);
  }

  DumpIRPass(const DumpIRPass &other) : PassWrapper(other) {
    outputFile.setValue(other.outputFile.getValue());
  }

  void runOnOperation() override {
    assert(!outputFile.empty());

    std::string error;
    auto mlirFile = openOutputFile(outputFile.getValue(), &error);
    if (!mlirFile) {
      errs() << error;
      return signalPassFailure();
    }

    if (failed(printOp(getOperation(), mlirFile->os())))
      return signalPassFailure();
    mlirFile->keep();
    markAllAnalysesPreserved();
  }

  Pass::Option<std::string> outputFile{*this, "output-file",
                                       cl::desc("filename"), cl::init("-")};
};

/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, firtool::FirtoolOptions &firtoolOptions,
    TimingScope &ts, llvm::SourceMgr &sourceMgr,
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
    options.infoLocatorHandling = infoLocHandling;
    options.numAnnotationFiles = numAnnotationFiles;
    options.scalarizePublicModules = scalarizePublicModules;
    options.scalarizeInternalModules = scalarizeIntModules;
    options.scalarizeExtModules = scalarizeExtModules;
    options.enableLayers = enableLayers;
    options.disableLayers = disableLayers;
    options.selectInstanceChoice = selectInstanceChoice;

    switch (defaultLayerSpecialization) {
    case LayerSpecializationOpt::None:
      options.defaultLayerSpecialization = std::nullopt;
      break;
    case LayerSpecializationOpt::Enable:
      options.defaultLayerSpecialization = firrtl::LayerSpecialization::Enable;
      break;
    case LayerSpecializationOpt::Disable:
      options.defaultLayerSpecialization = firrtl::LayerSpecialization::Disable;
      break;
    }

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
    pm.addInstrumentation(
        std::make_unique<
            VerbosePassInstrumentation<firrtl::CircuitOp, mlir::ModuleOp>>(
            "firtool"));
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  if (failed(firtool::populatePreprocessTransforms(pm, firtoolOptions)))
    return failure();

  // If the user asked for --parse-only, stop after running LowerAnnotations.
  if (outputFormat == OutputParseOnly) {
    if (failed(pm.run(module.get())))
      return failure();
    auto outputTimer = ts.nest("Print .mlir output");
    return printOp(*module, (*outputFile)->os());
  }

  if (!highFIRRTLPassPlugin.empty())
    if (failed(parsePassPipeline(StringRef(highFIRRTLPassPlugin), pm)))
      return failure();

  if (failed(firtool::populateCHIRRTLToLowFIRRTL(pm, firtoolOptions,
                                                 inputFilename)))
    return failure();

  if (!lowFIRRTLPassPlugin.empty())
    if (failed(parsePassPipeline(StringRef(lowFIRRTLPassPlugin), pm)))
      return failure();

  // Lower if we are going to verilog or if lowering was specifically
  // requested.
  if (outputFormat != OutputIRFir) {
    if (failed(firtool::populateLowFIRRTLToHW(pm, firtoolOptions)))
      return failure();
    if (!hwPassPlugin.empty())
      if (failed(parsePassPipeline(StringRef(hwPassPlugin), pm)))
        return failure();
    // Add passes specific to btor2 emission
    if (outputFormat == OutputBTOR2)
      if (failed(firtool::populateHWToBTOR2(pm, firtoolOptions,
                                            (*outputFile)->os())))
        return failure();

    // If requested, emit the HW IR to hwOutFile.
    if (!hwOutFile.empty())
      pm.addPass(std::make_unique<DumpIRPass>(hwOutFile.getValue()));

    if (outputFormat != OutputIRHW)
      if (failed(firtool::populateHWToSV(pm, firtoolOptions)))
        return failure();
    if (!svPassPlugin.empty())
      if (failed(parsePassPipeline(StringRef(svPassPlugin), pm)))
        return failure();
  }

  // If the user requested HGLDD debug info emission, enable Verilog location
  // tracking.
  if (emitHGLDD)
    loweringOptions.emitVerilogLocations = true;

  // Load the emitter options from the command line. Command line options if
  // specified will override any module options.
  if (loweringOptions.toString() != LoweringOptions().toString())
    loweringOptions.setAsAttribute(module.get());

  // Add passes specific to Verilog emission if we're going there.
  if (outputFormat == OutputVerilog || outputFormat == OutputSplitVerilog ||
      outputFormat == OutputIRVerilog) {

    // Emit a single file or multiple files depending on the output format.
    switch (outputFormat) {
    default:
      llvm_unreachable("can't reach this");
    case OutputVerilog:
      if (failed(firtool::populateExportVerilog(pm, firtoolOptions,
                                                (*outputFile)->os())))
        return failure();
      if (emitHGLDD)
        pm.addPass(std::make_unique<EmitHGLDDPass>((*outputFile)->os()));
      break;
    case OutputSplitVerilog:
      if (failed(firtool::populateExportSplitVerilog(
              pm, firtoolOptions, firtoolOptions.getOutputFilename())))
        return failure();
      if (emitHGLDD)
        pm.addPass(std::make_unique<EmitSplitHGLDDPass>());
      break;
    case OutputIRVerilog:
      // Run the ExportVerilog pass to get its lowering, but discard the output.
      if (failed(firtool::populateExportVerilog(pm, firtoolOptions,
                                                llvm::nulls())))
        return failure();
      break;
    }

    // If requested, print the final MLIR into mlirOutFile.
    if (!mlirOutFile.empty()) {
      // Run final IR mutations to clean it up after ExportVerilog and before
      // emitting the final MLIR.
      if (failed(firtool::populateFinalizeIR(pm, firtoolOptions)))
        return failure();

      pm.addPass(std::make_unique<DumpIRPass>(mlirOutFile.getValue()));
    }
  }

  if (failed(pm.run(module.get())))
    return failure();

  if (outputFormat == OutputIRFir || outputFormat == OutputIRHW ||
      outputFormat == OutputIRSV || outputFormat == OutputIRVerilog) {
    auto outputTimer = ts.nest("Print .mlir output");
    if (failed(printOp(*module, (*outputFile)->os())))
      return failure();
  }

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

class FileLineColLocsAsNotesDiagnosticHandler : public ScopedDiagnosticHandler {
public:
  FileLineColLocsAsNotesDiagnosticHandler(MLIRContext *ctxt)
      : ScopedDiagnosticHandler(ctxt) {
    setHandler([](Diagnostic &d) {
      SmallPtrSet<Location, 8> locs;
      // Recursively scan for FileLineColLoc locations.
      d.getLocation()->walk([&](Location loc) {
        if (isa<FileLineColLoc>(loc))
          locs.insert(loc);
        return WalkResult::advance();
      });

      // Drop top-level location the diagnostic is reported on.
      locs.erase(d.getLocation());
      // As well as the location the SourceMgrDiagnosticHandler will use.
      if (auto reportLoc = d.getLocation()->findInstanceOf<FileLineColLoc>())
        locs.erase(reportLoc);

      // Attach additional locations as notes on the diagnostic.
      for (auto l : locs)
        d.attachNote(l) << "additional location here";
      return failure();
    });
  }
};

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult processInputSplit(
    MLIRContext &context, firtool::FirtoolOptions &firtoolOptions,
    TimingScope &ts, std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  sourceMgr.setIncludeDirs(includeDirs);
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
                                                &context /*, shouldShow */);
    FileLineColLocsAsNotesDiagnosticHandler addLocs(&context);
    return processBuffer(context, firtoolOptions, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, firtoolOptions, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, firtool::FirtoolOptions &firtoolOptions,
             TimingScope &ts, std::unique_ptr<llvm::MemoryBuffer> input,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, firtoolOptions, ts, std::move(input),
                             outputFile);

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
        return processInputSplit(context, firtoolOptions, ts, std::move(buffer),
                                 outputFile);
      },
      llvm::outs());
}

/// This implements the top-level logic for the firtool command, invoked once
/// command line options are parsed and LLVM/MLIR are all set up and ready to
/// go.
static LogicalResult executeFirtool(MLIRContext &context,
                                    firtool::FirtoolOptions &firtoolOptions) {
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
    if (StringRef(inputFilename).ends_with(".fir"))
      inputFormat = InputFIRFile;
    else if (StringRef(inputFilename).ends_with(".mlir") ||
             StringRef(inputFilename).ends_with(".mlirbc") ||
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
    outputFile.emplace(
        openOutputFile(firtoolOptions.getOutputFilename(), &errorMessage));
    if (!(*outputFile)) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
  } else {
    // Create an output directory.
    if (firtoolOptions.isDefaultOutputFilename()) {
      llvm::errs() << "missing output directory: specify with -o=<dir>\n";
      return failure();
    }
    auto error =
        llvm::sys::fs::create_directories(firtoolOptions.getOutputFilename());
    if (error) {
      llvm::errs() << "cannot create output directory '"
                   << firtoolOptions.getOutputFilename()
                   << "': " << error.message() << "\n";
      return failure();
    }
  }

  // Register our dialects.
  context.loadDialect<chirrtl::CHIRRTLDialect, emit::EmitDialect,
                      firrtl::FIRRTLDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, om::OMDialect, sv::SVDialect,
                      verif::VerifDialect, ltl::LTLDialect, debug::DebugDialect,
                      sim::SimDialect>();

  // Process the input.
  if (failed(processInput(context, firtoolOptions, ts, std::move(input),
                          outputFile)))
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

  /// Set the callback to load a pass plugin.
  passPlugins.setCallback([&](const std::string &pluginPath) {
    llvm::errs() << "[firtool] load plugin " << pluginPath << '\n';
    auto plugin = PassPlugin::load(pluginPath);
    if (!plugin) {
      errs() << plugin.takeError() << '\n';
      errs() << "Failed to load passes from '" << pluginPath
             << "'. Request ignored.\n";
      return;
    }
    plugin.get().registerPassRegistryCallbacks();
  });

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
    om::registerPasses();
    sv::registerPasses();
    hw::registerFlattenModulesPass();
    verif::registerVerifyClockedAssertLikePass();

    // Export passes:
    registerExportChiselInterfacePass();
    registerExportSplitChiselInterfacePass();
    registerExportSplitVerilogPass();
    registerExportVerilogPass();

    // Conversion passes:
    registerPrepareForEmissionPass();
    registerHWLowerInstanceChoicesPass();
    registerLowerFIRRTLToHWPass();
    registerLegalizeAnonEnumsPass();
    registerLowerSeqToSVPass();
    registerLowerSimToSVPass();
    registerLowerVerifToSVPass();
    registerLowerLTLToCorePass();
    registerConvertHWToBTOR2Pass();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  firtool::registerFirtoolCLOptions();
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR-based FIRRTL compiler\n");

  MLIRContext context;
  // Get firtool options from cmdline
  firtool::FirtoolOptions firtoolOptions;

  // Do the guts of the firtool process.
  auto result = executeFirtool(context, firtoolOptions);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
