//===- circt-verilog.cpp - Getting Verilog into CIRCT ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility to parse Verilog and SystemVerilog input
// files. This builds on CIRCT's ImportVerilog library, which ultimately relies
// on slang to do the heavy lifting.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

namespace {
enum class Format {
  SV,
  MLIR,
};

enum class LoweringMode {
  OnlyPreprocess,
  OnlyLint,
  OnlyParse,
  OutputIRMoore,
  OutputIRLLHD,
  OutputIRHW,
  Full
};

struct CLOptions {
  cl::OptionCategory cat{"Verilog Frontend Options"};

  cl::opt<Format> format{
      "format", cl::desc("Input file format (auto-detected by default)"),
      cl::values(
          clEnumValN(Format::SV, "sv", "Parse as SystemVerilog files"),
          clEnumValN(Format::MLIR, "mlir", "Parse as MLIR or MLIRBC file")),
      cl::cat(cat)};

  cl::list<std::string> inputFilenames{cl::Positional,
                                       cl::desc("<input files>"), cl::cat(cat)};

  cl::opt<std::string> outputFilename{
      "o", cl::desc("Output filename (`-` for stdout)"),
      cl::value_desc("filename"), cl::init("-"), cl::cat(cat)};

  cl::opt<bool> verifyDiagnostics{
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match expected-* lines on the "
               "corresponding line"),
      cl::init(false), cl::Hidden, cl::cat(cat)};

  cl::opt<LoweringMode> loweringMode{
      cl::desc("Specify how to process the input:"),
      cl::values(
          clEnumValN(
              LoweringMode::OnlyPreprocess, "E",
              "Only run the preprocessor (and print preprocessed files)"),
          clEnumValN(LoweringMode::OnlyLint, "lint-only",
                     "Only lint the input, without elaboration and mapping to "
                     "CIRCT IR"),
          clEnumValN(LoweringMode::OnlyParse, "parse-only",
                     "Only parse and elaborate the input, without mapping to "
                     "CIRCT IR"),
          clEnumValN(LoweringMode::OutputIRMoore, "ir-moore",
                     "Run the entire pass manager to just before MooreToCore "
                     "conversion, and emit the resulting Moore dialect IR"),
          clEnumValN(
              LoweringMode::OutputIRLLHD, "ir-llhd",
              "Run the entire pass manager to just before the LLHD pipeline "
              ", and emit the resulting LLHD+Core dialect IR"),
          clEnumValN(LoweringMode::OutputIRHW, "ir-hw",
                     "Run the MooreToCore conversion and emit the resulting "
                     "core dialect IR")),
      cl::init(LoweringMode::Full), cl::cat(cat)};

  cl::opt<bool> debugInfo{"g", cl::desc("Generate debug information"),
                          cl::cat(cat)};

  cl::opt<bool> lowerAlwaysAtStarAsComb{
      "always-at-star-as-comb",
      cl::desc("Interpret `always @(*)` as `always_comb`"), cl::init(true),
      cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Include paths
  //===--------------------------------------------------------------------===//

  cl::list<std::string> includeDirs{
      "I", cl::desc("Additional include search paths"), cl::value_desc("dir"),
      cl::Prefix, cl::cat(cat)};
  cl::alias includeDirsLong{"include-dir", cl::desc("Alias for -I"),
                            cl::aliasopt(includeDirs), cl::NotHidden,
                            cl::cat(cat)};

  cl::list<std::string> includeSystemDirs{
      "isystem", cl::desc("Additional system include search paths"),
      cl::value_desc("dir"), cl::cat(cat)};

  cl::list<std::string> libDirs{
      "y",
      cl::desc(
          "Library search paths, which will be searched for missing modules"),
      cl::value_desc("dir"), cl::Prefix, cl::cat(cat)};
  cl::alias libDirsLong{"libdir", cl::desc("Alias for -y"),
                        cl::aliasopt(libDirs), cl::NotHidden, cl::cat(cat)};

  cl::list<std::string> libExts{
      "Y", cl::desc("Additional library file extensions to search"),
      cl::value_desc("ext"), cl::Prefix, cl::cat(cat)};
  cl::alias libExtsLong{"libext", cl::desc("Alias for -Y"),
                        cl::aliasopt(libExts), cl::NotHidden, cl::cat(cat)};

  cl::list<std::string> excludeExts{
      "exclude-ext",
      cl::desc("Exclude provided source files with these extensions"),
      cl::value_desc("ext"), cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Preprocessor
  //===--------------------------------------------------------------------===//

  cl::list<std::string> defines{
      "D",
      cl::desc("Define <macro> to <value> (or 1 if <value> omitted) in all "
               "source files"),
      cl::value_desc("<macro>=<value>"), cl::Prefix, cl::cat(cat)};
  cl::alias definesLong{"define-macro", cl::desc("Alias for -D"),
                        cl::aliasopt(defines), cl::NotHidden, cl::cat(cat)};

  cl::list<std::string> undefines{
      "U", cl::desc("Undefine macro name at the start of all source files"),
      cl::value_desc("macro"), cl::Prefix, cl::cat(cat)};
  cl::alias undefinesLong{"undefine-macro", cl::desc("Alias for -U"),
                          cl::aliasopt(undefines), cl::NotHidden, cl::cat(cat)};

  cl::opt<uint32_t> maxIncludeDepth{
      "max-include-depth",
      cl::desc("Maximum depth of nested include files allowed"),
      cl::value_desc("depth"), cl::cat(cat)};

  cl::opt<bool> librariesInheritMacros{
      "libraries-inherit-macros",
      cl::desc("If true, library files will inherit macro definitions from the "
               "primary source files. --single-unit must also be passed when "
               "this option is used."),
      cl::init(false), cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Compilation
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> timeScale{
      "timescale",
      cl::desc("Default time scale to use for design elements that don't "
               "specify one explicitly"),
      cl::value_desc("<base>/<precision>"), cl::cat(cat)};

  cl::opt<bool> allowUseBeforeDeclare{
      "allow-use-before-declare",
      cl::desc(
          "Don't issue an error for use of names before their declarations."),
      cl::init(false), cl::cat(cat)};

  cl::opt<bool> ignoreUnknownModules{
      "ignore-unknown-modules",
      cl::desc("Don't issue an error for instantiations of unknown modules, "
               "interface, and programs."),
      cl::init(false), cl::cat(cat)};

  cl::list<std::string> topModules{
      "top",
      cl::desc("One or more top-level modules to instantiate (instead of "
               "figuring it out automatically)"),
      cl::value_desc("name"), cl::cat(cat)};

  cl::list<std::string> paramOverrides{
      "G",
      cl::desc("One or more parameter overrides to apply when instantiating "
               "top-level modules"),
      cl::value_desc("<name>=<value>"), cl::Prefix, cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Diagnostics control
  //===--------------------------------------------------------------------===//

  cl::list<std::string> warningOptions{
      "W", cl::desc("Control the specified warning"), cl::value_desc("warning"),
      cl::Prefix, cl::cat(cat)};

  cl::opt<uint32_t> errorLimit{
      "error-limit",
      cl::desc("Limit on the number of errors that will be printed. Setting "
               "this to zero will disable the limit."),
      cl::value_desc("limit"), cl::cat(cat)};

  cl::list<std::string> suppressWarningsPaths{
      "suppress-warnings",
      cl::desc("One or more paths in which to suppress warnings"),
      cl::value_desc("filename"), cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // File lists
  //===--------------------------------------------------------------------===//

  cl::opt<bool> singleUnit{
      "single-unit",
      cl::desc("Treat all input files as a single compilation unit"),
      cl::init(false), cl::cat(cat)};

  cl::list<std::string> libraryFiles{
      "l",
      cl::desc(
          "One or more library files, which are separate compilation units "
          "where modules are not automatically instantiated."),
      cl::value_desc("filename"), cl::Prefix, cl::cat(cat)};
};
} // namespace

static CLOptions opts;

//===----------------------------------------------------------------------===//
// Pass Pipeline
//===----------------------------------------------------------------------===//

/// Optimize and simplify the Moore dialect IR.
static void populateMooreTransforms(PassManager &pm) {
  {
    // Perform an initial cleanup and preprocessing across all
    // modules/functions.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }

  {
    // Perform module-specific transformations.
    auto &modulePM = pm.nest<moore::SVModuleOp>();
    modulePM.addPass(moore::createLowerConcatRefPass());
    // TODO: Enable the following once it not longer interferes with @(...)
    // event control checks. The introduced dummy variables make the event
    // control observe a static local variable that never changes, instead of
    // observing a module-wide signal.
    // modulePM.addPass(moore::createSimplifyProceduresPass());
  }

  {
    // Perform a final cleanup across all modules/functions.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createSROA());
    anyPM.addPass(mlir::createMem2Reg());
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }
}

/// Convert Moore dialect IR into core dialect IR
static void populateMooreToCoreLowering(PassManager &pm) {
  // Perform the conversion.
  pm.addPass(createConvertMooreToCorePass());

  {
    // Conversion to the core dialects likely uncovers new canonicalization
    // opportunities.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }
}

/// Convert LLHD dialect IR into core dialect IR
static void populateLLHDLowering(PassManager &pm) {
  pm.addPass(createInlinerPass());
  {
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createSROA());
  }
  pm.addNestedPass<hw::HWModuleOp>(llhd::createEarlyCodeMotion());
  pm.addNestedPass<hw::HWModuleOp>(llhd::createTemporalCodeMotion());
  {
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }
  pm.addNestedPass<hw::HWModuleOp>(llhd::createDesequentialization());
  pm.addPass(llhd::createProcessLowering());
  pm.addNestedPass<hw::HWModuleOp>(llhd::createSig2Reg());
  {
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }
}

/// Populate the given pass manager with transformations as configured by the
/// command line options.
static void populatePasses(PassManager &pm) {
  populateMooreTransforms(pm);
  if (opts.loweringMode == LoweringMode::OutputIRMoore)
    return;
  populateMooreToCoreLowering(pm);
  if (opts.loweringMode == LoweringMode::OutputIRLLHD)
    return;
  populateLLHDLowering(pm);
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

static LogicalResult executeWithSources(MLIRContext *context,
                                        llvm::SourceMgr &sourceMgr) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Map the command line options to `ImportVerilog`'s conversion options.
  ImportVerilogOptions options;
  if (opts.loweringMode == LoweringMode::OnlyLint)
    options.mode = ImportVerilogOptions::Mode::OnlyLint;
  else if (opts.loweringMode == LoweringMode::OnlyParse)
    options.mode = ImportVerilogOptions::Mode::OnlyParse;
  options.debugInfo = opts.debugInfo;
  options.lowerAlwaysAtStarAsComb = opts.lowerAlwaysAtStarAsComb;

  options.includeDirs = opts.includeDirs;
  options.includeSystemDirs = opts.includeSystemDirs;
  options.libDirs = opts.libDirs;
  options.libExts = opts.libExts;
  options.excludeExts = opts.excludeExts;

  options.defines = opts.defines;
  options.undefines = opts.undefines;
  if (opts.maxIncludeDepth.getNumOccurrences() > 0)
    options.maxIncludeDepth = opts.maxIncludeDepth;
  options.librariesInheritMacros = opts.librariesInheritMacros;

  if (opts.timeScale.getNumOccurrences() > 0)
    options.timeScale = opts.timeScale;
  options.allowUseBeforeDeclare = opts.allowUseBeforeDeclare;
  options.ignoreUnknownModules = opts.ignoreUnknownModules;
  if (opts.loweringMode != LoweringMode::OnlyLint)
    options.topModules = opts.topModules;
  options.paramOverrides = opts.paramOverrides;

  options.warningOptions = opts.warningOptions;
  if (opts.errorLimit.getNumOccurrences() > 0)
    options.errorLimit = opts.errorLimit;
  options.suppressWarningsPaths = opts.suppressWarningsPaths;

  options.singleUnit = opts.singleUnit;
  options.libraryFiles = opts.libraryFiles;

  // Open the output file.
  std::string errorMessage;
  auto outputFile = openOutputFile(opts.outputFilename, &errorMessage);
  if (!outputFile) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

  // Parse the input as SystemVerilog or MLIR file.
  OwningOpRef<ModuleOp> module;
  switch (opts.format) {
  case Format::SV: {
    // If the user requested for the files to be only preprocessed, do so and
    // print the results to the configured output file.
    if (opts.loweringMode == LoweringMode::OnlyPreprocess) {
      auto result =
          preprocessVerilog(sourceMgr, context, ts, outputFile->os(), &options);
      if (succeeded(result))
        outputFile->keep();
      return result;
    }

    // Parse the Verilog input into an MLIR module.
    module = ModuleOp::create(UnknownLoc::get(context));
    if (failed(importVerilog(sourceMgr, context, ts, module.get(), &options)))
      return failure();

    // If the user requested for the files to be only linted, the module remains
    // empty and there is nothing left to do.
    if (opts.loweringMode == LoweringMode::OnlyLint)
      return success();
  } break;

  case Format::MLIR: {
    auto parserTimer = ts.nest("MLIR Parser");
    module = parseSourceFile<ModuleOp>(sourceMgr, context);
  } break;
  }
  if (!module)
    return failure();

  // If the user requested anything besides simply parsing the input, run the
  // appropriate transformation passes according to the command line options.
  if (opts.loweringMode != LoweringMode::OnlyParse) {
    PassManager pm(context);
    pm.enableVerifier(true);
    pm.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    populatePasses(pm);
    if (failed(pm.run(module.get())))
      return failure();
  }

  // Print the final MLIR.
  auto outputTimer = ts.nest("MLIR Printer");
  module->print(outputFile->os());
  outputFile->keep();
  return success();
}

static LogicalResult execute(MLIRContext *context) {
  // Default to reading from stdin if no files were provided.
  if (opts.inputFilenames.empty())
    opts.inputFilenames.push_back("-");

  // Auto-detect the input format if it was not explicitly specified.
  if (opts.format.getNumOccurrences() == 0) {
    std::optional<Format> detectedFormat = std::nullopt;
    for (const auto &inputFilename : opts.inputFilenames) {
      std::optional<Format> format = std::nullopt;
      auto name = StringRef(inputFilename);
      if (name.ends_with(".v") || name.ends_with(".sv") ||
          name.ends_with(".vh") || name.ends_with(".svh"))
        format = Format::SV;
      else if (name.ends_with(".mlir") || name.ends_with(".mlirbc"))
        format = Format::MLIR;
      if (!format)
        continue;
      if (detectedFormat && format != detectedFormat) {
        detectedFormat = std::nullopt;
        break;
      }
      detectedFormat = format;
    }
    if (!detectedFormat) {
      WithColor::error() << "cannot auto-detect input format; use --format\n";
      return failure();
    }
    opts.format = *detectedFormat;
  }

  // Open the input files.
  llvm::SourceMgr sourceMgr;
  for (const auto &inputFilename : opts.inputFilenames) {
    std::string errorMessage;
    auto buffer = openInputFile(inputFilename, &errorMessage);
    if (!buffer) {
      WithColor::error() << errorMessage << "\n";
      return failure();
    }
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  }

  // Call `executeWithSources` with either the regular diagnostic handler, or,
  // if `--verify-diagnostics` is set, with the verifying handler.
  if (opts.verifyDiagnostics) {
    SourceMgrDiagnosticVerifierHandler handler(sourceMgr, context);
    context->printOpOnDiagnostic(false);
    (void)executeWithSources(context, sourceMgr);
    return handler.verify();
  }
  SourceMgrDiagnosticHandler handler(sourceMgr, context);
  return executeWithSources(context, sourceMgr);
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Print the CIRCT and Slang versions when requested.
  cl::AddExtraVersionPrinter([](raw_ostream &os) {
    os << getCirctVersion() << '\n';
    os << getSlangVersion() << '\n';
  });

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv,
                              "Verilog and SystemVerilog frontend\n");

  // Register the dialects.
  // clang-format off
  DialectRegistry registry;
  registry.insert<
    cf::ControlFlowDialect,
    comb::CombDialect,
    debug::DebugDialect,
    func::FuncDialect,
    hw::HWDialect,
    llhd::LLHDDialect,
    moore::MooreDialect,
    scf::SCFDialect,
    seq::SeqDialect,
    sim::SimDialect,
    verif::VerifDialect
  >();
  // clang-format on

  // Perform the actual work and use "exit" to avoid slow context teardown.
  mlir::func::registerInlinerExtension(registry);
  MLIRContext context(registry);
  exit(failed(execute(&context)));
}
