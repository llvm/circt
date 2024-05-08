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
#include "circt/Support/Version.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

namespace {
enum class LoweringMode { OnlyPreprocess, OnlyLint, OnlyParse, Full };

struct CLOptions {
  cl::OptionCategory cat{"Verilog Frontend Options"};

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
                     "CIRCT IR")),
      cl::init(LoweringMode::Full), cl::cat(cat)};

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
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

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
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(context)));
  if (failed(importVerilog(sourceMgr, context, ts, module.get(), &options)))
    return failure();

  // Print the final MLIR.
  module->print(outputFile->os());
  outputFile->keep();
  return success();
}

static LogicalResult execute(MLIRContext *context) {
  // Open the input files.
  llvm::SourceMgr sourceMgr;
  for (const auto &inputFilename : opts.inputFilenames) {
    std::string errorMessage;
    auto buffer = openInputFile(inputFilename, &errorMessage);
    if (!buffer) {
      llvm::errs() << errorMessage << "\n";
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

  // Perform the actual work and use "exit" to avoid slow context teardown.
  MLIRContext context;
  exit(failed(execute(&context)));
}
