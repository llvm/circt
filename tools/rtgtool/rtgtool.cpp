//===- rtgtool.cpp - The Random Test Generation Driver Tool ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'rtgtool' tool
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Target/EmitRTGAssembly.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

// TODO: take python file as input such that you can pass it the context created
// in this tool and the python file adds the instructions using that context
// (maybe to the passed module op?)

//===----------------------------------------------------------------------===//
// Command-line option helpers
//===----------------------------------------------------------------------===//

inline llvm::Expected<std::optional<unsigned>> parseSeedOption(StringRef arg) {
  if (arg == "none")
    return std::nullopt;

  unsigned val;
  if (arg.getAsInteger(10, val))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Not an integer: %s", arg.data());

  return val;
}

// A simple CL parser for '--seed='
class SeedParser : public cl::parser<std::optional<unsigned>> {
public:
  SeedParser(cl::Option &o) : cl::parser<std::optional<unsigned>>(o) {}

  bool parse(cl::Option &o, StringRef argName, StringRef arg,
             std::optional<unsigned> &v) {
    auto resultOrErr = parseSeedOption(arg);

    if (!resultOrErr)
      return o.error("Invalid argument '" + arg +
                     "', only integer or 'none' is supported.");

    v = *resultOrErr;
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("rtgtool Options");

static cl::opt<std::string>
    moduleName("module",
               cl::desc("Specify a named module to verify properties over."),
               cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<std::optional<unsigned>, false, SeedParser>
    seed("seed", cl::desc("Seed for all RNGs."), cl::init(std::nullopt),
         cl::cat(mainCategory));

static cl::opt<std::string> unsupportedInstructionsFile(
    "unsupported-instructions-file",
    cl::desc("File with a comma-separated list of instructions "
             "not supported by the assembler."),
    cl::init(""), cl::cat(mainCategory));

static cl::list<std::string> unsupportedInstructions(
    "unsupported-instructions",
    cl::desc(
        "Comma-separated list of instructions not supported by the assembler."),
    cl::MiscFlags::CommaSeparated, cl::cat(mainCategory));

static cl::list<std::string>
    dialectPlugins("load-dialect-plugin",
                   cl::desc("Load dialects from plugin library"));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

enum OutputFormat { OutputMLIR, OutputRenderedMLIR, OutputASM, OutputELF };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir",
                          "Emit RTG+ISA MLIR dialects"),
               clEnumValN(OutputRenderedMLIR, "emit-rendered-mlir",
                          "Emit ISA MLIR dialects"),
               clEnumValN(OutputASM, "emit-asm", "Emit Assembly file"),
               clEnumValN(OutputELF, "emit-elf", "Emit ELF file")),
    cl::init(OutputASM), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

static void setDialectPluginsCallback(DialectRegistry &registry) {
  dialectPlugins.setCallback([&](const std::string &pluginPath) {
    auto plugin = DialectPlugin::load(pluginPath);
    if (!plugin) {
      llvm::errs() << "Failed to load dialect plugin from '" << pluginPath
                   << "'. Request ignored.\n";
      return;
    };
    plugin.get().registerDialectRegistryCallbacks(registry);
  });
}

/// This function initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeRTGTool(MLIRContext &context) {
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
            "circt-bmc"));

  pm.addPass(createSimpleCanonicalizerPass());
  if (outputFormat != OutputMLIR) {
    rtg::ElaborationOptions options;
    options.seed = seed;
    pm.addPass(rtg::createElaborationPass(options));
    pm.addPass(createSimpleCanonicalizerPass());
  }

  if (failed(pm.run(module.get())))
    return failure();

  if (outputFormat == OutputMLIR || outputFormat == OutputRenderedMLIR) {
    auto timer = ts.nest("Print MLIR output");
    OpPrintingFlags printingFlags;
    module->print(outputFile.value()->os(), printingFlags);
    outputFile.value()->keep();
    return success();
  }

  if (outputFormat == OutputASM) {
    auto timer = ts.nest("Print Assembly output");
    EmitRTGAssembly::EmitRTGAssemblyOptions options;
    options.unsupportedInstructions = {unsupportedInstructions.begin(),
                                       unsupportedInstructions.end()};
    EmitRTGAssembly::parseUnsupportedInstructionsFile(
        unsupportedInstructionsFile, options.unsupportedInstructions);
    if (failed(EmitRTGAssembly::emitRTGAssembly(
            module.get(), outputFile.value()->os(), options)))
      return failure();

    return success();
  }

  if (outputFormat == OutputELF) {
    auto timer = ts.nest("Translate to and print LLVM output");
    llvm::errs() << "ELF output not supported yet.\n";
    return failure();
  }

  return failure();
}

/// The entry point for the `rtgtool` tool: configures and parses the
/// command-line options, registers all dialects within a MLIR context, and
/// calls the `executeBMC` function to do the actual work.
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
  cl::ParseCommandLineOptions(
      argc, argv,
      "rtgtool - Random Test Generator\n\n"
      "\tThis tool takes test snippets with randomization constructs and "
      "constraints as input and produces fully elaborated tests.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  DialectRegistry registry;
  setDialectPluginsCallback(registry);
  // Register the supported CIRCT dialects and create a context to work with.
  registry.insert<circt::rtg::RTGDialect, circt::rtgtest::RTGTestDialect,
                  mlir::arith::ArithDialect, mlir::BuiltinDialect>();
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeRTGTool(context)));
}
