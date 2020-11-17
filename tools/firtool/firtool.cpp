//===- firtool.cpp - The firtool utility for working with .fir files ------===//
//
// This file implements 'firtool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Dialect.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/SV/Dialect.h"
#include "circt/EmitVerilog.h"
#include "circt/FIRParser.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
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

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> disableOptimization("disable-opt",
                                         cl::desc("disable optimizations"));

static cl::opt<bool> lowerToRTL("lower-to-rtl",
                                cl::desc("run the lower-to-rtl pass"));

static cl::opt<bool>
    ignoreFIRLocations("ignore-fir-locators",
                       cl::desc("ignore the @info locations in the .fir file"),
                       cl::init(false));

enum OutputFormatKind { OutputMLIR, OutputVerilog, OutputDisabled };

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(clEnumValN(OutputMLIR, "mlir", "Emit MLIR dialect"),
               clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
               clEnumValN(OutputDisabled, "disable-output",
                          "Do not output anything")),
    cl::init(OutputMLIR));

/// Process a single buffer of the input.
static LogicalResult
processBuffer(std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              raw_ostream &os) {
  MLIRContext context;

  // Register our dialects.
  context.loadDialect<firrtl::FIRRTLDialect, rtl::RTLDialect, sv::SVDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Nothing in the parser is threaded.  Disable synchronization overhead.
  context.disableMultithreading();

  OwningModuleRef module;
  if (inputFormat == InputFIRFile) {
    FIRParserOptions options;
    options.ignoreInfoLocators = ignoreFIRLocations;
    module = parseFIRFile(sourceMgr, &context, options);
  } else {
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // Allow optimizations to run multithreaded.
  context.disableMultithreading(false);

  // If enabled, run the optimizer.
  if (!disableOptimization) {
    // Apply any pass manager command line options.
    PassManager pm(&context);
    pm.enableVerifier(true);
    applyPassManagerCLOptions(pm);

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Run the lower-to-rtl pass if requested.
    if (lowerToRTL) {
      pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
          firrtl::createLowerFIRRTLTypesPass());
      pm.addPass(firrtl::createLowerFIRRTLToRTLModulePass());
      pm.nest<rtl::RTLModuleOp>().addPass(firrtl::createLowerFIRRTLToRTLPass());
    }

    if (failed(pm.run(module.get())))
      return failure();
  }

  // Finally, emit the output.
  switch (outputFormat) {
  case OutputMLIR:
    module->print(os);
    return success();
  case OutputDisabled:
    return success();
  case OutputVerilog:
    return emitVerilog(module.get(), os);
  }
};

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "circt modular optimizer driver\n");

  // Figure out the input format if unspecified.
  if (inputFormat == InputUnspecified) {
    if (StringRef(inputFilename).endswith(".fir"))
      inputFormat = InputFIRFile;
    else if (StringRef(inputFilename).endswith(".mlir"))
      inputFormat = InputMLIRFile;
    else {
      llvm::errs() << "unknown input format: "
                      "specify with -format=fir or -format=mlir\n";
      exit(1);
    }
  }

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (failed(processBuffer(std::move(input), output->os())))
    return 1;

  output->keep();
  return 0;
}
