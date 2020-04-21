//===- firtool.cpp - The firtool utility for working with .fir files ------===//
//
// This file implements 'firtool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
//#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
//#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
//#include "mlir/Transforms/Passes.h"
#include "spt/Dialect/FIRRTL/Dialect.h"
#include "spt/EmitVerilog.h"
#include "spt/FIRParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace spt;

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
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  OwningModuleRef module;
  if (inputFormat == InputFIRFile)
    module = parseFIRFile(sourceMgr, &context);
  else {
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile(sourceMgr, &context);
  }
  if (!module)
    return failure();

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

  // Register the standard passes we want.
  //#define GEN_PASS_REGISTRATION_Canonicalizer
  //#define GEN_PASS_REGISTRATION_CSE
  //#include "mlir/Transforms/Passes.h.inc"

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Register FIRRTL stuff.
  registerDialect<firrtl::FIRRTLDialect>();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "spt modular optimizer driver\n");

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
