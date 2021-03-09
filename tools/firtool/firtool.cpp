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
#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Translation/ExportVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
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
    enableLowerTypes("enable-lower-types",
                     cl::desc("run the lower-types pass within lower-to-rtl"),
                     cl::init(false));

static cl::opt<bool>
    blackboxMemory("blackbox-memory",
                   cl::desc("Create a blackbox for all memory operations"),
                   cl::init(false));

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

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

/// Process a single buffer of the input.
static LogicalResult
processBuffer(std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              raw_ostream &os) {
  MLIRContext context;

  // Register our dialects.
  context.loadDialect<firrtl::FIRRTLDialect, rtl::RTLDialect, comb::CombDialect,
                      sv::SVDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Nothing in the parser is threaded.  Disable synchronization overhead.
  context.disableMultithreading();

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  applyPassManagerCLOptions(pm);

  OwningModuleRef module;
  if (inputFormat == InputFIRFile) {
    firrtl::FIRParserOptions options;
    options.ignoreInfoLocators = ignoreFIRLocations;
    module = importFIRRTL(sourceMgr, &context, options);

    // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
    if (!disableOptimization) {
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
    }
  } else {
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // Allow optimizations to run multithreaded.
  context.disableMultithreading(false);

  if (blackboxMemory)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createBlackBoxMemoryPass());

  // Lower if we are going to verilog or if lowering was specifically requested.
  if (lowerToRTL || outputFormat == OutputVerilog) {
    if (enableLowerTypes)
      pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass());
    pm.addPass(createLowerFIRRTLToRTLModulePass());
    pm.addNestedPass<rtl::RTLModuleOp>(createLowerFIRRTLToRTLPass());

    // If enabled, run the optimizer.
    if (!disableOptimization) {
      pm.addNestedPass<rtl::RTLModuleOp>(sv::createRTLCleanupPass());
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
    }
  }

  if (failed(pm.run(module.get())))
    return failure();

  // Finally, emit the output.
  switch (outputFormat) {
  case OutputMLIR:
    module->print(os);
    return success();
  case OutputDisabled:
    return success();
  case OutputVerilog:
    return exportVerilog(module.get(), os);
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
