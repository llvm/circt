//===- querytool.cpp - The querytool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'querytool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Query/Query.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Transforms/Passes.h"
#include "circt/Translation/ExportVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// Yes this was copy pasted from firtool, don't judge me >w<
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

static cl::opt<std::string> filterInput("filter",
                                   cl::desc("the filter to check the file against"));

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool> disableOptimization("disable-opt",
                                         cl::desc("disable optimizations"));

static cl::opt<bool> inliner("inline",
                             cl::desc("Run the FIRRTL module inliner"),
                             cl::init(false));

static cl::opt<bool> lowerToHW("lower-to-hw",
                               cl::desc("run the lower-to-hw pass"));

static cl::opt<bool> enableAnnotationWarning(
    "warn-on-unprocessed-annotations",
    cl::desc("Warn about annotations that were not removed by lower-to-hw"),
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
static cl::opt<bool>
    lowerTypesV2("lower-types-v2",
                 cl::desc("run the lower-types pass within lower-to-hw"),
                 cl::init(false));

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
    inferWidths("infer-widths",
                cl::desc("run the width inference pass on firrtl"),
                cl::init(true));

static cl::opt<bool> extractTestCode("extract-test-code",
                                     cl::desc("run the extract test code pass"),
                                     cl::init(false));
static cl::opt<bool>
    grandCentral("firrtl-grand-central",
                 cl::desc("create interfaces and data/memory taps from SiFive "
                          "Grand Central annotations"),
                 cl::init(false));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<std::string>
    inputAnnotationFilename("annotation-file",
                            cl::desc("Optional input annotation file"),
                            cl::value_desc("filename"));

static cl::opt<std::string> blackBoxRootPath(
    "blackbox-path",
    cl::desc("Optional path to use as the root of black box annotations"),
    cl::value_desc("path"), cl::init(""));

static cl::opt<std::string> blackBoxRootResourcePath(
    "blackbox-resource-path",
    cl::desc(
        "Optional path to use as the root of black box resource annotations"),
    cl::value_desc("path"), cl::init(""));

#include <iostream>

/// Process a single buffer of the input.
static LogicalResult
processBuffer(std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              StringRef annotationFilename, TimingScope &ts,
              MLIRContext &context) {
  // Register our dialects.
  context.loadDialect<firrtl::FIRRTLDialect, hw::HWDialect, comb::CombDialect,
                      sv::SVDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Add the annotation file if one was explicitly specified.
  std::string annotationFilenameDetermined;
  if (!annotationFilename.empty()) {
    if (!(sourceMgr.AddIncludeFile(annotationFilename.str(), llvm::SMLoc(),
                                   annotationFilenameDetermined))) {
      llvm::errs() << "cannot open input annotation file '"
                   << annotationFilename << "': No such file or directory\n";
      return failure();
    }
  }

  OwningModuleRef module;
  if (inputFormat == InputFIRFile) {
    auto parserTimer = ts.nest("FIR Parser");
    firrtl::FIRParserOptions options;
    options.ignoreInfoLocators = ignoreFIRLocations;
    module = importFIRRTL(sourceMgr, &context, options);
  } else {
    auto parserTimer = ts.nest("MLIR Parser");
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  if (!disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createCSEPass());
  }

  // Width inference creates canonicalization opportunities.
  if (inferWidths)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferWidthsPass());

  // The input mlir file could be firrtl dialect so we might need to clean
  // things up.
  if (lowerTypes && !lowerTypesV2)
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass());
  if (!lowerTypes && lowerTypesV2)
    pm.addNestedPass<firrtl::CircuitOp>(
        firrtl::createLowerBundleVectorTypesPass());
  if (lowerTypes || lowerTypesV2) {
    auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
    // Only enable expand whens if lower types is also enabled.
    if (expandWhens)
      modulePM.addPass(firrtl::createExpandWhensPass());
  }

  // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
  if (!disableOptimization) {
    auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
    modulePM.addPass(createSimpleCanonicalizerPass());
  }

  if (inliner)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());

  if (imconstprop)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());

  if (blackBoxMemory)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createBlackBoxMemoryPass());

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
  }

  // Lower if lowering was specifically requested.
  if (lowerToHW ) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createCheckWidthsPass());
    pm.addPass(createLowerFIRRTLToHWPass(enableAnnotationWarning.getValue()));
    pm.addPass(sv::createHWMemSimImplPass());

    if (extractTestCode)
      pm.addPass(sv::createSVExtractTestCodePass());

    // If enabled, run the optimizer.
    if (!disableOptimization) {
      auto &modulePM = pm.nest<hw::HWModuleOp>();
      modulePM.addPass(sv::createHWCleanupPass());
      modulePM.addPass(createCSEPass());
      modulePM.addPass(createSimpleCanonicalizerPass());
    }
  }

  // Load the emitter options from the command line. Command line options if
  // specified will override any module options.
  applyLoweringCLOptions(module.get());

  if (failed(pm.run(module.get())))
    return failure();

  auto outputTimer = ts.nest("Output");

  // Create the filter and filter from the module
  query::Filter filter = query::Filter(filterInput);
  auto mod = module.release();
  auto vec = query::filterAsVector(filter, mod);

  for (auto v : vec) {
    for (auto *op : v) {
      std::cout << "::";
      llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<hw::HWModuleOp>([&](auto &op) {
          std::cout << op.getNameAttr().getValue().str();
        })
        .Case<hw::HWModuleExternOp>([&](hw::HWModuleExternOp &op) {
          std::cout << op.getNameAttr().getValue().str();
        })
        .Default([&](auto &op) {
          std::cout << "???";
        });
    }
    std::cout << std::endl;
  }

  return success();
}

/// This implements the top-level logic for the querytool command, invoked once
/// command line options are parsed and LLVM/MLIR are all set up and ready to
/// go.
static LogicalResult executeQuerytool(MLIRContext &context) {
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

  return processBuffer(std::move(input), inputAnnotationFilename, ts, context);
}

/// Main driver for querytool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeQuerytool'.  This is set up
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
  cl::ParseCommandLineOptions(argc, argv, "circt modular optimizer driver\n");

  // -disable-opt turns off constant propagation (unless it was explicitly
  // enabled).
  if (disableOptimization && imconstprop.getNumOccurrences() == 0)
    imconstprop = false;

  MLIRContext context;

  // Do the guts of the querytool process.
  auto result = executeQuerytool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
