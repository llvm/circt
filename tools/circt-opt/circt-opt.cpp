//===- circt-opt.cpp - The circt-opt driver -------------------------------===//
//
// This file implements the 'circt-opt' tool, which is the circt analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FIRRTLToLLHD/FIRRTLToLLHD.h"
#include "circt/Conversion/HandshakeToFIRRTL/HandshakeToFIRRTL.h"
#include "circt/Conversion/LLHDToLLVM/LLHDToLLVM.h"
#include "circt/Conversion/StandardToHandshake/StandardToHandshake.h"
#include "circt/Conversion/StandardToStaticLogic/StandardToStaticLogic.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/Dialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/SV/Dialect.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<bool>
    showDialects("show-dialects",
                 cl::desc("Print the list of registered dialects"),
                 cl::init(false));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operation with no registered dialects"), cl::init(false));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<StandardOpsDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<mlir::AffineDialect>();

// Register the standard passes we want.
#include "mlir/Transforms/Passes.h.inc"
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Register printer command line options.
  registerAsmPrinterCLOptions();

  // Register our dialects.
  registry.insert<firrtl::FIRRTLDialect>();
  firrtl::registerFIRRTLPasses();

  registry.insert<handshake::HandshakeOpsDialect>();
  registry.insert<staticlogic::StaticLogicDialect>();
  staticlogic::registerStandardToStaticLogicPasses();
  handshake::registerStandardToHandshakePasses();
  handshake::registerHandshakeToFIRRTLPasses();

  registry.insert<esi::ESIDialect>();
  registry.insert<llhd::LLHDDialect>();
  registry.insert<rtl::RTLDialect>();
  registry.insert<sv::SVDialect>();

  llhd::initLLHDTransformationPasses();
  llhd::initLLHDToLLVMPass();
  llhd::registerFIRRTLToLLHDPasses();

  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "circt modular optimizer driver\n");

  if (showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    for (const auto &nameAndRegistrationIt : registry) {
      llvm::outs() << nameAndRegistrationIt.first << "\n";
    }
    return 0;
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  return failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                            registry, splitInputFile, verifyDiagnostics,
                            verifyPasses, allowUnregisteredDialects));
}
