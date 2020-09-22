//===- llhd-sim.cpp - LLHD simulator tool -----------------------*- C++ -*-===//
//
// This file implements a command line tool to run LLHD simulation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LLHDToLLVM/LLHDToLLVM.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Simulator/Engine.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input-file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<int> nSteps("n", cl::desc("Set the maximum number of steps"),
                           cl::value_desc("max-steps"));

static cl::opt<bool>
    dumpLLVMDialect("dump-llvm-dialect",
                    cl::desc("Dump the LLVM IR dialect module"));

static cl::opt<bool> dumpLLVMIR("dump-llvm-ir",
                                cl::desc("Dump the LLVM IR module"));

static cl::opt<bool> dumpMLIR("dump-mlir",
                              cl::desc("Dump the original MLIR module"));

static cl::opt<bool> dumpLayout("dump-layout",
                                cl::desc("Dump the gathered instance layout"));

static cl::opt<std::string> root(
    "root",
    cl::desc("Specify the name of the entity to use as root of the design"),
    cl::value_desc("root_name"), cl::init("root"));
static cl::alias rootA("r", cl::desc("Alias for -root"), cl::aliasopt(root));

static int parseMLIR(MLIRContext &context, OwningModuleRef &module) {
  module = parseSourceFile(inputFilename, &context);
  if (!module)
    return 1;
  return 0;
}

static int dumpLLVM(ModuleOp module, MLIRContext &context) {
  if (dumpLLVMDialect) {
    module.dump();
    llvm::errs() << "\n";
    return 0;
  }

  // Translate the module, that contains the LLVM dialect, to LLVM IR.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int main(int argc, char **argv) {
  llhd::initLLHDToLLVMPass();

  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "LLHD simulator\n");

  // Set up the input and output files.
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

  // Parse the input file.
  MLIRContext context;
  OwningModuleRef module;

  // Load the dialects
  context
      .loadDialect<llhd::LLHDDialect, LLVM::LLVMDialect, StandardOpsDialect>();

  if (parseMLIR(context, module))
    return 1;

  if (dumpMLIR) {
    module->dump();
    llvm::errs() << "\n";
    return 0;
  }

  llhd::sim::Engine engine(output->os(), *module, context, root);

  if (dumpLLVMDialect || dumpLLVMIR) {
    return dumpLLVM(engine.getModule(), context);
  }

  if (dumpLayout) {
    engine.dumpStateLayout();
    engine.dumpStateSignalTriggers();
    return 0;
  }

  engine.simulate(nSteps);

  output->keep();
  return 0;
}
