//===- circt-reduce.cpp - The circt-reduce driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-reduce' tool, which is the circt analog of
// mlir-reduce, used to drive test case reduction.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "circt-reduce"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<std::string>
    outputFilename("o", cl::init("-"),
                   cl::desc("Output filename for the reduced test case"));

static cl::opt<bool>
    keepBest("keep-best", cl::init(false),
             cl::desc("Keep overwriting the output with better reductions"));

static cl::opt<std::string> testerCommand(
    "test", cl::Required,
    cl::desc("A command or script to check if output is interesting"));

static cl::list<std::string>
    testerArgs("test-arg", cl::ZeroOrMore,
               cl::desc("Additional arguments to the test"));

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that maps `firrtl.module` to `firrtl.extmodule`.
struct ModuleExternalizer {
  LogicalResult match(Operation *op) const {
    return success(isa<firrtl::FModuleOp>(op));
  }

  void rewrite(Operation *op) const {
    auto module = cast<firrtl::FModuleOp>(op);
    OpBuilder builder(module);
    builder.create<firrtl::FExtModuleOp>(
        module->getLoc(),
        module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        firrtl::getModulePortInfo(module), StringRef(),
        module.annotationsAttr());
    module->erase();
  }
};

/// Helper function that writes the current MLIR module to the configured output
/// file. Called for intermediate states if the `keepBest` options has been set,
/// or at least at the very end of the run.
static LogicalResult writeOutput(ModuleOp module) {
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    mlir::emitError(UnknownLoc::get(module.getContext()),
                    "unable to open output file \"")
        << outputFilename << "\": " << errorMessage << "\n";
    return failure();
  }
  module.print(output->os());
  output->keep();
  return success();
}

/// Execute the main chunk of work of the tool. This function reads the input
/// module and iteratively applies the reduction strategies until no options
/// make it smaller.
static LogicalResult execute(MLIRContext &context) {
  std::string errorMessage;

  // Parse the input file.
  LLVM_DEBUG(llvm::dbgs() << "Reading input\n");
  OwningModuleRef module = parseSourceFile(inputFilename, &context);
  if (!module)
    return failure();

  // Evaluate the unreduced input.
  LLVM_DEBUG(llvm::dbgs() << "Evaluating initial input\n");
  LLVM_DEBUG(llvm::dbgs() << "The test is `" << testerCommand << "`\n");
  for (auto &arg : testerArgs)
    LLVM_DEBUG(llvm::dbgs() << "  with argument `" << arg << "`\n");
  Tester tester(testerCommand, testerArgs);
  auto initialTest = tester.isInteresting(module.get());
  if (initialTest.first != Tester::Interestingness::True) {
    mlir::emitError(UnknownLoc::get(&context), "input is not interesting");
    return failure();
  }
  auto bestSize = initialTest.second;
  LLVM_DEBUG(llvm::dbgs() << "Initial module has size " << bestSize << "\n");

  // Iteratively reduce the input module by applying the current reduction
  // pattern to successively smaller subsets of the operations until we find one
  // that retains the interesting behavior.
  ModuleExternalizer pattern;
  size_t rangeBase = 0;
  size_t rangeLength = -1;
  while (rangeLength > 0) {
    LLVM_DEBUG(llvm::dbgs() << "Trying op range " << rangeBase << " to "
                            << (rangeBase + rangeLength) << "\n");

    // Apply the pattern to the subset of operations selected by `rangeBase` and
    // `rangeLength`.
    size_t opIdx = 0;
    OwningModuleRef newModule = module->clone();
    newModule->walk([&](Operation *op) {
      if (failed(pattern.match(op)))
        return;
      auto i = opIdx++;
      if (i < rangeBase || i - rangeBase >= rangeLength)
        return;
      pattern.rewrite(op);
    });
    if (opIdx == 0) {
      LLVM_DEBUG(llvm::dbgs() << "No more ops where the pattern applies\n");
      break;
    }

    // Check if this reduced module is still interesting, and its overall size
    // is smaller than what we had before.
    // LLVM_DEBUG(llvm::dbgs() << "Trying: " << **newModule << "\n");
    auto test = tester.isInteresting(newModule.get());
    if (test.first == Tester::Interestingness::True && test.second < bestSize) {
      // Make this reduced module the new baseline and reset our search strategy
      // to start again from the beginning, since this reduction may have
      // created additional opportunities.
      bestSize = test.second;
      module = std::move(newModule);
      rangeBase = 0;
      rangeLength = -1;
      LLVM_DEBUG(llvm::dbgs()
                 << "Accepting module of size " << bestSize << "\n");

      // Write the current state to disk if the user asked for it.
      if (keepBest)
        if (failed(writeOutput(module.get())))
          return failure();
    } else {
      // Try the pattern on the next `rangeLength` number of operations. If we
      // go past the end of the input, reduce the size of the chunk of
      // operations we're reducing and start again from the top.
      rangeBase += rangeLength;
      if (rangeBase >= opIdx) {
        // Exhausted all subsets of this size. Try to go smaller.
        rangeLength = opIdx / 2;
        rangeBase = 0;
      }
    }
  }

  // Write the reduced test case to the output.
  LLVM_DEBUG(llvm::dbgs() << "All reduction strategies exhausted\n");
  return writeOutput(module.get());
}

/// The entry point for the `circt-reduce` tool. Configures and parses the
/// command line options, registers all dialects with a context, and calls the
/// `execute` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse the command line options provided by the user.
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "CIRCT test case reduction tool\n");

  // Register all the dialects and create a context to work wtih.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  // Do the actual processing and use `exit` to avoid the slow teardown of the
  // context.
  exit(failed(execute(context)));
}
