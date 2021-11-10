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

#include "Reduction.h"
#include "Tester.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "circt-reduce"
#define VERBOSE(X)                                                             \
  do {                                                                         \
    if (verbose) {                                                             \
      X;                                                                       \
    }                                                                          \
  } while (false)

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
    keepBest("keep-best", cl::init(true),
             cl::desc("Keep overwriting the output with better reductions"));

static cl::opt<bool> skipInitial(
    "skip-initial", cl::init(false),
    cl::desc("Skip checking the initial input for interestingness"));

static cl::opt<bool> listReductions("list", cl::init(false),
                                    cl::desc("List all available reductions"));

static cl::list<std::string> includeReductions(
    "include", cl::ZeroOrMore,
    cl::desc("Only run a subset of the available reductions"));

static cl::list<std::string>
    excludeReductions("exclude", cl::ZeroOrMore,
                      cl::desc("Do not run some of the available reductions"));

static cl::opt<std::string> testerCommand(
    "test", cl::Required,
    cl::desc("A command or script to check if output is interesting"));

static cl::list<std::string>
    testerArgs("test-arg", cl::ZeroOrMore,
               cl::desc("Additional arguments to the test"));

static cl::opt<bool> verbose("v", cl::init(true),
                             cl::desc("Print reduction progress to stderr"));

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

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

  // Gather the sets of included and excluded reductions.
  llvm::DenseSet<StringRef> inclusionSet(includeReductions.begin(),
                                         includeReductions.end());
  llvm::DenseSet<StringRef> exclusionSet(excludeReductions.begin(),
                                         excludeReductions.end());

  // Gather a list of reduction patterns that we should try.
  SmallVector<std::unique_ptr<Reduction>> patterns;
  createAllReductions(&context, [&](auto reduction) {
    auto name = reduction->getName();
    if (!inclusionSet.empty() && !inclusionSet.count(name))
      return;
    if (exclusionSet.count(name))
      return;
    patterns.push_back(std::move(reduction));
  });

  // Print the list of patterns.
  if (listReductions) {
    for (auto &pattern : patterns)
      llvm::outs() << pattern->getName() << "\n";
    return success();
  }

  // Parse the input file.
  VERBOSE(llvm::errs() << "Reading input\n");
  OwningModuleRef module = parseSourceFile(inputFilename, &context);
  if (!module)
    return failure();

  // Evaluate the unreduced input.
  VERBOSE({
    llvm::errs() << "Testing input with `" << testerCommand << "`\n";
    for (auto &arg : testerArgs)
      llvm::errs() << "  with argument `" << arg << "`\n";
  });
  Tester tester(testerCommand, testerArgs);
  auto initialTest = tester.get(module.get());
  if (!skipInitial && !initialTest.isInteresting()) {
    mlir::emitError(UnknownLoc::get(&context), "input is not interesting");
    return failure();
  }
  auto bestSize = initialTest.getSize();
  VERBOSE(llvm::errs() << "Initial module has size " << bestSize << "\n");

  // Iteratively reduce the input module by applying the current reduction
  // pattern to successively smaller subsets of the operations until we find one
  // that retains the interesting behavior.
  // ModuleExternalizer pattern;
  BitVector appliedOneShotPatterns(patterns.size(), false);
  auto lastReportTime = std::chrono::high_resolution_clock::now();
  constexpr double reportPeriod = 0.1 /*seconds*/;
  for (unsigned patternIdx = 0; patternIdx < patterns.size();) {
    Reduction &pattern = *patterns[patternIdx];
    if (pattern.isOneShot() && appliedOneShotPatterns[patternIdx]) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping one-shot `" << pattern.getName() << "`\n");
      ++patternIdx;
      continue;
    }
    VERBOSE(llvm::errs() << "Trying reduction `" << pattern.getName() << "`\n");
    size_t rangeBase = 0;
    size_t rangeLength = -1;
    bool patternDidReduce = false;
    while (rangeLength > 0) {
      // Apply the pattern to the subset of operations selected by `rangeBase`
      // and `rangeLength`.
      size_t opIdx = 0;
      OwningModuleRef newModule = module->clone();
      newModule->walk([&](Operation *op) {
        if (!pattern.match(op))
          return;
        auto i = opIdx++;
        if (i < rangeBase || i - rangeBase >= rangeLength)
          return;
        (void)pattern.rewrite(op);
      });
      if (opIdx == 0) {
        VERBOSE(llvm::errs() << "- No more ops where the pattern applies\n");
        break;
      }

      // Show some progress indication.
      VERBOSE({
        auto thisReportTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = thisReportTime - lastReportTime;
        if (elapsed.count() >= reportPeriod) {
          lastReportTime = thisReportTime;
          size_t boundLength = std::min(rangeLength, opIdx);
          size_t numDone = rangeBase / boundLength + 1;
          size_t numTotal = (opIdx + boundLength + 1) / boundLength;
          llvm::errs() << "  [" << numDone << "/" << numTotal << "; "
                       << (numDone * 100 / numTotal) << "%]\r";
        }
      });

      // Check if this reduced module is still interesting, and its overall size
      // is smaller than what we had before.
      auto shouldAccept = [&](TestCase &test) {
        if (!test.isValid())
          return false; // don't write to disk if module is busted
        if (test.getSize() >= bestSize && !pattern.acceptSizeIncrease())
          return false; // don't run test if size already bad
        return test.isInteresting();
      };
      auto test = tester.get(newModule.get());
      if (shouldAccept(test)) {
        // Make this reduced module the new baseline and reset our search
        // strategy to start again from the beginning, since this reduction may
        // have created additional opportunities.
        patternDidReduce = true;
        bestSize = test.getSize();
        VERBOSE(llvm::errs()
                << "- Accepting module of size " << bestSize << "\n");
        module = std::move(newModule);

        // We leave `rangeBase` and `rangeLength` untouched in this case. This
        // causes the next iteration of the loop to try the same pattern again
        // at the same offset. If the pattern has reached a fixed point, nothing
        // changes and we proceed. If the pattern has removed an operation, this
        // will already operate on the next batch of operations which have
        // likely moved to this point. The only exception are operations that
        // are marked as "one shot", which explicitly ask to not be re-applied
        // at the same location.
        if (pattern.isOneShot())
          rangeBase += rangeLength;

        // Write the current state to disk if the user asked for it.
        if (keepBest)
          if (failed(writeOutput(module.get())))
            return failure();
      } else {
        // Try the pattern on the next `rangeLength` number of operations.
        rangeBase += rangeLength;
      }

      // If we have gone past the end of the input, reduce the size of the chunk
      // of operations we're reducing and start again from the top.
      if (rangeBase >= opIdx) {
        rangeLength = std::min(rangeLength, opIdx) / 2;
        rangeBase = 0;
        if (rangeLength > 0)
          VERBOSE(llvm::errs()
                  << "- Trying " << rangeLength << " ops at once\n");
      }
    }

    // If this was a one-shot pattern, mark it as having been applied. This will
    // prevent further reapplication.
    if (pattern.isOneShot())
      appliedOneShotPatterns.set(patternIdx);

    // If the pattern provided a successful reduction, restart with the first
    // pattern again, since we might have uncovered additional reduction
    // opportunities. Otherwise we just keep going to try the next pattern.
    if (patternDidReduce && patternIdx > 0) {
      VERBOSE(llvm::errs() << "- Reduction `" << pattern.getName()
                           << "` was successful, starting at the top\n\n");
      patternIdx = 0;
    } else {
      ++patternIdx;
    }
  }

  // Write the reduced test case to the output.
  VERBOSE(llvm::errs() << "All reduction strategies exhausted\n");
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
