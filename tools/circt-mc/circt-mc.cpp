//===- circt-mc.cpp - The circt-mc model checker --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-mc' tool
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/InitAllDialects.h"
#include "circt/LogicalEquivalence/Circuit.h"
#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "circt/Support/Version.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"

namespace cl = llvm::cl;
using namespace circt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-mc Options");

static cl::opt<std::string>
    moduleName("module",
               cl::desc("Specify a named module to verify properties over."),
               cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<int> clockBound(
    "b", cl::Required,
    cl::desc("Specify a number of clock cycles to model check up to."),
    cl::value_desc("clock cycle count"), cl::cat(mainCategory));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<std::string> inputFileName(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static LogicalResult checkProperty(ModuleOp input, MLIRContext &context,
                                   int bound) {

  // Create solver and add circuit
  // TODO: replace 'false' with a statistics option
  Solver s(&context, false);
  Solver::Circuit *circuitModel = s.addCircuit(inputFileName);

  auto exporter = std::make_unique<LogicExporter>(moduleName, circuitModel);
  if (failed(exporter->run(input)))
    return failure();

  for (int i = 0; i < bound; i++) {
    if (!circuitModel->checkCycle(i)) {
      lec::outs() << "Failure\n";
      return input->emitError("Properties do not hold on module.");
    }
  }
  lec::outs() << "Success!\n";
  return success();
}

static LogicalResult processBuffer(MLIRContext &context,
                                   llvm::SourceMgr &sourceMgr) {
  OwningOpRef<ModuleOp> module;
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return failure();

  return checkProperty(module.get(), context, clockBound);
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult
processInputSplit(MLIRContext &context,
                  std::unique_ptr<llvm::MemoryBuffer> buffer) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, sourceMgr);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, sourceMgr);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult processInput(MLIRContext &context,
                                  std::unique_ptr<llvm::MemoryBuffer> input) {
  if (!splitInputFile)
    return processInputSplit(context, std::move(input));

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<llvm::MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, std::move(buffer));
      },
      llvm::outs());
}

int main(int argc, char **argv) {
  // Configure the relevant command-line options.
  cl::HideUnrelatedOptions(mainCategory);
  registerMLIRContextCLOptions();

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv,
                              "circt-mc - bounded model checker\n\n"
                              "\tThis tool checks that properties hold in a "
                              "design over a symbolic bounded execution.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::verif::VerifDialect>();
  MLIRContext context(registry);

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFileName, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(false);
  }

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(processInput(context, std::move(input))));
}
