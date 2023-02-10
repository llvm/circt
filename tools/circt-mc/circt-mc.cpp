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

#include "circt/LogicalEquivalence/Circuit.h"
#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "circt/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "circt/Support/Version.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-mc Options");

static cl::opt<std::string>
    moduleName1("c1",
                cl::desc("Specify a named module to verify properties over."),
                cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<int> clockBound(
    "b", cl::Required,
    cl::desc("Specify a number of clock cycles to model check up to."),
    cl::value_desc("clock cycle count"), cl::cat(mainCategory));

static cl::opt<std::string> inputFileName(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static mlir::LogicalResult checkProperty(mlir::MLIRContext &context,
                                         int bound) {

  mlir::OwningOpRef<mlir::ModuleOp> inputFile =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFileName, &context);
  if (!inputFile)
    return mlir::failure();

  // Create solver and add circuit
  // TODO: replace 'false' with a statistics option
  Solver s(&context, false);
  Solver::Circuit *circuitModel = s.addCircuit(inputFileName);

  auto exporter = std::make_unique<LogicExporter>(moduleName1, circuitModel);
  mlir::ModuleOp m = mlir::parseSourceFile<mlir::ModuleOp>(inputFileName, &context).get();
  if (failed(exporter->run(m)))
    return mlir::failure();


  // TODO: load property constraints
  // circuitModel->loadProperty();

  for (int i = 0; i < bound; i++) {
    if (!circuitModel->checkCycle(i)) {
      lec::outs() << "Failure\n";
      return mlir::failure();
    }
  }
  lec::outs() << "Success!\n";
  return mlir::success();
}

int main(int argc, char **argv) {
  // Configure the relevant command-line options.
  cl::HideUnrelatedOptions(mainCategory);
  mlir::registerMLIRContextCLOptions();

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv,
                              "circt-mc - bounded model checker\n\n"
                              "\tThis tool checks that properties hold in a "
                              "design over a symbolic bounded execution.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect, circt::seq::SeqDialect>();
  mlir::MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(checkProperty(context, clockBound)));
}
