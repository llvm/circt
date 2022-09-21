//===- circt-dis.cpp - Convert MLIRBC to MLIR -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert MLIR bytecode (MLIRBC) input to MLIR textual format.
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

#include <string>

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input .mlirbc file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Override output filename"),
                                           cl::value_desc("filename"));

static constexpr const char toolName[] = "circt-dis";

/// Print error and return failure.
static LogicalResult emitError(const Twine &err) {
  WithColor::error(errs(), toolName) << err << "\n";
  return failure();
}

namespace {
/// Wrapper for OwningOpRef that leaks the module.
struct LeakModule {
  OwningOpRef<ModuleOp> module;
  ~LeakModule() { (void)module.release(); }
};
} // end anonymous namespace

static LogicalResult execute(MLIRContext &context) {
  // Figure out where we're writing the output.
  if (outputFilename.empty()) {
    StringRef input = inputFilename;
    if (input == "-")
      outputFilename = "-";
    else {
      input.consume_back(".mlirbc");
      outputFilename = (input + ".mlir").str();
    }
  }

  // Open output for writing, early error if problem.
  std::string err;
  auto output = openOutputFile(outputFilename, &err);
  if (!output)
    return emitError(err);

  // Read input MLIR bytecode.
  SourceMgr srcMgr;
  SourceMgrDiagnosticHandler handler(srcMgr, &context);

  LeakModule leakMod{
      parseSourceFile<ModuleOp>(inputFilename, srcMgr, &context)};
  auto &module = leakMod.module;
  if (!module)
    return failure();

  // Write MLIR.
  module->print(output->os());
  output->keep();

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  mlir::DialectRegistry registry;

  circt::registerAllDialects(registry);

  cl::ParseCommandLineOptions(argc, argv, "CIRCT .mlirbc -> .mlir disassembler\n");

  MLIRContext context;
  context.appendDialectRegistry(registry);

  // Do the guts of the process.
  auto result = execute(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
