//===- circt-as.cpp - Convert MLIR to MLIRBC ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert MLIR textual input to MLIR bytecode (MLIRBC).
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeWriter.h"
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
    inputFilename(cl::Positional, cl::desc("<input .mlir file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Override output filename"),
                                           cl::value_desc("filename"));

static cl::opt<bool> forceOutput("f",
                                 cl::desc("Enable binary output on terminals"),
                                 cl::init(false));

static constexpr const char toolName[] = "circt-as";

/// Print error and return failure.
static LogicalResult emitError(const Twine &err) {
  WithColor::error(errs(), toolName) << err << "\n";
  return failure();
}

/// Check output stream before writing bytecode to it.
/// Warn and return true if output is known to be displayed.
static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    WithColor::warning(errs(), toolName)
        << "You're attempting to print out a bytecode file.\n"
           "This is inadvisable as it may cause display problems. If\n"
           "you REALLY want to taste MLIR bytecode first-hand, you\n"
           "can force output with the `-f' option.\n\n";
    return true;
  }
  return false;
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
      input.consume_back(".mlir");
      outputFilename = (input + ".mlirbc").str();
    }
  }

  // Open output for writing, early error if problem.
  std::string err;
  auto output = openOutputFile(outputFilename, &err);
  if (!output)
    return emitError(err);

  if (!forceOutput && checkBytecodeOutputToConsole(output->os()))
    return emitError("not writing bytecode to console");

  // Read input MLIR.
  SourceMgr srcMgr;
  SourceMgrDiagnosticHandler handler(srcMgr, &context);

  LeakModule leakMod{
      parseSourceFile<ModuleOp>(inputFilename, srcMgr, &context)};
  auto &module = leakMod.module;
  if (!module)
    return failure();

  // Write bytecode.
  writeBytecodeToFile(*module, output->os());
  output->keep();

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  mlir::DialectRegistry registry;

  circt::registerAllDialects(registry);

  cl::ParseCommandLineOptions(argc, argv, "CIRCT .mlir -> .mlirbc assembler\n");

  MLIRContext context;
  context.appendDialectRegistry(registry);

  // Do the guts of the process.
  auto result = execute(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
