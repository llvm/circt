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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

#include <string>

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"));

static cl::opt<bool> forceOutput("f",
                                 cl::desc("Replace output file if it exists"),
                                 cl::init(false));

static constexpr const char toolName[] = "circt-as";

static LogicalResult emitError(const Twine &err) {
  WithColor::error(errs(), toolName) << err << "\n";
  return failure();
}

struct LeakModule {
  OwningOpRef<ModuleOp> module;
  ~LeakModule() { (void)module.release(); }
};

static LogicalResult execute(MLIRContext &context) {
  // Figure out where we're writing the output.
  if (outputFilename.empty()) {
    StringRef input = inputFilename;
    if (input == "-")
      return emitError("no output filename given");
    SmallString<64> outputStr{input};
    sys::path::replace_extension(outputStr, "mlirbc");
    outputFilename = outputStr.str().str();
  }
  if (inputFilename == outputFilename)
    return emitError("input and output must be different files");
  if (sys::fs::exists(outputFilename)) {
    // Reject directory path.
    if (sys::fs::is_directory(outputFilename))
      return emitError("output path is a directory");
    // Reject file path if `-f` is not specified.
    if (!forceOutput)
      return emitError("output file exists.  Use -f flag to force overwrite");
  }

  // Open output for writing, early error if problem.
  std::string err;
  auto output = openOutputFile(outputFilename, &err);
  if (!output)
    return emitError(err);

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

  cl::ParseCommandLineOptions(argc, argv, "CIRCT assembler\n");

  MLIRContext context;
  context.appendDialectRegistry(registry);

  // Do the guts of the process.
  auto result = execute(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
