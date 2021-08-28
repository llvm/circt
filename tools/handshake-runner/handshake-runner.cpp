//===- handshake-runner.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tool which executes a restricted form of the standard dialect, and
// the handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/Simulation.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

static cl::opt<std::string>
    toplevelFunction("toplevelFunction", cl::Optional,
                     cl::desc("The toplevel function to execute"),
                     cl::init("main"), cl::cat(mainCategory));

// static opt<bool> runStats("runStats", cl::Optional,
//                           cl::desc("Print Execution Statistics"),
//                           cl::init(false), cl::cat(mainCategory));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "MLIR Standard dialect runner\n\n"
      "This application executes a function in the given MLIR module\n"
      "Arguments to the function are passed on the command line and\n"
      "results are returned on stdout.\n"
      "Memref types are specified as a comma-separated list of values.\n");

  auto file_or_err = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module.
  mlir::MLIRContext context;
  context.loadDialect<StandardOpsDialect, memref::MemRefDialect,
                      handshake::HandshakeDialect>();
  SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), SMLoc());
  mlir::OwningModuleRef module(mlir::parseSourceFile(source_mgr, &context));
  if (!module)
    return 1;

  mlir::Operation *mainP = module->lookupSymbol(toplevelFunction);
  // The toplevel function can accept any number of operands, and returns
  // any number of results.
  if (!mainP) {
    errs() << "Toplevel function " << toplevelFunction << " not found!\n";
    return 1;
  }

  return handshake::simulate(toplevelFunction, inputArgs, module, context);
}
