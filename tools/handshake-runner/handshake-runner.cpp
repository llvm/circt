//===- handshake-runner.cpp -----------------------------------------------===//
//
// Copyright 2021 The CIRCT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Tool which executes a restricted form of the standard dialect, and
// the handshake dialect.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "llvm/Support/InitLLVM.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/Simulation.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using cl::opt;

static cl::OptionCategory mainCategory("Application options");

static opt<std::string> inputFileName(cl::Positional, cl::desc("<input file>"),
                                      cl::init("-"), cl::cat(mainCategory));

static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

static opt<std::string>
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
  context.loadDialect<StandardOpsDialect, handshake::HandshakeOpsDialect>();
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

  return simulate(toplevelFunction, inputArgs, module, context);
}
