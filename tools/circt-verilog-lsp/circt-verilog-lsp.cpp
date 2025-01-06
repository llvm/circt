//===- circt-verilog-lsp.cpp - Getting Verilog into CIRCT -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility to parse Verilog and SystemVerilog input
// files. This builds on CIRCT's ImportVerilog library, which ultimately relies
// on slang to do the heavy lifting.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-verilog-lsp/CirctVerilogLspServerMain.h"

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"

// using namespace mlir;
// using namespace mlir::lsp;
// using namespace circt::lsp;

int main(int argc, char **argv) {
  llvm::cl::opt<mlir::lsp::JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(mlir::lsp::JSONStreamStyle::Standard,
                                  "standard", "usual LSP protocol"),
                       clEnumValN(mlir::lsp::JSONStreamStyle::Delimited,
                                  "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(mlir::lsp::JSONStreamStyle::Standard),
      llvm::cl::Hidden,
  };
  llvm::cl::opt<bool> litTest{
      "lit-test",
      llvm::cl::desc(
          "Abbreviation for -input-style=delimited -pretty -log=verbose. "
          "Intended to simplify lit tests"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<mlir::lsp::Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(clEnumValN(mlir::lsp::Logger::Level::Error, "error",
                                  "Error messages only"),
                       clEnumValN(mlir::lsp::Logger::Level::Info, "info",
                                  "High level execution tracing"),
                       clEnumValN(mlir::lsp::Logger::Level::Debug, "verbose",
                                  "Low level details")),
      llvm::cl::init(mlir::lsp::Logger::Level::Info),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::list<std::string> extraIncludeDirs(
      "verilog-extra-dir", llvm::cl::desc("Extra directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);
  llvm::cl::list<std::string> compilationDatabases(
      "verilog-compilation-database",
      llvm::cl::desc("Compilation YAML databases containing additional "
                     "compilation information for .verilog files"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "Verilog LSP Language Server");

  if (litTest) {
    inputStyle = mlir::lsp::JSONStreamStyle::Delimited;
    logLevel = mlir::lsp::Logger::Level::Debug;
    prettyPrint = true;
  }

  // Configure the logger.
  mlir::lsp::Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  llvm::sys::ChangeStdinToBinary();
  mlir::lsp::JSONTransport transport(stdin, llvm::outs(), inputStyle,
                                     prettyPrint);

  // Configure the servers and start the main language server.
  circt::lsp::VerilogServerOptions options(compilationDatabases,
                                           extraIncludeDirs);
  return failed(circt::lsp::CirctVerilogLspServerMain(options, transport));
}
