//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility to run CIRCT Verilog LSP server.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Transport.h"
#include "llvm/Support/Program.h"

using namespace llvm::lsp;

int main(int argc, char **argv) {
  //===--------------------------------------------------------------------===//
  // LSP options
  //===--------------------------------------------------------------------===//

  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      llvm::cl::init(Logger::Level::Info),
  };

  llvm::cl::opt<llvm::lsp::JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(llvm::lsp::JSONStreamStyle::Standard,
                                  "standard", "usual LSP protocol"),
                       clEnumValN(llvm::lsp::JSONStreamStyle::Delimited,
                                  "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(llvm::lsp::JSONStreamStyle::Standard),
      llvm::cl::Hidden,
  };

  //===--------------------------------------------------------------------===//
  // Include paths
  //===--------------------------------------------------------------------===//

  llvm::cl::list<std::string> libDirs{
      "y",
      llvm::cl::desc(
          "Library search paths, which will be searched for missing modules"),
      llvm::cl::value_desc("dir"), llvm::cl::Prefix};
  llvm::cl::alias libDirsLong{"libdir", llvm::cl::desc("Alias for -y"),
                              llvm::cl::aliasopt(libDirs), llvm::cl::NotHidden};

  //===--------------------------------------------------------------------===//
  // Testing
  //===--------------------------------------------------------------------===//

  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> litTest{
      "lit-test",
      llvm::cl::desc(
          "Abbreviation for -input-style=delimited -pretty -log=verbose. "
          "Intended to simplify lit tests"),
      llvm::cl::init(false),
  };

  llvm::cl::ParseCommandLineOptions(argc, argv, "Verilog LSP Language Server");

  if (litTest) {
    inputStyle = llvm::lsp::JSONStreamStyle::Delimited;
    logLevel = llvm::lsp::Logger::Level::Debug;
    prettyPrint = true;
  }

  // Configure the logger.
  llvm::lsp::Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  (void)llvm::sys::ChangeStdinToBinary();
  llvm::lsp::JSONTransport transport(stdin, llvm::outs(), inputStyle,
                                     prettyPrint);

  // Configure the servers and start the main language server.
  circt::lsp::VerilogServerOptions options(libDirs);
  return failed(circt::lsp::CirctVerilogLspServerMain(options, transport));
}
