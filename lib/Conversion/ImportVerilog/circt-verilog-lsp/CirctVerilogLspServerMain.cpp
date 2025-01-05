//===- .cpp - MLIR PDLL Language Server main ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-verilog-lsp/CirctVerilogLspServerMain.h"
#include "LSPServer.h"
#include "VerilogServer.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Error.h"

// using namespace mlir;
// using namespace mlir::lsp;
// using namespace circt::lsp;

llvm::LogicalResult circt::CirctVerilogLspServerMain(
    const circt::lsp::VerilogServer::Options &options,
    mlir::lsp::JSONTransport &transport) {
  circt::lsp::VerilogServer server(options);
  return circt::lsp::runVerilogLSPServer(server, transport);
}
