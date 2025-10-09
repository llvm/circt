//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"
#include "LSPServer.h"
#include "VerilogServerImpl/VerilogServer.h"
#include "llvm/Support/LSP/Transport.h"

llvm::LogicalResult circt::lsp::CirctVerilogLspServerMain(
    const circt::lsp::LSPServerOptions &lspOptions,
    const circt::lsp::VerilogServerOptions &options,
    llvm::lsp::JSONTransport &transport) {
  circt::lsp::VerilogServer server(options);
  return circt::lsp::runVerilogLSPServer(lspOptions, server, transport);
}
