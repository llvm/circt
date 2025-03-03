//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_LSPSERVER_H
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_LSPSERVER_H

#include <memory>
namespace llvm {
struct LogicalResult;
} // namespace llvm
namespace mlir {
namespace lsp {
class JSONTransport;
} // namespace lsp
} // namespace mlir

namespace circt {
namespace lsp {
class VerilogServer;

/// Run the main loop of the LSP server using the given Verilog server and
/// transport.
llvm::LogicalResult runVerilogLSPServer(VerilogServer &server,
                                        mlir::lsp::JSONTransport &transport);

} // namespace lsp
} // namespace circt

#endif // LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_LSPSERVER_H
