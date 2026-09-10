//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VerilogServer class, which is responsible for
// managing the state of the Verilog server. VerilogServer keeps track of the
// contents of all open text documents, and each document has a slang
// compilation result.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGSERVERCONTEXT_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGSERVERCONTEXT_H_

namespace circt {
namespace lsp {

struct VerilogServerOptions;

// A global context carried around by the server.
struct VerilogServerContext {
  VerilogServerContext(const circt::lsp::VerilogServerOptions &options)
      : options(options) {}
  const circt::lsp::VerilogServerOptions &options;
};
} // namespace lsp
} // namespace circt

#endif
