//===- CirctVerilogLspServerMain.h - CIRCT Verilog LSP Server main -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for circt-verilog-lsp-server for when built as standalone
// binary.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_CIRCTVERILOGLSPSERVERMAIN_H
#define CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_CIRCTVERILOGLSPSERVERMAIN_H
#include "circt/Support/Namespace.h"
#include <string>
#include <vector>

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
struct VerilogServerOptions {
  VerilogServerOptions(const std::vector<std::string> &compilationDatabases,
                       const std::vector<std::string> &extraDirs)
      : compilationDatabases(compilationDatabases), extraDirs(extraDirs) {}

  /// The filenames for databases containing compilation commands for PDLL
  /// files passed to the server.
  const std::vector<std::string> &compilationDatabases;

  /// Additional list of include directories to search.
  const std::vector<std::string> &extraDirs;
};
// namespace lsp

/// Implementation for tools like `circt-verilog-lsp-server`.
llvm::LogicalResult
CirctVerilogLspServerMain(const VerilogServerOptions &options,
                          mlir::lsp::JSONTransport &transport);

} // namespace lsp
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_CIRCTVERILOGLSPSERVERMAIN_H