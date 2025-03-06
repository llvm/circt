//===----------------------------------------------------------------------===//
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
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
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
  VerilogServerOptions(const std::vector<std::string> &libDirs,
                       const std::vector<std::string> &extraSourceLocationDirs,
                       int32_t hoverContextLineCount)
      : libDirs(libDirs), extraSourceLocationDirs(extraSourceLocationDirs),
        hoverContextLineCount(hoverContextLineCount) {}
  /// Additional list of RTL directories to search.
  const std::vector<std::string> &libDirs;

  /// Additional list of external source directories to search.
  const std::vector<std::string> &extraSourceLocationDirs;

  /// Number of lines to include in the hover documentation.
  const int32_t hoverContextLineCount;
};
// namespace lsp

/// Implementation for tools like `circt-verilog-lsp-server`.
llvm::LogicalResult
CirctVerilogLspServerMain(const VerilogServerOptions &options,
                          mlir::lsp::JSONTransport &transport);

} // namespace lsp
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_CIRCTVERILOGLSPSERVERMAIN_H
